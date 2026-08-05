from pathlib import Path
import shutil
import tempfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # Secure headless mode for automated pipeline execution
import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocess import resize_array
from src.utils.config import load_plotting_environment
from src.utils.plotting_styles import (
    error_cmap,
    error_norm,
    rainfall_bounds,
    rainfall_cmap,
    rainfall_norm,
)
from src.utils.predictions_io import load_prediction_sample


def generate_single_horizon_frame(prediction_step, target_step, mask_step, time_str, step_label_str, index_offset, lat, lon, extent, temp_output_path, is_historical=False):
    # Convert normalized single step frames to absolute millimeter metrics (1/100 mm per 5 min)
    clip_value = 1500.0
    global_log_max = np.log1p(clip_value)
    pred_clipped = np.clip(prediction_step, 0, 1)
    pred_mm = np.expm1(pred_clipped * global_log_max)
    true_mm = np.expm1(target_step * global_log_max)
    pred_mm = pred_mm.copy()
    true_mm = true_mm.copy()


    # Ensure arrays are strictly 2D spatial layouts [Height, Width]
    if pred_mm.ndim == 3: pred_mm = np.squeeze(pred_mm, axis=0)
    if true_mm.ndim == 3: true_mm = np.squeeze(true_mm, axis=0)
    if mask_step.ndim == 3: mask_step = np.squeeze(mask_step, axis=0)

    # Enforce masking boundaries over invalid blackout coordinates
    pred_mm[mask_step == 0] = -1
    true_mm[mask_step == 0] = -1
    
    if is_historical:
        # During history lookbacks, fill middle and right columns with blank missing elements (-1)
        pred_mm.fill(-1)
        error_mm = np.full_like(pred_mm, -1)
    else:
        error_mm = pred_mm - true_mm

    # Render exactly 1 row spanning 3 side-by-side columns
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),  
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = axes.reshape(1, 3)

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.85, wspace=0.08)

    # Column 1: Ground Truth
    mesh_gt = axes[0, 0].pcolormesh(
        lon, lat, true_mm,
        cmap=rainfall_cmap, norm=rainfall_norm, shading="auto", transform=ccrs.PlateCarree()
    )
    axes[0, 0].coastlines(resolution="50m", linewidth=0.8)
    axes[0, 0].add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axes[0, 0].set_extent(extent)
    axes[0, 0].set_title(f"Ground Truth\n{time_str} ({step_label_str})", fontsize=14, pad=4)

    # Column 2: Baseline CNN Forecast
    mesh_pred = axes[0, 1].pcolormesh(
        lon, lat, pred_mm,
        cmap=rainfall_cmap, norm=rainfall_norm, shading="auto", transform=ccrs.PlateCarree()
    )
    axes[0, 1].coastlines(resolution="50m", linewidth=0.8)
    axes[0, 1].add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axes[0, 1].set_extent(extent)
    axes[0, 1].set_title(f"Baseline CNN Forecast\n{time_str} ({step_label_str})", fontsize=14, pad=4)

    # Column 3: Spatial Error Residual
    mesh_error = axes[0, 2].pcolormesh(
        lon, lat, error_mm,
        cmap=error_cmap, norm=error_norm, shading="auto", transform=ccrs.PlateCarree()
    )
    axes[0, 2].coastlines(resolution="50m", linewidth=0.8)
    axes[0, 2].add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axes[0, 2].set_extent(extent)
    axes[0, 2].set_title(f"Error\n{time_str} ({step_label_str})", fontsize=14, pad=4)

    cax_bot_rain = fig.add_axes([0.02, 0.10, 0.62, 0.035])
    cbar_bot_rain = fig.colorbar(mesh_pred, cax=cax_bot_rain, orientation="horizontal", ticks=rainfall_bounds)
    cbar_bot_rain.set_label("Rainfall Depth Accumulation (1/100 mm)  |  -1 : Missing Scan Element Flag", fontsize=11, labelpad=4)
    cbar_bot_rain.ax.tick_params(labelsize=10)

    cax_bot_error = fig.add_axes([0.68, 0.10, 0.30, 0.035])
    cbar_bot_error = fig.colorbar(mesh_error, cax=cax_bot_error, orientation="horizontal")
    cbar_bot_error.set_label("Forecast Deviation Error (Pred - True)", fontsize=11, labelpad=4)
    cbar_bot_error.ax.tick_params(labelsize=10)

    phase_lbl = "History Phase" if is_historical else "Forecast Phase"
    fig.suptitle(
        f"Dynamic Inference Loop [{phase_lbl}]  |  Sample {index_offset}  |  Offset: {step_label_str}  |  MeteoNet Radar NW Zone",
        fontsize=16, weight='bold', y=0.973
    )

    plt.savefig(temp_output_path, dpi=150)
    plt.close("all")


def main():
    params = load_plotting_environment()

    # Load data smoothly via our shared dual-query backend loader utility
    sample = load_prediction_sample(
        predictions_dir=params["predictions_dir"], 
        query=params["query"]
    )

    # Base coordinates extraction tracking
    raw_coords_path = Path(params["radar_coords"])
    if not raw_coords_path.exists():
        raise FileNotFoundError(f"Missing base coordinate archive at {raw_coords_path}.")
        
    coords = np.load(raw_coords_path, allow_pickle=True)
    lat_raw = coords['lats']
    lon_raw = coords['lons']

    target_height = sample["predictions"].shape[-2]
    target_width = sample["predictions"].shape[-1]

    lat = resize_array(lat_raw, target_height, target_width)
    lon = resize_array(lon_raw, target_height, target_width)

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # Extract sequences out of the utility dict mapping variables
    input_sequence = sample["inputs"]
    input_masks = sample["input_masks"]
    prediction_sequence = sample["predictions"]
    target_sequence = sample["targets"]
    target_masks = sample["target_masks"]
    init_time = sample["date"]

    sequence_length = input_sequence.shape[0]
    predict_steps = prediction_sequence.shape[0]
    
    temp_dir = Path(tempfile.mkdtemp(prefix="radar_gif_"))
    frame_paths = []
    frame_count = 0

    try:
        # 1) Build the Lookback History Animation Frames
        for step in range(sequence_length):
            offset_minutes = (sequence_length - 1 - step) * 5
            current_timestamp = init_time - np.timedelta64(offset_minutes, "m")
            time_str = np.datetime_as_string(current_timestamp, unit='m').replace('T', ' ')
            display_label = "0 min" if offset_minutes == 0 else f"-{offset_minutes} min"

            temp_frame_path = temp_dir / f"frame_{frame_count:03d}.png"
            frame_paths.append(temp_frame_path)
            frame_count += 1

            pred_step = np.zeros_like(input_sequence[step])
            target_step = input_sequence[step]
            mask_step = input_masks[step]

            generate_single_horizon_frame(
                prediction_step=pred_step,
                target_step=target_step,
                mask_step=mask_step,
                time_str=time_str,
                step_label_str=display_label,
                index_offset=sample['resolved_global_idx'],
                lat=lat, lon=lon, extent=extent,
                temp_output_path=temp_frame_path,
                is_historical=True
            )

        # 2) Build the Forecast Horizon Animation Frames (+5 min to +30 min)
        for step in range(predict_steps):
            step_minutes = (step + 1) * 5
            current_timestamp = init_time + np.timedelta64(step_minutes, "m")
            time_str = np.datetime_as_string(current_timestamp, unit='m').replace('T', ' ')
            
            temp_frame_path = temp_dir / f"frame_{frame_count:03d}.png"
            frame_paths.append(temp_frame_path)
            frame_count += 1

            pred_step = prediction_sequence[step]
            target_step = target_sequence[step]
            mask_step = target_masks[step]

            generate_single_horizon_frame(
                prediction_step=pred_step,
                target_step=target_step,
                mask_step=mask_step,
                time_str=time_str,
                step_label_str=f"+{step_minutes} min",
                index_offset=sample['resolved_global_idx'],
                lat=lat, lon=lon, extent=extent,
                temp_output_path=temp_frame_path,
                is_historical=False
            )

        # Dynamically map the compilation path targets into the active test run folder structure
        sub_dir = params["test_dir"] / params["config"]["output"]["sub_dir_gif"]
        sub_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = sub_dir / f"forecast_sequence_{sample['resolved_global_idx']}.gif"

        print(f"Compiling {frame_count} sequential frames into a fluid loop at {params['fps']} FPS...")
        images = []
        for path in frame_paths:
            images.append(imageio.imread(path))
            
        imageio.mimsave(final_output_path, images, fps=params["fps"], loop=0)
        print(f"GIF animation successfully compiled and saved to: {final_output_path}")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
    