from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
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


def generate_prediction_matrix_plot(input_sequence, input_masks, prediction_sequence, target_sequence, target_masks, date_str, index_offset, lat, lon, output_path):
    sequence_length = input_sequence.shape[0]
    predict_steps = prediction_sequence.shape[0]
    total_steps = sequence_length + predict_steps

    # Define the log normalization factor matching your preprocessing pipeline
    clip_value = 1500.0
    global_log_max = np.log1p(clip_value)

    # Convert normalized frames to absolute metrics by inverting log1p
    input_mm = np.expm1(input_sequence * global_log_max)
    # Clip model predictions safely to the normalized [0, 1] interval before inversion
    prediction_clipped = np.clip(prediction_sequence, 0, 1)
    prediction_mm = np.expm1(prediction_clipped * global_log_max)
    target_mm = np.expm1(target_sequence * global_log_max)

    input_mm = input_mm.copy()
    prediction_mm = prediction_mm.copy()
    target_mm = target_mm.copy()

    # Enforce masking boundaries over invalid blackout coordinates
    input_mm[input_masks == 0] = -1
    prediction_mm[target_masks == 0] = -1
    target_mm[target_masks == 0] = -1
    error_mm = prediction_mm - target_mm

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    init_time = np.datetime64(date_str)

    # Base grid height scaling allocation
    fig, axes = plt.subplots(
        predict_steps, 3,
        figsize=(18, 5.0 * predict_steps),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    if predict_steps == 1:
        axes = axes.reshape(1, 3)

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.95, wspace=0.08, hspace=0.22)

    # Render loop for forecasting rows
    for step in range(predict_steps):
        step_minutes = (step + 1) * 5
        valid_time = init_time + np.timedelta64(step_minutes, "m")
        time_str = np.datetime_as_string(valid_time, unit='m').replace('T', ' ')

        # 1. Ground Truth Subplot Column
        mesh_gt = axes[step, 0].pcolormesh(
            lon, lat, target_mm[step],
            cmap=rainfall_cmap, norm=rainfall_norm, shading="auto", transform=ccrs.PlateCarree()
        )
        axes[step, 0].coastlines(resolution="50m", linewidth=0.8)
        axes[step, 0].add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
        axes[step, 0].set_extent(extent)
        axes[step, 0].set_title(f"Ground Truth\n{time_str} (+{step_minutes} min)", fontsize=17, pad=4)

        # 2. Prediction Model Subplot Column
        mesh_pred = axes[step, 1].pcolormesh(
            lon, lat, prediction_mm[step],
            cmap=rainfall_cmap, norm=rainfall_norm, shading="auto", transform=ccrs.PlateCarree()
        )
        axes[step, 1].coastlines(resolution="50m", linewidth=0.8)
        axes[step, 1].add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
        axes[step, 1].set_extent(extent)
        axes[step, 1].set_title(f"Baseline CNN Forecast\n{time_str} (+{step_minutes} min)", fontsize=17, pad=4)

        # 3. Residual Error Spatial Matrix Column
        mesh_error = axes[step, 2].pcolormesh(
            lon, lat, error_mm[step],
            cmap=error_cmap, norm=error_norm, shading="auto", transform=ccrs.PlateCarree()
        )
        axes[step, 2].coastlines(resolution="50m", linewidth=0.8)
        axes[step, 2].add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
        axes[step, 2].set_extent(extent)
        axes[step, 2].set_title(f"Error\n{time_str} (+{step_minutes} min)", fontsize=17, pad=4)

    cax_bot_rain = fig.add_axes([0.02, 0.035, 0.62, 0.015])
    cbar_bot_rain = fig.colorbar(mesh_pred, cax=cax_bot_rain, orientation="horizontal", ticks=rainfall_bounds)
    cbar_bot_rain.set_label("Rainfall Depth Accumulation (1/100 mm)  |  -1 : Missing Scan Element Flag", fontsize=16, labelpad=8)
    cbar_bot_rain.ax.tick_params(labelsize=14)

    cax_bot_error = fig.add_axes([0.68, 0.035, 0.30, 0.015])
    cbar_bot_error = fig.colorbar(mesh_error, cax=cax_bot_error, orientation="horizontal")
    cbar_bot_error.set_label("Forecast Deviation Error (Pred - True)", fontsize=16, labelpad=8)
    cbar_bot_error.ax.tick_params(labelsize=14)

    init_time_str = np.datetime_as_string(init_time, unit='m').replace('T', ' ')
    fig.suptitle(
        f"Inference Sequence Matrix \n Sample {index_offset}: {init_time_str} UTC  |  MeteoNet Radar NW Zone",
        fontsize=20, weight='bold', y=0.993
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=200)
    plt.close("all")


def main():
    # Fetch environment, paths, and config overrides instantly
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

    # Dynamically build output sub-directory and file path inside the active test run folder
    sub_dir = params["test_dir"] / params["config"]["output"]["sub_dir_grid"]
    sub_dir.mkdir(parents=True, exist_ok=True)
    output_path = sub_dir / f"prediction_{sample['resolved_global_idx']}.png"

    print(f"Ingesting model forecasts out of matrix archive directory path: {params['predictions_dir']}")
    print(f"Rendering spatiotemporal map layout metrics for window sequence offset: {sample['resolved_global_idx']}")

    generate_prediction_matrix_plot(
        input_sequence=sample["inputs"],
        input_masks=sample["input_masks"],
        prediction_sequence=sample["predictions"],
        target_sequence=sample["targets"],
        target_masks=sample["target_masks"],
        date_str=sample["date"],
        index_offset=sample["resolved_global_idx"],
        lat=lat,
        lon=lon,
        output_path=output_path
    )
    print(f"Figure successfully saved to destination output path: {output_path}")


if __name__ == "__main__":
    main()
    