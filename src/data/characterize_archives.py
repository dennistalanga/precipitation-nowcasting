import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from src.data.dataset import StratificationRadarDataset
from src.utils.config import load_stratification_environment
from src.utils.logger import get_logger, print_configuration


def calculate_archive_descriptors(
    batch, rain_thresh, moderate_thresh, heavy_thresh, min_coverage,
    min_dist, max_dist, device, stride
):
    # Dynamically extract spatial shapes directly from data configurations
    # batch shape from DataLoader is (1, Strided_Frames, 1, H, W)
    data = batch["data_tensor"].to(device, non_blocking=True).squeeze(0)  # Shape: (Strided_Frames, 1, H, W)
    masks = batch["mask_tensor"].to(device, non_blocking=True).squeeze(0) # Shape: (Strided_Frames, 1, H, W)
    
    # Skip if temporal downsampling left fewer than 2 frames
    if data.shape[0] < 2:
        return {
            "rain_coverage_light": 0.0, "rain_coverage_moderate": 0.0, "rain_coverage_heavy": 0.0,
            "mean_intensity": 0.0, "max_intensity": 0.0, "spatial_variance": 0.0,
            "temporal_variability": 0.0, "average_motion": 0.0, "rainy_frames_count": 0
        }

    # Remove the 1-channel dimension cleanly for spatial mathematics
    data = data.squeeze(1)   # Shape: (Strided_Frames, H, W)
    masks = masks.squeeze(1) # Shape: (Strided_Frames, H, W)

    # Extract dynamic dimensions from the processed data tensor
    _, height, width = data.shape

    # Run Coverage Analytics
    total_pixels = masks.sum().item()
    if total_pixels == 0:
        return {
            "rain_coverage_light": 0.0, "rain_coverage_moderate": 0.0, "rain_coverage_heavy": 0.0,
            "mean_intensity": 0.0, "max_intensity": 0.0, "spatial_variance": 0.0,
            "temporal_variability": 0.0, "average_motion": 0.0, "rainy_frames_count": 0
        }

    rain_mask_light = (data > rain_thresh) & masks
    rain_pixel_count = torch.sum(rain_mask_light).item()
    
    rain_coverage_light = float(rain_pixel_count / total_pixels)
    
    if rain_pixel_count == 0:
        return {
            "rain_coverage_light": 0.0, "rain_coverage_moderate": 0.0, "rain_coverage_heavy": 0.0,
            "mean_intensity": 0.0, "max_intensity": 0.0, "spatial_variance": 0.0,
            "temporal_variability": 0.0, "average_motion": 0.0, "rainy_frames_count": 0
        }

    rain_coverage_moderate = float(torch.sum((data >= moderate_thresh) & masks).item() / total_pixels)
    rain_coverage_heavy = float(torch.sum((data >= heavy_thresh) & masks).item() / total_pixels)
    
    mean_intensity = float(torch.mean(data[rain_mask_light]).item())
    max_intensity = float(torch.max(data).item())
    
    spatial_variance = float(torch.mean(torch.var(data, dim=(1, 2))).item())
    
    frame_diffs = torch.abs(data[1:] - data[:-1])
    temporal_variability = float(torch.mean(frame_diffs).item())
    
    # Active frame calculation
    valid_pixels_per_frame = torch.sum(masks, dim=(1, 2))
    rain_pixels_per_frame = torch.sum(rain_mask_light, dim=(1, 2))
    frame_rain_fractions = rain_pixels_per_frame.float() / (valid_pixels_per_frame.float() + 1e-6)
    
    rainy_frames_count = int(torch.sum(frame_rain_fractions > min_coverage).item()) * stride
    
    # Centroid Motion Vector Estimation
    displacements = []
    frames_indices = torch.nonzero(frame_rain_fractions > min_coverage).flatten()
    
    if frames_indices.numel() > 1:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device), 
            indexing='ij'
        )
        for idx in range(len(frames_indices) - 1):
            t1 = int(frames_indices[idx])
            t2 = int(frames_indices[idx+1])
            
            if (t2 - t1) == 1:
                m1, m2 = rain_mask_light[t1].float(), rain_mask_light[t2].float()
                sum1, sum2 = m1.sum(), m2.sum()
                if sum1 > 0 and sum2 > 0:
                    c1_y = (m1 * grid_y).sum() / sum1
                    c1_x = (m1 * grid_x).sum() / sum1
                    c2_y = (m2 * grid_y).sum() / sum2
                    c2_x = (m2 * grid_x).sum() / sum2
                    
                    dist = torch.sqrt((c1_y - c2_y)**2 + (c1_x - c2_x)**2).item()
                    if min_dist < dist < max_dist:
                        displacements.append(dist)
                        
    average_motion = float(np.mean(displacements)) if displacements else 0.0
    
    return {
        "rain_coverage_light": rain_coverage_light, "rain_coverage_moderate": rain_coverage_moderate, "rain_coverage_heavy": rain_coverage_heavy,
        "mean_intensity": mean_intensity, "max_intensity": max_intensity, "spatial_variance": spatial_variance,
        "temporal_variability": temporal_variability, "average_motion": average_motion, 
        "rainy_frames_count": rainy_frames_count
    }


def calculate_data_driven_splits(df, weights_dict, train_ratio, val_ratio, logger):
    """
    Clusters archives using standard scalar configurations and computes the composite difficulty 
    score mapped cleanly to your active YAML config rules with defensive NaN handling.
    """
    feature_cols = [
        "rain_coverage_light", "rain_coverage_moderate", "rain_coverage_heavy", 
        "mean_intensity", "max_intensity", "spatial_variance", 
        "temporal_variability", "average_motion", "rainy_frames_count"
    ]
    
    # Fill in any missing or NaN parameters before scaling to ensure numerical stability
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    # Fallback guard to clear any scaling NaNs induced by zero-variance divisions
    scaled_features = np.nan_to_num(scaled_features)
    
    logger.info("Running Unsupervised K-Means Pattern Discovery across Descriptor Space...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["weather_regime_cluster"] = kmeans.fit_predict(scaled_features)
    
    # Computes composite difficulty score using weights from config
    df["computed_difficulty_score"] = (
        (df["temporal_variability"] / (df["temporal_variability"].max() + 1e-6)) * weights_dict.get("temporal_variability", 0.25) +
        (df["spatial_variance"] / (df["spatial_variance"].max() + 1e-6)) * weights_dict.get("spatial_variance", 0.20) +
        (df["rain_coverage_heavy"] / (df["rain_coverage_heavy"].max() + 1e-6)) * weights_dict.get("heavy_tail", 0.20) +
        (df["rain_coverage_light"] / (df["rain_coverage_light"].max() + 1e-6)) * weights_dict.get("frequency", 0.15) +
        (df["max_intensity"] / (df["max_intensity"].max() + 1e-6)) * weights_dict.get("max_intensity", 0.10) +
        (df["average_motion"] / (df["average_motion"].max() + 1e-6)) * weights_dict.get("dynamics", 0.10)
    )

    cluster_order = df.groupby("weather_regime_cluster")["computed_difficulty_score"].mean().sort_values().index
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_order)}
    df["weather_regime_cluster"] = df["weather_regime_cluster"].map(cluster_mapping)
    
    df["split"] = "unassigned"
    for cluster_id in range(3):
        cluster_df = df[df["weather_regime_cluster"] == cluster_id]
        shuffled_indices = cluster_df.sample(frac=1.0, random_state=42).index
        
        n_total = len(shuffled_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        df.loc[shuffled_indices[:n_train], "split"] = "train"
        df.loc[shuffled_indices[n_train:n_train + n_val], "split"] = "val"
        df.loc[shuffled_indices[n_train + n_val:], "split"] = "test"
        
    return df


def main():
    run_start_time = time.time()

    params = load_stratification_environment()
    config = params["config"]
    strat_name = params["strat_name"]
    zone = config["paths"]["zone"]
    weights = config["weights"]
    stride = config["dataset"].get("stride", 3)
    preprocessing_metadata = params["preprocessing_metadata"]
    clip_val = preprocessing_metadata["clip_value"]

    # Logger Setup
    logger = get_logger(name="characterize", log_dir="stratify", log_file=f"{strat_name}.log")
    logger.info("=" * 60)
    logger.info("Archive Characterization started via Infrastructure Config Workflow")
    logger.info("=" * 60)
    
    print_configuration(config=config, logger=logger)

    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = Path(config["paths"]["output_dir"]) / zone / strat_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    logger.info(f"Initializing Dataset using uncompressed data arrays inside: {processed_dir}")
    dataset = StratificationRadarDataset(processed_dir=processed_dir, stride=stride, clip_value=clip_val)
    
    if len(dataset) == 0:
        logger.error(f"No valid preprocessed directories discovered matching pattern inside {processed_dir}")
        sys.exit(1)
        
    logger.info(f"Discovered {len(dataset)} processed archive directories.")

    # Dataloader
    num_workers = config["dataset"].get("num_workers", 2)
    use_cuda = torch.cuda.is_available()
    pin_memory = config["dataset"].get("pin_memory", True) if use_cuda else False
    prefetch_factor = config["dataset"].get("prefetch_factor", 1) if num_workers > 0 else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, # Whole multi-day tracks parsed sequentially
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Metrics tracking offloaded to execution environment: {device}")
    
    rain_thresh = config["meteorology"]["rain_threshold"]
    moderate_thresh = config["meteorology"]["moderate_rain_threshold"]
    heavy_thresh = config["meteorology"]["heavy_rain_threshold"]
    min_rain_coverage_fraction = config["meteorology"]["min_rain_coverage_fraction"]
    min_displacement_pixels = config["motion_tracking"]["min_displacement_pixels"]
    max_displacement_pixels = config["motion_tracking"]["max_displacement_pixels"]
    
    records = []
    extraction_start = time.time()

    for idx, batch in enumerate(dataloader):
        if not batch["valid"][0]:
            continue
            
        descriptors = calculate_archive_descriptors(
            batch=batch,
            rain_thresh=rain_thresh,
            moderate_thresh=moderate_thresh,
            heavy_thresh=heavy_thresh,
            min_coverage=min_rain_coverage_fraction,
            min_dist=min_displacement_pixels,
            max_dist=max_displacement_pixels,
            device=device,
            stride=stride
        )
        
        # Append identifying metadata to the feature record dictionary
        descriptors["archive_name"] = batch["archive_name"][0]
        descriptors["relative_path"] = str(processed_dir / batch["dir_name"][0])
        descriptors["total_frames"] = int(batch["total_frames"][0].item())
        descriptors["missing_data_ratio"] = float(batch["missing_data_ratio"][0].item())
        
        records.append(descriptors)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataloader):
            logger.info(f"Progress Metric: [{idx + 1}/{len(dataloader)}] preprocessed archives modeled.")
            
    if not records:
        logger.error("Zero descriptive records extracted. Program terminating.")
        sys.exit(1)
        
    logger.info(f"Feature computation complete in {(time.time() - extraction_start)/60:.2f} minutes.")
    
    df = pd.DataFrame(records)
    
    train_ratio = config["split_ratios"]["train"]
    val_ratio = config["split_ratios"]["val"]
    df = calculate_data_driven_splits(df, weights, train_ratio, val_ratio, logger)
    
    output_csv_path = output_dir / "dataset_difficulty_metadata.csv"
    df.to_csv(output_csv_path, index=False)
    
    logger.info(f"Master archive descriptors successfully written to: {output_csv_path}")

    # Save stratification metadata
    metadata = {
        "zone": zone,
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "rain_threshold": rain_thresh,
        "moderate_rain_threshold": moderate_thresh,
        "heavy_rain_threshold": heavy_thresh,
        "min_rain_coverage_fraction": min_rain_coverage_fraction,
        "min_displacement_pixels": min_displacement_pixels,
        "max_displacement_pixels": max_displacement_pixels,
        "dataset_stride": stride,
        "weights": {
            "temporal_variability": weights["temporal_variability"],
            "spatial_variance": weights["spatial_variance"],
            "heavy_tail": weights["heavy_tail"],
            "frequency": weights["frequency"],
            "max_intensity": weights["max_intensity"],
            "dynamics": weights["dynamics"]
        },
        "split_ratios": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio
        },
        "preprocessing": preprocessing_metadata
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Format and redirect the final scientific summary table directly into logs
    logger.info("\n" + "="*85 + "\nSCIENTIFIC METEOROLOGICAL PARITY REPORT (STRATIFIED DATA SPLITS)\n" + "="*85)
    summary_report = df.groupby("split").agg(
        block_count=("archive_name", "count"),
        avg_diff_score=("computed_difficulty_score", "mean"),
        light_rain_cov=("rain_coverage_light", "mean"),
        heavy_convect_cov=("rain_coverage_heavy", "mean"),
        mean_intensity_val=("mean_intensity", "mean"),
        max_intensity_val=("max_intensity", "mean"),
        temporal_variability_mad=("temporal_variability", "mean"),
        system_motion_velocity=("average_motion", "mean")
    ).reindex(["train", "val", "test"])
    
    for line in summary_report.to_string().split("\n"):
        logger.info(line)
    logger.info("="*85)
    
    logger.info(f"Process complete. Total pipeline runtime: {(time.time() - run_start_time)/60:.2f} minutes.")


if __name__ == "__main__":
    main()
