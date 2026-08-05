from datetime import datetime
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader

from src.data.dataset import RadarDataset
from src.evaluation.losses import MaskedMSELoss
import src.evaluation.metrics as eval_metrics
from src.models.factory import get_model
from src.utils.config import load_evaluation_environment
from src.utils.logger import get_logger, get_log_prefix, print_configuration
from src.utils.predictions_io import flush_chunk_to_disk


def create_test_dir(test_name):
    test_dir = Path("output/evaluation") / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def create_test_dataset(processed_dir, data_split, sequence_length, predict_steps):
    dataset = RadarDataset(
        processed_dir=processed_dir,
        start_datetime=data_split["test_start"],
        end_datetime=data_split["test_end"],
        sequence_length=sequence_length,
        predict_steps=predict_steps
    )
    if len(dataset) == 0:
        raise ValueError(f"Test dataset is empty for bounds specified inside {processed_dir}.")
    return dataset


def create_model(model_architecture, sequence_length, predict_steps, base_channels, input_channels, num_groups):
    model = get_model(
        model_name=model_architecture,
        sequence_length=sequence_length,
        predict_steps=predict_steps,
        base_channels=base_channels,
        input_channels=input_channels,
        num_groups=num_groups
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    return model, device, num_parameters


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def run_test(model, test_loader, device, criterion, logger, test_dir, batch_size, config, flush_sequence_capacity=240):
    """
    Evaluates the model on the test split using mixed precision.
    Streams output predictions directly to disk in fixed batch chunks to maintain
    a flat memory profile, and generates an optimized O(1) random-access manifest.json.
    """
    model.eval()

    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    running_valid_pixels = 0
    num_steps = test_loader.dataset.predict_steps

    _, height, width = test_loader.dataset[0][2].shape

    running_loss_per_step = None
    running_mae_per_step = None
    running_rmse_per_step = None
    running_valid_pixels_per_step = None

    # Dynamic config resolution
    met_cfg = config.get("meteorology", {})
    raw_yaml_thresholds = met_cfg.get("thresholds", {"Light Rain": 0.12, "Moderate Rain": 5.0, "Heavy Rain": 10.0})
    scales = met_cfg.get("fss_scales", [5, 15, 31])
    
    # Transform mm/h thresholds to raw array units (1/100 mm per 5-min steps)
    thresholds = {
        t_name: float(t_val * (100.0 / 12.0)) 
        for t_name, t_val in raw_yaml_thresholds.items()
    }
    
    logger.info("Converted operational intensity limits for array comparisons:")
    for k, v in thresholds.items():
        logger.info(f"  {k}: {raw_yaml_thresholds[k]} mm/h -> {v:.2f} (1/100 mm / 5 min)")
    
    # Track contingency cell statistics [hits, false_alarms, misses, correct_negatives] per step
    gpu_contingency_sums = {
        step: {t_name: torch.zeros(4, device=device, dtype=torch.float64) for t_name in thresholds}
        for step in range(num_steps)
    }
    
    # Registries to aggregate Fractions Skill Scores (FSS) and Structure-Amplitude-Location (SAL)
    fss_registry = {step: {t_name: {scale: [] for scale in scales} for t_name in thresholds} for step in range(num_steps)}
    sal_registry = {step: {"S": [], "A": [], "L": []} for step in range(num_steps)}
    
    # Hybrid Correlation Registries: VRAM sums (Pearson) + Stratified List Buckets (Spearman)
    gpu_corr_accumulators = torch.zeros(6, device=device, dtype=torch.float64)
    bucket_light = []
    bucket_mod = []
    bucket_heavy = []
    
    # 2D grid accumulators tracking geographic MAE footprint density
    spatial_mae_accumulator = torch.zeros((height, width), device=device, dtype=torch.float64)
    spatial_counter = torch.zeros((height, width), device=device, dtype=torch.float64)

    # Minimal RAM Isolation Buffers & Fast O(1) Index Trackers
    chunk_inputs, chunk_preds, chunk_targets, chunk_masks, chunk_dates = [], [], [], [], []
    manifest = {"global_to_local": {}, "date_to_global": {}, "total_samples": 0}
    global_counter = 0
    chunk_idx = 0

    num_batches = len(test_loader)
    use_amp = (device.type == 'cuda')

    with torch.no_grad():
        for batch_idx, (x, x_mask, y, y_mask, timestamps) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            x_mask = x_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)
            x = torch.cat([x, x_mask], dim=2)
            
            with torch.autocast(device_type='cuda', enabled=use_amp):
                predictions = model(x)
                loss = criterion(predictions, y, y_mask)

            # Ensure high-precision Float32 for metrics calculation
            predictions_f32 = predictions.float()

            # Dynamic valid pixel weights extracted per batch
            valid_pixels = y_mask.sum().item()
            running_valid_pixels += valid_pixels

            # Calculate overall weighted metric metrics accumulated exactly like train loop
            running_loss += loss.item() * valid_pixels
            running_mae += eval_metrics.masked_mae(predictions_f32, y, y_mask).item() * valid_pixels
            running_rmse += eval_metrics.masked_rmse(predictions_f32, y, y_mask).item() * valid_pixels

            # Forecast horizon step extraction checks
            loss_per_step = eval_metrics.masked_rmse_per_step(predictions_f32, y, y_mask).detach() ** 2
            mae_per_step = eval_metrics.masked_mae_per_step(predictions_f32, y, y_mask).detach()
            rmse_per_step = eval_metrics.masked_rmse_per_step(predictions_f32, y, y_mask).detach()
            valid_pixels_per_step = y_mask.sum(dim=(0, 2, 3))

            if running_loss_per_step is None:
                running_loss_per_step = torch.zeros_like(loss_per_step)
                running_mae_per_step = torch.zeros_like(mae_per_step)
                running_rmse_per_step = torch.zeros_like(rmse_per_step)
                running_valid_pixels_per_step = torch.zeros_like(valid_pixels_per_step)

            running_loss_per_step += loss_per_step * valid_pixels_per_step
            running_mae_per_step += mae_per_step * valid_pixels_per_step
            running_rmse_per_step += rmse_per_step * valid_pixels_per_step
            running_valid_pixels_per_step += valid_pixels_per_step
            
            # Inline GPU Advanced Metrics Accumulation
            _accumulate_advanced_meteorological_metrics(
                predictions_f32=predictions_f32, y=y, y_mask=y_mask, batch_idx=batch_idx,
                thresholds=thresholds, scales=scales, gpu_contingency_sums=gpu_contingency_sums,
                fss_registry=fss_registry, sal_registry=sal_registry,
                spatial_mae_accumulator=spatial_mae_accumulator, spatial_counter=spatial_counter,
                bucket_light=bucket_light, bucket_mod=bucket_mod, bucket_heavy=bucket_heavy
            )
            
            # Pearson: Accumulate running algebraic products natively on the GPU VRAM
            batch_corr_components = eval_metrics.compute_batch_correlation_sums(
                predictions_f32, y, y_mask, thresholds["Light Rain"]
            )
            gpu_corr_accumulators += batch_corr_components

            # Append current batch arrays cleanly to RAM chunk lists
            chunk_inputs.append(x.cpu().numpy())
            chunk_preds.append(predictions_f32.cpu().numpy())
            chunk_targets.append(y.cpu().numpy())
            chunk_masks.append(y_mask.cpu().numpy())
            chunk_dates.extend(timestamps)

            # Volume-based memory flush trigger
            current_buffered_count = len(chunk_dates)
            is_last_batch = (batch_idx + 1 == num_batches)
            
            if current_buffered_count >= flush_sequence_capacity or is_last_batch:
                if len(chunk_inputs) > 0:
                    file_name, num_saved, flushed_dates = flush_chunk_to_disk(
                        chunk_inputs, chunk_preds, chunk_targets, chunk_masks, chunk_dates, test_dir, chunk_idx
                    )
                    
                    for local_idx in range(num_saved):
                        global_str = str(global_counter)
                        manifest["global_to_local"][global_str] = [file_name, local_idx]
                        
                        raw_timestamp = flushed_dates[local_idx]
                        naive_dt = np.datetime64(int(raw_timestamp), 's')
                        iso_date_str = str(naive_dt).replace(' ', 'T')
                        
                        manifest["date_to_global"][iso_date_str] = global_counter
                        global_counter += 1

                    chunk_idx += 1
                
                # Reset buffers and clear hardware pointers completely
                chunk_inputs, chunk_preds, chunk_targets, chunk_masks, chunk_dates = [], [], [], [], []
                gc.collect()

            print(f"\r\033[K{get_log_prefix(logger.name)}Test: Batch {batch_idx + 1}/{num_batches}", end="", flush=True)

    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    manifest["total_samples"] = global_counter
    
    with open(Path(test_dir) / "predictions" / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    denom = max(running_valid_pixels, 1)
    denom_per_step = torch.clamp(running_valid_pixels_per_step, min=1.0)

    # Finalize and save computed metrics to json
    _compile_and_save_meteorological_artifacts(
        test_dir=test_dir, thresholds=thresholds, scales=scales, gpu_contingency_sums=gpu_contingency_sums,
        fss_registry=fss_registry, sal_registry=sal_registry, gpu_corr_accumulators=gpu_corr_accumulators,
        bucket_light=bucket_light, bucket_mod=bucket_mod, bucket_heavy=bucket_heavy,
        spatial_mae_accumulator=spatial_mae_accumulator, spatial_counter=spatial_counter, 
        running_loss_per_step=running_loss_per_step, running_rmse_per_step=running_rmse_per_step, 
        denom_per_step=denom_per_step, logger=logger
    )

    metrics = {
        "overall": {
            "loss": running_loss / denom,
            "mae": running_mae / denom,
            "rmse": running_rmse / denom
        },
        "forecast_horizon": {
            "minutes": [5 * (i + 1) for i in range(num_steps)],
            "loss": (running_loss_per_step / denom_per_step).cpu().tolist(),
            "mae": (running_mae_per_step / denom_per_step).cpu().tolist(),
            "rmse": (running_rmse_per_step / denom_per_step).cpu().tolist()
        }
    }
    return {"metrics": metrics}


def save_test_results(
    test_path, test_name, model_architecture, model_path, processed_dir, data_split,
    batch_size, sequence_length, predict_steps, preprocessing_metadata,
    discarded_test_sequences, num_test_samples, num_test_batches, num_parameters,
    test_metrics, device, dataset_time_seconds, test_time_seconds, total_test_time_seconds
):
    test_info = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        "model": {
            "architecture": model_architecture,
            "num_parameters": num_parameters,
            "checkpoint": str(model_path)
        },
        "dataset": {
            "processed_dir": processed_dir,
            "test_start": data_split["test_start"],
            "test_end": data_split["test_end"],
            "discarded_test_sequences": discarded_test_sequences,
            "num_test_samples": num_test_samples,
            "num_test_batches": num_test_batches,
        },
        "test": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "predict_steps": predict_steps,
            "loss_function": "MaskedMSELoss"
        },
        "preprocessing": preprocessing_metadata,
        "runtime": {
            "device": str(device),
            "dataset_creation_time_seconds": dataset_time_seconds,
            "test_time_seconds": test_time_seconds,
            "total_time_seconds": total_test_time_seconds
        },
        "results": test_metrics,
    }

    with open(test_path, "w") as f:
        json.dump(test_info, f, indent=4)


def _accumulate_advanced_meteorological_metrics(
    predictions_f32, y, y_mask, batch_idx, thresholds, scales,
    gpu_contingency_sums, fss_registry, sal_registry, 
    spatial_mae_accumulator, spatial_counter,
    bucket_light, bucket_mod, bucket_heavy
):
    """
    Asynchronously updates verification metrics on the GPU and collects
    stratified intensity samples into memory-capped buckets to protect rare convective events.
    """
    num_steps = predictions_f32.shape[1]

    # 1. Update Core Categorical Contingency Arrays across Intensity Thresholds
    for t_name, t_val in thresholds.items():
        # Returns a (4, predict_steps) tensor cell block representing [Hits, FA, Miss, CN] over active steps
        batch_cells = eval_metrics.compute_batch_contingency(predictions_f32, y, y_mask, t_val)
        for step in range(num_steps):
            gpu_contingency_sums[step][t_name] += batch_cells[:, step]

        # 2. Run Neighborhood Scale Parsing Periodically to Maintain Optimal Loop Velocity
        if batch_idx % 4 == 0:
            for scale in scales:
                fss_horizon = eval_metrics.compute_batch_fss(predictions_f32, y, y_mask, scale, t_val)
                for step in range(num_steps):
                    fss_registry[step][t_name][scale].append(fss_horizon[step].item())

    # 3. Physical Structural Attribute Analysis (SAL Moment Tensors)
    if batch_idx % 4 == 0:
        sal_horizon = eval_metrics.compute_batch_sal(predictions_f32, y, y_mask)
        for step in range(num_steps):
            sal_registry[step]["S"].append(sal_horizon[0, step].item())
            sal_registry[step]["A"].append(sal_horizon[1, step].item())
            sal_registry[step]["L"].append(sal_horizon[2, step].item())

    # 4. Invert Log Scale only once for the Final Lead Horizon (Maximum Step / Final Forecast Horizon) Heatmap
    preds_mm_final = eval_metrics._inverse_log_transform(predictions_f32[:, -1])
    targets_mm_final = eval_metrics._inverse_log_transform(y[:, -1])
    diff_final = torch.abs(preds_mm_final - targets_mm_final)
    
    spatial_mae_accumulator += (diff_final * y_mask[:, -1]).sum(dim=0)
    spatial_counter += y_mask[:, -1].sum(dim=0)

    # 5. Meteorological stratified sampling (ensures extreme events are represented)
    preds_mm_full = eval_metrics._inverse_log_transform(predictions_f32)
    targets_mm_full = eval_metrics._inverse_log_transform(y)
    rain_mask_full = (targets_mm_full > thresholds["Light Rain"]) & (y_mask > 0.5)

    if torch.any(rain_mask_full):
        gt_pts = targets_mm_full[rain_mask_full].cpu().numpy().flatten()
        pd_pts = preds_mm_full[rain_mask_full].cpu().numpy().flatten()

        for gt_v, pd_v in zip(gt_pts, pd_pts):
            # Category C: Severe Heavy Convective Storm Cores (e.g., >= 10.0 mm/h)
            if gt_v >= thresholds["Heavy Rain"]:
                if len(bucket_heavy) < 20000:
                    bucket_heavy.append((float(gt_v), float(pd_v)))
            # Category B: Moderate Stratiform Rainbands
            elif gt_v >= thresholds["Moderate Rain"]:
                if len(bucket_mod) < 20000:
                    bucket_mod.append((float(gt_v), float(pd_v)))
            # Category A: Light Trace Rain / Drizzle Noise floor
            else:
                if len(bucket_light) < 20000:
                    bucket_light.append((float(gt_v), float(pd_v)))


def _compile_and_save_meteorological_artifacts(
    test_dir, thresholds, scales, gpu_contingency_sums, fss_registry, sal_registry,
    gpu_corr_accumulators, bucket_light, bucket_mod, bucket_heavy, 
    spatial_mae_accumulator, spatial_counter, running_loss_per_step, running_rmse_per_step, 
    denom_per_step, logger
):
    """
    Synchronizes GPU accumulators, computes specialized operational verification metrics,
    and runs an instantaneous, memory-safe stratified Spearman rank correlation.
    """
    logger.info("Compiling final advanced meteorological verification database...")
    final_metrics = {"horizon": {}, "categorical": {}, "fss": {}, "sal": {}}
    
    c_loss = running_loss_per_step.cpu().numpy()
    c_mse = running_rmse_per_step.cpu().numpy()
    c_denom_step = denom_per_step.cpu().numpy()
    
    num_steps = len(c_denom_step)

    for step in range(num_steps):
        mins = int((step + 1) * 5)
        
        # 1. Finalize continuous field degradation vectors
        final_metrics["horizon"][mins] = {
            "MAE": float(c_loss[step] / c_denom_step[step]),
            "RMSE": float(np.sqrt(c_mse[step] / c_denom_step[step]))
        }
        
        # 2. Finalize threshold-based contingency indicators
        final_metrics["categorical"][mins] = {}
        for t_name in thresholds:
            h, fa, m, cn = gpu_contingency_sums[step][t_name].cpu().numpy()
            tot = h + fa + m + cn
            
            pod = h / (h + m) if (h + m) > 0 else 0.0
            far = fa / (h + fa) if (h + fa) > 0 else 0.0
            csi = h / (h + fa + m) if (h + fa + m) > 0 else 0.0
            
            expected_hits = ((h + m) * (h + fa)) / tot if tot > 0 else 0.0
            expected_cn = ((cn + m) * (cn + fa)) / tot if tot > 0 else 0.0
            hss = ((h + cn) - (expected_hits + expected_cn)) / (tot - (expected_hits + expected_cn) + 1e-7) if tot > 0 else 0.0
            
            base_rate = (h + m) / tot if tot > 0 else 0.0
            seds = 0.0 if (base_rate == 0 or base_rate == 1 or csi == 0) else (np.log((h + fa) / tot) + np.log(base_rate)) / np.log(h / tot)
            
            final_metrics["categorical"][mins][t_name] = {
                "POD": float(pod), "FAR": float(far), "CSI": float(csi), "HSS": float(hss), "SEDS": float(seds)
            }
            
        # 3. Finalize neighborhood Fractions Skill Scores
        final_metrics["fss"][mins] = {}
        for t_name in thresholds:
            final_metrics["fss"][mins][t_name] = {
                str(scale): float(np.mean(fss_registry[step][t_name][scale])) if fss_registry[step][t_name][scale] else 0.0
                for scale in scales
            }
            
        # 4. Finalize structural decomposition segments
        final_metrics["sal"][mins] = {
            "S": float(np.mean(sal_registry[step]["S"])) if sal_registry[step]["S"] else 0.0,
            "A": float(np.mean(sal_registry[step]["A"])) if sal_registry[step]["A"] else 0.0,
            "L": float(np.mean(sal_registry[step]["L"])) if sal_registry[step]["L"] else 0.0
        }

    # 5. Compute global, pixel-perfect Pearson correlation across ALL points via streamed VRAM sums
    sum_x, sum_y, sum_x2, sum_y2, sum_xy, n_total_pixels = gpu_corr_accumulators.cpu().numpy()

    if n_total_pixels > 2:
        numerator = (n_total_pixels * sum_xy) - (sum_x * sum_y)
        denominator = np.sqrt(((n_total_pixels * sum_x2) - (sum_x ** 2)) * ((n_total_pixels * sum_y2) - (sum_y ** 2)))
        pearson_r = float(numerator / denominator) if denominator > 0 else 0.0
    else:
        pearson_r = 0.0

    # 6. Compute Spearman Rank using the balanced, intensity-stratified memory buckets
    master_stratified_pairs = bucket_light + bucket_mod + bucket_heavy
    if master_stratified_pairs and len(master_stratified_pairs) > 2:
        pairs_arr = np.array(master_stratified_pairs)
        gt_s = pairs_arr[:, 0]
        pd_s = pairs_arr[:, 1]
        spearman_rho, _ = spearmanr(gt_s, pd_s)
        spearman_rho = float(spearman_rho)
    else:
        spearman_rho = 0.0

    final_metrics["correlations"] = {
        "pearson": pearson_r,
        "spearman": spearman_rho
    }

    # Save summary verification files directly to the test workspace run folder root
    with open(Path(test_dir) / "calculated_verification_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    spatial_mae = (spatial_mae_accumulator / torch.clamp(spatial_counter, min=1.0)).cpu().numpy()
    np.save(Path(test_dir) / "spatial_mae_footprint.npy", spatial_mae)


def main():
    run_start_time = time.time()

    params = load_evaluation_environment()
    config = params["config"]
    test_name = params["test_name"]
    experiment_dir = params["experiment_dir"]
    preprocessing_metadata = params["preprocessing_metadata"]
    training_metadata = params["training_metadata"]

    sequence_length = training_metadata["hyperparameters"]["sequence_length"]
    predict_steps = training_metadata["hyperparameters"]["predict_steps"]
    base_channels = training_metadata["hyperparameters"]["base_channels"]
    model_architecture = training_metadata["model"]["architecture"]
    input_channels = training_metadata["hyperparameters"].get("input_channels", 2)
    num_groups = training_metadata["hyperparameters"].get("num_groups", 4)

    # Logger Setup
    logger = get_logger(name="evaluate", log_dir="evaluate", log_file=f"{test_name}.log")
    logger.info("=" * 60)
    logger.info("Evaluation started via Infrastructure Config Workflow")
    logger.info("=" * 60)

    # Path Setup
    test_dir = create_test_dir(test_name)
    test_path = test_dir / "test.json"
    model_path = experiment_dir / "checkpoints" / "_best_epoch.pt"

    print_configuration(config=config, logger=logger)

    # Dataset Creation
    logger.info("Creating dataset...")
    dataset_start_time = time.time()
    
    test_dataset = create_test_dataset(
        processed_dir=config["paths"]["processed_dir"],
        data_split=config["data_split"],
        sequence_length=sequence_length,
        predict_steps=predict_steps
    )
    num_test_samples = len(test_dataset)

    logger.info(f"Discarded test sequences: {test_dataset.invalid_sequence_count}")
    logger.info(f"Test samples: {num_test_samples}")
    logger.info(f"Test range: {config['data_split']['test_start']} -> {config['data_split']['test_end']}")

    dataset_time_seconds = time.time() - dataset_start_time
    logger.info(f"Dataset creation time: {dataset_time_seconds / 60:.2f} minutes")

    # DataLoader Setup
    logger.info("Creating DataLoader...")
    num_workers = config["parameters"]["num_workers"]
    use_cuda = torch.cuda.is_available()
    pin_memory = config["parameters"]["pin_memory"] if use_cuda else False
    prefetch_factor= config["parameters"]["prefetch_factor"] if num_workers > 0 else None

    batch_size = config["parameters"]["batch_size"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )
    num_test_batches = len(test_loader)
    logger.info(f"Test batches: {num_test_batches}")

    # Model Reconstruction
    logger.info("Loading Checkpoint...")
    model, device, num_parameters = create_model(
        model_architecture=model_architecture,
        sequence_length=sequence_length,
        predict_steps=predict_steps,
        base_channels=base_channels,
        input_channels=input_channels,
        num_groups=num_groups
    )

    load_checkpoint(model=model, checkpoint_path=model_path, device=device)
    logger.info(f"Using device: {device}")
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Parameters: {num_parameters:,}")
    logger.info(f"Loaded checkpoint: {model_path}")

    criterion = MaskedMSELoss()
    logger.info(f"Loss function: {criterion}")

    # Run Evaluation Loop
    logger.info("Evaluating model and streaming predictions to disk...")
    test_start_time = time.time()

    test_result = run_test(
        model=model,
        test_loader=test_loader,
        device=device,
        criterion=criterion,
        logger=logger,
        test_dir=test_dir,
        batch_size=batch_size,
        config=config,
        flush_sequence_capacity=240
    )

    test_metrics = test_result["metrics"]
    overall = test_metrics["overall"]
    logger.info(f"Test split evaluation complete! Metrics: {overall}")
    logger.info("Per-step metrics:")

    for minute, loss, mae, rmse in zip(
        test_metrics["forecast_horizon"]["minutes"],
        test_metrics["forecast_horizon"]["loss"],
        test_metrics["forecast_horizon"]["mae"],
        test_metrics["forecast_horizon"]["rmse"]
    ):
        logger.info(f"+{minute:2d} min | Loss={loss:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f}")

    test_time_seconds = time.time() - test_start_time
    logger.info(f"Evaluation time: {test_time_seconds / 60:.2f} minutes")

    total_test_time_seconds = time.time() - run_start_time

    save_test_results(
        test_path=test_path,
        test_name=test_name,
        model_architecture=model_architecture,
        model_path=model_path,
        processed_dir=config["paths"]["processed_dir"],
        data_split=config["data_split"],
        batch_size=config["parameters"]["batch_size"],
        sequence_length=sequence_length,
        predict_steps=predict_steps,
        preprocessing_metadata=preprocessing_metadata,
        discarded_test_sequences=test_dataset.invalid_sequence_count,
        num_test_samples=num_test_samples,
        num_test_batches=num_test_batches,
        num_parameters=num_parameters,
        test_metrics=test_metrics,
        device=device,
        dataset_time_seconds=dataset_time_seconds,
        test_time_seconds=test_time_seconds,
        total_test_time_seconds=total_test_time_seconds
    )
    logger.info(f"Saved evaluation tracking json array directory to: {test_dir}")

    # Post-evaluation visualization
    logger.info("=" * 60)
    logger.info("Evaluation complete. Invoking visualization engine scripts...")
    logger.info("=" * 60)

    # Forecast Horizon Degradation Curves
    logger.info(f"Plotting forecast horizons...")
    horizon_cmd = [
        sys.executable, "-m", "src.visualization.plot_forecast_horizon",
        "--test-dir", str(test_dir)
    ]
    try:
        subprocess.run(horizon_cmd, check=True)
        logger.info(f"Forecast horizon plots generated inside: {test_dir}")
    except Exception as e:
        logger.error(f"Post-testing horizon plotting failed to render: {e}")

    # Gather default plotting configurations
    idx_target = config["parameters"].get("prediction_plot_index", 250)
    plots_subdir = test_dir / "plots"
    plots_subdir.mkdir(parents=True, exist_ok=True)
    
    # Define centralized plotting config yml route
    plotting_config_path = "configs/plotting.yml"

    # Spatial Sample Prediction Matrix Map Grid
    logger.info(f"Plotting prediction matrix for designated sample index: {idx_target}...")
    predict_cmd = [
        sys.executable, "-m", "src.visualization.plot_prediction",
        "--config", plotting_config_path,
        "--opts", f"paths.test_dir={test_dir}", f"query.target={idx_target}"
    ]
    try:
        subprocess.run(predict_cmd, check=True)
        logger.info(f"Spatial prediction matrix maps rendered successfully into: {plots_subdir}")
    except Exception as e:
        logger.error(f"Post-testing spatial prediction plotting failed to render: {e}")

    # Convective Nowcasting GIF Loop
    logger.info(f"Creating input to prediction GIF for designated sample index: {idx_target}...")
    gif_cmd = [
        sys.executable, "-m", "src.visualization.create_gif",
        "--config", plotting_config_path,
        "--opts", f"paths.test_dir={test_dir}", f"query.target={idx_target}", "query.fps=2"
    ]
    try:
        subprocess.run(gif_cmd, check=True)
        logger.info(f"Convective nowcasting loop animation compiled inside: {test_dir / 'animations'}")
    except Exception as e:
        logger.error(f"Post-testing GIF animation compilation failed to render: {e}")

    logger.info("=" * 60)
    logger.info("Testing completed successfully")
    logger.info("=" * 60)

    total_time_seconds = time.time() - run_start_time
    logger.info(f"Total execution time: {total_time_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    main()
