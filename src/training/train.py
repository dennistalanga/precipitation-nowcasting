import contextlib
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import mlflow
import mlflow.data
import mlflow.pytorch

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.data.dataset import RadarDataset
from src.training.validate import evaluate_train, evaluate_val
from src.evaluation.losses import MaskedMSELoss
from src.evaluation.metrics import masked_mae, masked_rmse
from src.models.factory import get_model
from src.utils.config import load_training_environment
from src.utils.logger import get_logger, get_log_prefix, print_configuration



def create_experiment_dir(model_output_dir, train_name):
    experiment_dir = Path(model_output_dir) / train_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def create_train_datasets(processed_dir, data_split, dataset_cfg):
    train_dataset = RadarDataset(
        processed_dir=processed_dir,
        start_datetime=data_split["train_start"],
        end_datetime=data_split["train_end"],
        sequence_length=dataset_cfg["sequence_length"],
        predict_steps=dataset_cfg["predict_steps"]
    )

    val_dataset = RadarDataset(
        processed_dir=processed_dir,
        start_datetime=data_split["val_start"],
        end_datetime=data_split["val_end"],
        sequence_length=dataset_cfg["sequence_length"],
        predict_steps=dataset_cfg["predict_steps"]
    )

    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty for parameters inside {processed_dir}.")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty for parameters inside {processed_dir}.")

    return train_dataset, val_dataset


def create_model(
    model_architecture, sequence_length, predict_steps,
    input_channels, base_channels, num_groups
):
    model = get_model(
        model_name=model_architecture,
        sequence_length=sequence_length,
        predict_steps=predict_steps,
        input_channels=input_channels,
        base_channels=base_channels,
        num_groups=num_groups        
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_parameters = sum(p.numel() for p in model.parameters())

    return model, device, num_parameters


def save_experiment_info(
    experiment_path, train_name, config, experiment_dir, preprocessing_metadata,
    discarded_train_sequences, discarded_val_sequences, num_train_samples, num_val_samples,
    num_train_batches, num_val_batches, num_parameters, best_epoch, best_metrics,
    device, dataset_time_seconds, train_time_seconds, total_time_seconds, history,
    optimizer, scheduler, peak_vram_gb, run_id, experiment_id
):  
    pytorch_version = torch.__version__
    hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    experiment_info = {
        "train_name": train_name,
        "timestamp": datetime.now().isoformat(),
        "mlflow_metadata": {
            "run_id": run_id,
            "experiment_id": experiment_id
        },
        "env_framework": f"PyTorch {pytorch_version}",
        "model": {
            "architecture": config["model"]["architecture"],
            "num_parameters": num_parameters,
            "experiment_dir": str(experiment_dir),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "checkpoint_state": "_best_epoch_state.pt",
            "checkpoint_optimizer": "_best_epoch_optimizer.pt"
        },
        "dataset": {
            "processed_dir": config["paths"]["processed_dir"],
            "train_start": config["data_split"]["train_start"],
            "train_end": config["data_split"]["train_end"],
            "val_start": config["data_split"]["val_start"],
            "val_end": config["data_split"]["val_end"],
            "discarded_train_sequences": discarded_train_sequences,
            "discarded_val_sequences": discarded_val_sequences,
            "num_train_samples": num_train_samples,
            "num_val_samples": num_val_samples,
            "num_train_batches": num_train_batches,
            "num_val_batches": num_val_batches,
        },
        "hyperparameters": {
            "batch_size": config["optimization"]["batch_size"],
            "learning_rate": config["optimization"]["learning_rate"],
            "weight_decay": config["optimization"]["weight_decay"],
            "base_channels": config["model"]["base_channels"],
            "num_groups": config["model"]["num_groups"],
            "epochs": config["optimization"]["epochs"],
            "sequence_length": config["dataset"]["sequence_length"],
            "predict_steps": config["dataset"]["predict_steps"],
            "optimizer": optimizer.__class__.__name__,
            "loss_function": "MaskedMSELoss",
            "scheduler": {
                "type": scheduler.__class__.__name__,
                "mode": scheduler.mode,
                "factor": scheduler.factor,
                "patience": scheduler.patience,
                "min_lr": scheduler.min_lrs[0]
            },
        },
        "preprocessing": preprocessing_metadata,
        "runtime": {
            "device": str(hardware_name),
            "dataset_creation_time_seconds": dataset_time_seconds,
            "training_time_seconds": train_time_seconds,
            "total_time_seconds": total_time_seconds
        },
        "peak_vram_gb": peak_vram_gb,
        "final_metrics": {
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "train_mae": history["train_mae"][-1] if history["train_mae"] else None,
            "val_mae": history["val_mae"][-1] if history["val_mae"] else None,
            "train_rmse": history["train_rmse"][-1] if history["train_rmse"] else None,
            "val_rmse": history["val_rmse"][-1] if history["val_rmse"] else None
        }
    }

    with open(experiment_path, "w") as f:
        json.dump(experiment_info, f, indent=4)

def save_checkpoint(model, optimizer, scheduler, checkpoint_dir, epoch_num, train_metrics, val_metrics, model_config, logger):
    checkpoint_path = checkpoint_dir / f"_epoch-{epoch_num}.pt"
    torch.save({
        "epoch": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model_config,
        "metrics": {
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"]
        }
    }, checkpoint_path)
    logger.info(f"Saved checkpoint for epoch {epoch_num}")


def update_best_model(model, optimizer, scheduler, checkpoint_dir, epoch_num, train_metrics, val_metrics, model_config, best_val_loss, logger):
    if val_metrics["loss"] >= best_val_loss:
        return None

    checkpoint_path = checkpoint_dir / f"_best_epoch.pt"
    torch.save({
        "epoch": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model_config,
        "metrics": {
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"]
        }
    }, checkpoint_path)
    logger.info(f"New best model at epoch {epoch_num}")

    best_metrics = {
        "epoch": epoch_num,
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "train_mae": train_metrics["mae"],
        "val_mae": val_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "val_rmse": val_metrics["rmse"]
    }
    return {
        "best_val_loss": val_metrics["loss"],
        "best_epoch": epoch_num,
        "best_metrics": best_metrics
    }


def update_history(history, epoch_num, train_metrics, val_metrics, learning_rate, epoch_runtime_seconds):
    history["epoch"].append(epoch_num)
    history["train_loss"].append(train_metrics["loss"])
    history["val_loss"].append(val_metrics["loss"])
    history["train_mae"].append(train_metrics["mae"])
    history["val_mae"].append(val_metrics["mae"])
    history["train_rmse"].append(train_metrics["rmse"])
    history["val_rmse"].append(val_metrics["rmse"])
    history["learning_rate"].append(learning_rate)
    history["epoch_runtime_seconds"].append(epoch_runtime_seconds)
    history["gradient_norm"].append(train_metrics["gradient_norm"])


def save_history(history_path, history):
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)


def train_one_epoch(model, train_loader, optimizer, criterion, device, logger, epoch, scaler):
    model.train()

    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    running_valid_pixels = 0
    running_grad_norm = 0.0

    num_batches = len(train_loader)

    for batch_idx, (x, x_mask, y, y_mask, _) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        x_mask = x_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_mask = y_mask.to(device, non_blocking=True)
        x = torch.cat([x, x_mask], dim=2) # dim=2 accounts for the Batch dimension (Batch, Time, Channel, H, W)

        optimizer.zero_grad()

        # Enforce mixed precision forward pass
        with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
            predictions = model(x)
            loss = criterion(predictions, y, y_mask)

        if epoch == 0 and batch_idx == 0:
            # Clear line first in case a progress bar was present
            print("\r\033[K", end="", flush=True)
            logger.info(f"x shape: {x.shape}")
            logger.info(f"pred shape: {predictions.shape}")
            logger.info(f"y shape: {y.shape}")

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Only step the optimizer and log metrics if the gradient is finite
            if not torch.isnan(total_norm) and not torch.isinf(total_norm):
                scaler.step(optimizer)
                scaler.update()
                running_grad_norm += total_norm.item()
            else:
                # Scaler met an overflow; it will skip this batch, lower its scale, and try again
                scaler.update() 
                running_grad_norm += 0.0  # Log 0 to avoid breaking running metrics
        else:
            loss.backward()
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_grad_norm += total_norm.item()

        # pixel-level weighting
        valid_pixels = y_mask.sum().item()
        running_valid_pixels += valid_pixels

        running_loss += loss.item() * valid_pixels
        running_mae += masked_mae(predictions, y, y_mask).item() * valid_pixels
        running_rmse += masked_rmse(predictions, y, y_mask).item() * valid_pixels

        # Generate live logger-style prefix
        log_prefix = get_log_prefix(logger.name)
        
        # Print dynamic line: \r resets cursor, \033[K clears trailing text
        progress_str = f"\r\033[K{log_prefix}Epoch {epoch+1} Training: Batch {batch_idx + 1}/{num_batches}"
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    # Remove the line completely when the loop ends
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


    denom = max(running_valid_pixels, 1)
    metrics = evaluate_train(
        denom=denom,
        total_loss=running_loss,
        total_mae=running_mae,
        total_rmse=running_rmse,
        total_grad_norm=running_grad_norm / num_batches,
    )
    return metrics


def build_mlflow_dataset_profile(radar_dataset, context_name: str, config: dict):
    """
    Constructs a lightweight MLflow dataset tracking profile using 
    calculated metadata attributes from the offline preprocessed archives.
    """
    # 1. Extract structural array properties as a numeric matrix
    # This matches the shape/type validations required by the backend
    meta_features = np.array([
        float(radar_dataset.valid_sequence_count),
        float(radar_dataset.invalid_sequence_count),
        float(radar_dataset.sequence_length),
        float(radar_dataset.predict_steps),
        float(radar_dataset.total_candidate_sequences)
    ])
    
    # 2. Extract the string folder labels to pass to the registry name
    active_folders = [Path(p).name for p in radar_dataset.archive_paths]
    folders_joined = "-".join(active_folders)
    
    # 3. Build the profile without passing a string dictionary to the targets parameter
    # The targets parameter is set to an array containing the sequence count to prevent string crashes
    return mlflow.data.from_numpy(
        features=meta_features,
        source=str(config["paths"]["processed_dir"]),
        name=f"NW_{context_name.upper()}_files_{folders_joined}",
        targets=np.array([radar_dataset.valid_sequence_count])
    )



def main():
    run_start_time = time.time()
    
    params = load_training_environment()
    config = params["config"]
    train_name = params["train_name"]
    preprocessing_metadata = params["preprocessing_metadata"]

    # Logger Setup
    logger = get_logger(name="train", log_dir="train", log_file=f"{train_name}.log")
    # Strictly prevent MLflow's root logger from intercepting and duplicating console lines
    logger.propagate = False

    logger.info("=" * 60)
    logger.info("Training started via Infrastructure Config Workflow")
    logger.info("=" * 60)

    # Path Setup
    experiment_dir = create_experiment_dir(config["paths"]["model_output_dir"], train_name)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = experiment_dir / "history.json"
    experiment_path = experiment_dir / "experiment.json"

    hpo_parent_id = config.get("runtime", {}).get("hpo_parent_run_id", None)
    is_nested_trial = True if hpo_parent_id else False

    # Track completion status to handle conditional workspace deletion
    run_concluded_successfully = False

    try:
        # If this is a standalone baseline, we use standard fluent initialization
        if not is_nested_trial:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "precipitation-nowcasting"))
            
            # Start a normal root-level training tracking session
            mlflow_context = mlflow.start_run(run_name=train_name)
        else:
            # INDUSTRY STANDARD FOR SUBPROCESSES:
            # We explicitly open the parent run context by its exact UUID hash string
            # inside the child process first, then instantly declare a nested block
            mlflow_context = mlflow.start_run(run_id=hpo_parent_id)

        # Open the resolved context framework
        with mlflow_context:
            # If it's a nested trial, we enter the second layer to anchor it as a child node
            # If it's a standalone run, this secondary block is skipped cleanly
            child_context = mlflow.start_run(run_name=train_name, nested=True) if is_nested_trial else contextlib.nullcontext()
            
            with child_context as child_run:
                # Resolve the active tracking run ID automatically
                # If child_run is a nullcontext (standalone), fallback to the parent layer handle
                active_run_id = mlflow.active_run().info.run_id
                active_experiment_id = mlflow.active_run().info.experiment_id
                logger.info(f"Initialized MLflow context tracking node ID: {active_run_id} (Nested Trial: {is_nested_trial})")

                # Inject the active tracking ID back into runtime configurations
                config["runtime"] = {"mlflow_run_id": active_run_id}
                print_configuration(config=config, logger=logger)

                # Datasets
                logger.info("Creating datasets...")
                dataset_start_time = time.time()

                train_dataset, val_dataset = create_train_datasets(
                    processed_dir=config["paths"]["processed_dir"],
                    data_split=config["data_split"],
                    dataset_cfg=config["dataset"]
                )

                logger.info(f"Discarded train sequences: {train_dataset.invalid_sequence_count}")
                logger.info(f"Discarded validation sequences: {val_dataset.invalid_sequence_count}")
                logger.info(f"Train samples: {len(train_dataset)}")
                logger.info(f"Validation samples: {len(val_dataset)}")
                logger.info(f"Train range: {config['data_split']['train_start']} -> {config['data_split']['train_end']}")
                logger.info(f"Validation range: {config['data_split']['val_start']} -> {config['data_split']['val_end']}")

                dataset_time_seconds = time.time() - dataset_start_time
                logger.info(f"Total dataset creation time: {dataset_time_seconds / 60:.2f} minutes")

                # DataLoaders
                logger.info("Creating DataLoaders...")
                num_workers = config["dataset"]["num_workers"]
                use_cuda = torch.cuda.is_available()
                pin_memory = config["dataset"]["pin_memory"] if use_cuda else False
                prefetch_factor= config["dataset"]["prefetch_factor"] if num_workers > 0 else None

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config["optimization"]["batch_size"],
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                    prefetch_factor=prefetch_factor
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config["optimization"]["batch_size"],
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                    prefetch_factor=prefetch_factor
                )

                num_train_batches = len(train_loader)
                num_val_batches = len(val_loader)

                logger.info(f"Train batches: {num_train_batches}")
                logger.info(f"Validation batches: {num_val_batches}")

                # ─── LOG FLAT PARAMETERS & SOURCE ARRAYS LINEAGE ───
                mlflow.set_tags({
                    "mlflow.source.name": "src/training/train.py",
                    "mlflow.source.type": "LOCAL",
                    "pipeline_stage": "Hyperparameter_Optimization" if is_nested_trial else "Model_Training",
                    "model_status": "HPO_Trial" if is_nested_trial else "Candidate"
                })

                # ─── LOG REAL DATASET LINEAGE METADATA ───
                try:
                    # Profile the active training data configuration
                    train_profile = build_mlflow_dataset_profile(
                        radar_dataset=train_dataset, 
                        context_name="train", 
                        config=config
                    )
                    mlflow.log_input(train_profile, context="training")
                    
                    # Profile the active validation data configuration
                    val_profile = build_mlflow_dataset_profile(
                        radar_dataset=val_dataset, 
                        context_name="val", 
                        config=config
                    )
                    mlflow.log_input(val_profile, context="validation")
                    
                    logger.info("🎯 Successfully secured offline preprocessed dataset profiles in MLflow.")
                except Exception as data_err:
                    logger.warning(f"⚠️ Metadata dataset lineage logging dropped: {data_err}")

                # Model Setup
                pytorch_version = torch.__version__
                hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

                logger.info("Initializing model...")
                model, device, num_parameters = create_model(
                    model_architecture=config["model"]["architecture"],
                    sequence_length=config["dataset"]["sequence_length"],
                    predict_steps=config["dataset"]["predict_steps"],
                    input_channels=config["dataset"]["input_channels"],
                    base_channels=config["model"]["base_channels"],
                    num_groups=config["model"]["num_groups"]
                )
                mlflow.log_metric("num_parameters", num_parameters)

                logger.info(f"Using device: {hardware_name}")
                logger.info(f"Model architecture:\n{model}")
                logger.info(f"Parameters: {num_parameters:,}")

                # ─── PYTORCH HARDWARE SAFEGUARDS TIER ───
                # 1. Clear out any residual context or allocated memory leaks from prior failed runs
                torch.cuda.empty_cache()
                
                # 2. Deactivate the cuDNN benchmark tuner if your trials test variable channel/group topologies.
                # When enabled, cuDNN benches multiple backend algorithms per batch. If memory is tight,
                # benchmarking a heavy channel layer layout instantly triggers a status initialization crash.
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

                # Loss + Optimizer
                criterion = MaskedMSELoss()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["optimization"]["learning_rate"],
                    weight_decay=config["optimization"]["weight_decay"]
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
                )

                # Map complete operational parameters straight into searchable flat keys
                mlflow.log_params({
                    "env_hardware": hardware_name,
                    "env_framework": f"PyTorch {pytorch_version}",

                    "dataset_zone": preprocessing_metadata["zone"],
                    "dataset_resolution": f"{preprocessing_metadata['resize_height']}x{preprocessing_metadata['resize_width']}",
                    "dataset_transform": preprocessing_metadata["transform"],
                    "dataset_max_clip": preprocessing_metadata["clip_value"],
                    "dataset_sequence_length": config["dataset"]["sequence_length"],
                    "dataset_predict_steps": config["dataset"]["predict_steps"],
                    "dataset_train_window": f"{config['data_split']['train_start']}->{config['data_split']['train_end']}",
                    "dataset_val_window": f"{config['data_split']['val_start']}->{config['data_split']['val_end']}",

                    "model_architecture": config["model"]["architecture"],
                    "model_base_channels": config["model"]["base_channels"],
                    "model_num_groups": config["model"]["num_groups"],

                    "train_epochs": config["optimization"]["epochs"],
                    "train_optimizer": "AdamW",
                    "train_loss_function": "MaskedMSELoss",
                    "train_base_lr": config["optimization"]["learning_rate"],
                    "train_weight_decay": config["optimization"]["weight_decay"],
                    "train_batch_size": config["optimization"]["batch_size"],
                    "train_lr_scheduler": "ReduceLROnPlateau",
                })

                logger.info(f"Loss function: {criterion}")
                logger.info(f"Optimizer: {optimizer}")

                # Training Loop
                history = {
                    "epoch": [],
                    "train_loss": [],
                    "val_loss": [],
                    "train_mae": [],
                    "val_mae": [],
                    "train_rmse": [],
                    "val_rmse": [],
                    "learning_rate": [],
                    "epoch_runtime_seconds": [],
                    "gradient_norm": []
                }

                logger.info(f"Starting training for {config['optimization']['epochs']} epochs.")
                train_start_time = time.time()
                best_val_loss = float("inf")
                best_epoch = -1
                best_metrics = {}

                # Initialize the gradient scaler once at the start of training
                scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' else None

                # Model configuration to save inside checkpoints
                model_config = {
                    "architecture": config["model"]["architecture"],
                    "sequence_length": config["dataset"]["sequence_length"],
                    "predict_steps": config["dataset"]["predict_steps"],
                    "base_channels": config["model"]["base_channels"],
                    "num_groups": config["model"]["num_groups"]
                }

                for epoch in range(config["optimization"]["epochs"]):
                    epoch_start = time.perf_counter()

                    train_metrics = train_one_epoch(
                        model=model,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        logger=logger,
                        epoch=epoch,
                        scaler=scaler
                    )

                    val_metrics = evaluate_val(model, val_loader, device, criterion, logger=logger, epoch=epoch)
                    epoch_num = epoch + 1
                    current_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(val_metrics["loss"])

                    epoch_runtime_seconds = (time.perf_counter() - epoch_start)
                    train_time_seconds = time.time() - train_start_time
                    total_time_seconds = time.time() - run_start_time

                    # ─── STREAM CHRONOLOGICAL MATRIX LOSS CURVES EVERY EPOCH ───
                    mlflow.log_metrics({
                        "train/loss": train_metrics["loss"],
                        "train/mae": train_metrics["mae"],
                        "train/rmse": train_metrics["rmse"],
                        "val/loss": val_metrics["loss"],
                        "val/mae": val_metrics["mae"],
                        "val/rmse": val_metrics["rmse"],
                        "system/gradient_norm": train_metrics["gradient_norm"],
                        "system/epoch_runtime_seconds": epoch_runtime_seconds,
                        "system/current_lr": current_lr
                    }, step=epoch_num)

                    logger.info(f"Epoch [{epoch_num}/{config['optimization']['epochs']}]")

                    logger.info(
                        f"Runtime: {epoch_runtime_seconds:6.2f}s | LR: {current_lr:.2e} | "
                        f"GradNorm: {train_metrics['gradient_norm']:.3f} | BadEpochs: {scheduler.num_bad_epochs}"
                    )
                    logger.info(f"Train | Loss: {train_metrics['loss']:.6f} | MAE: {train_metrics['mae']:.6f} | RMSE: {train_metrics['rmse']:.6f}")
                    logger.info(f"Val   | Loss: {val_metrics['loss']:.6f} | MAE: {val_metrics['mae']:.6f} | RMSE: {val_metrics['rmse']:.6f}")

                    if epoch == 0 and torch.cuda.is_available():
                        peak_gpu_gb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
                        logger.info(f"Memory| Peak GPU Memory Allocated (Epoch 1): {peak_gpu_gb:.2f} GB")
                        torch.cuda.reset_peak_memory_stats(device=device)

                    # Saving
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_dir=checkpoint_dir,
                        epoch_num=epoch_num,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        model_config=model_config,
                        logger=logger
                    )
                    best_result = update_best_model(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_dir=checkpoint_dir,
                        epoch_num=epoch_num,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        model_config=model_config,
                        best_val_loss=best_val_loss,
                        logger=logger
                    )

                    if best_result is not None:
                        best_val_loss = best_result["best_val_loss"]
                        best_epoch = best_result["best_epoch"]
                        best_metrics = best_result["best_metrics"]

                    update_history(
                        history=history,
                        epoch_num=epoch_num,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        learning_rate=current_lr,
                        epoch_runtime_seconds=epoch_runtime_seconds
                    )
                    save_history(history_path, history)

                    save_experiment_info(
                        experiment_path=experiment_path,
                        train_name=train_name,
                        config=config,
                        experiment_dir=experiment_dir,
                        preprocessing_metadata=preprocessing_metadata,
                        discarded_train_sequences=train_dataset.invalid_sequence_count,
                        discarded_val_sequences=val_dataset.invalid_sequence_count,
                        num_train_samples=len(train_dataset),
                        num_val_samples=len(val_dataset),
                        num_train_batches=num_train_batches,
                        num_val_batches=num_val_batches,
                        num_parameters=num_parameters,
                        best_epoch=best_epoch,
                        best_metrics=best_metrics,
                        device=device,
                        dataset_time_seconds=dataset_time_seconds,
                        train_time_seconds=train_time_seconds,
                        total_time_seconds=total_time_seconds,
                        history=history,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        peak_vram_gb=peak_gpu_gb,
                        run_id=active_run_id, 
                        experiment_id=active_experiment_id
                    )

                # ─── PIPELINE PACKAGING TIER (Executed after training completes) ───
                if torch.cuda.is_available():
                    peak_gpu_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                    mlflow.log_metric("system/gpu_memory_usage_megabytes", peak_gpu_mb, step=0)

                mlflow.log_metric("system/total_runtime_seconds", train_time_seconds)
                runtime_human_string = f"{train_time_seconds / 3600:.2f} hours ({int(train_time_seconds / 60)} mins)"
                mlflow.log_param("env_total_runtime_display", runtime_human_string)

                # ─── CONDITIONAL MODEL REGISTRATION ───
                # Avoid registering models if this is a short-lived HPO trial
                if not is_nested_trial and config["mlflow"]["register_model"]:
                    # Fallback to the architecture token name if no explicit registry path config is defined
                    default_model_name = f"precipitation-nowcasting-{config['model']['architecture']}"
                    registry_name = config.get("mlflow", {}).get("registered_model_name", default_model_name)
                    model_desc = config.get("mlflow", {}).get("model_description", "Production candidate model.")

                    logger.info(f"Standalone run detected. Packaging model weights for Registry: {registry_name}")
                    
                    # 1. Log the model asset to the run artifacts tier natively
                    model_info = mlflow.pytorch.log_model(
                        pytorch_model=model,
                        name="model_checkpoints",
                        serialization_format="pickle",
                        registered_model_name=registry_name
                    )
                    
                    # Capture the newly created version number from the registration payload
                    registered_model_version = model_info.registered_model_version
                    # 2. SEPARATE CLIENT PASS: INJECT TAGS DIRECTLY TO THE MODEL REGISTRY
                    # This ensures your metadata tags are visible in the Models Tab, not just the Runs Tab
                    try:
                        client = mlflow.client.MlflowClient()
                        
                        # Set a high-level version description notes block
                        client.update_model_version(
                            name=registry_name,
                            version=str(registered_model_version),
                            description=f"{model_desc} | Parent Run ID: {active_run_id}"
                        )
                        
                        # Tag the model version explicitly to allow sorting/filtering in the UI
                        client.set_model_version_tag(
                            name=registry_name,
                            version=str(registered_model_version),
                            key="pipeline_stage",
                            value="Model_Training"
                        )
                        client.set_model_version_tag(
                            name=registry_name,
                            version=str(registered_model_version),
                            key="architecture",
                            value=config["model"]["architecture"]
                        )
                        client.set_model_version_tag(
                            name=registry_name,
                            version=str(registered_model_version),
                            key="run_id",
                            value=active_run_id
                        )
                        logger.info(f"🎯 Successfully pushed metadata tags to Model Version {registered_model_version}")
                        
                    except Exception as registry_tag_err:
                        logger.warning(f"⚠️ Model Registry metadata tag injection dropped: {registry_tag_err}")

                else:
                    logger.info("📌 Skipping model registration to keep Registry clean. Instead log model checkpoint as artifact.")
                    mlflow.log_artifact(str(experiment_dir / "checkpoints/_best_epoch.pt"), artifact_path="model_checkpoints")
                
                # CALL POST-TRAINING VISUALIZATION INTERNALLY
                logger.info("Invoking visualization engine inside tracking session context...")
                plot_command = [
                    sys.executable, "-m", "src.visualization.plot_train_history",
                    "--experiment-dir", str(experiment_dir)
                ]
                try:
                    subprocess.run(plot_command, check=True)
                    logger.info("Train history plot generated successfully.")
                except Exception as plot_err:
                    logger.error(f"Post-training plotting step failed: {plot_err}")

                # ─── SELECTIVE ARTIFACT UPLOAD ───
                if history_path.exists(): 
                    mlflow.log_artifact(str(history_path), artifact_path="")
                if experiment_path.exists(): 
                    mlflow.log_artifact(str(experiment_path), artifact_path="")
                    
                # Upload only the generated PNG diagnostic curves into a dedicated 'plots' folder
                for png_file in experiment_dir.glob("*.png"):
                    mlflow.log_artifact(str(png_file), artifact_path="plots")
                    
                # Flip the success marker flag to authorize hard drive cleanup
                run_concluded_successfully = True

    except Exception as pipeline_error:
        logger.error(f"❌ Pipeline crashed mid-execution: {pipeline_error}")
        raise pipeline_error

    finally:

        total_time_seconds = time.time() - run_start_time
        logger.info(f"Total time: {total_time_seconds / 60:.2f} minutes")
        logger.info("=" * 60)

        # 2. FLUSH AND DETACH THE FILE LOGGER HANDLER NATIVELY
        # This releases the system file lock and guarantees every line is written to disk
        try:
            import logging
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
        except Exception as logger_err:
            print(f"⚠️ Failed to gracefully flush logger file buffers: {logger_err}")

        # 3. USE LOW-LEVEL MLFLOW CLIENT TO UPLOAD THE COMPLETED TRACE LOG FILE
        # This operates completely outside the fluent 'with' context block
        try:
            # Locate the exact file on your disk
            log_file_path = Path("output/logs/train") / f"{train_name}.log"
            if log_file_path.exists():
                client = mlflow.client.MlflowClient()
                client.log_artifact(
                    run_id=active_run_id, 
                    local_path=str(log_file_path), 
                    artifact_path="logs"
                )
                print(f"🎯 Complete terminal trace log securely archived in MLflow run: {active_run_id}")
        except Exception as mlflow_client_err:
            print(f"⚠️ Backend client failed to archive final text logs: {mlflow_client_err}")

        # 4. AUTOMATED CLEANUP HOOK
        if run_concluded_successfully and not is_nested_trial:
            print("Training concluded cleanly. Initiating automated workspace cleanup...")
            try:
                import shutil
                if experiment_dir.exists():
                    shutil.rmtree(experiment_dir)
                    print(f"🧹 Local scratchpad workspace cleanly deleted: {experiment_dir}")
            except Exception as cleanup_err:
                print(f"Workspace cleanup encountered a soft failure notice: {cleanup_err}")
        elif is_nested_trial:
            print("📌 HPO child trial concluded. Keeping folder intact for parent sweep lifecycle orchestration.")
        else:
            print("⚠️ Training was cut short! Keeping local scratchpad for crash analysis and state recovery.")

        print("Training script execution loop finalized.")
        print("=" * 60)


if __name__ == "__main__":
    main()
