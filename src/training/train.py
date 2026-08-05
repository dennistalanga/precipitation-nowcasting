from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

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
    optimizer, scheduler
):
    experiment_info = {
        "train_name": train_name,
        "timestamp": datetime.now().isoformat(),
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
            "device": str(device),
            "dataset_creation_time_seconds": dataset_time_seconds,
            "training_time_seconds": train_time_seconds,
            "total_time_seconds": total_time_seconds
        },
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


def main():
    run_start_time = time.time()
    
    params = load_training_environment()
    config = params["config"]
    train_name = params["train_name"]
    preprocessing_metadata = params["preprocessing_metadata"]

    # Logger Setup
    logger = get_logger(name="train", log_dir="train", log_file=f"{train_name}.log")
    logger.info("=" * 60)
    logger.info("Training started via Infrastructure Config Workflow")
    logger.info("=" * 60)

    # Path Setup
    experiment_dir = create_experiment_dir(config["paths"]["model_output_dir"], train_name)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = experiment_dir / "history.json"
    experiment_path = experiment_dir / "experiment.json"

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

    # Model Setup
    logger.info("Initializing model...")
    model, device, num_parameters = create_model(
        model_architecture=config["model"]["architecture"],
        sequence_length=config["dataset"]["sequence_length"],
        predict_steps=config["dataset"]["predict_steps"],
        input_channels=config["dataset"]["input_channels"],
        base_channels=config["model"]["base_channels"],
        num_groups=config["model"]["num_groups"]
    )

    logger.info(f"Using device: {device}")
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Parameters: {num_parameters:,}")

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
            scheduler=scheduler
        )

    # Post-training visualization
    logger.info("=" * 60)
    logger.info("Training complete. Invoking visualization engine...")
    logger.info("=" * 60)

    plot_command = [
        sys.executable, "-m", "src.visualization.plot_train_history",
        "--experiment-dir", str(experiment_dir)
    ]

    try:
        subprocess.run(plot_command, check=True)
        logger.info(f"Train history plot generated successfully inside: {experiment_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Post-training plotting failed to render graphics: {e}")
    except Exception as e:
        logger.error(f"Unexpected visualization system error occurred: {e}")

    train_time_seconds = time.time() - train_start_time
    total_time_seconds = time.time() - run_start_time

    logger.info(f"Total training time: {train_time_seconds / 60:.2f} minutes")
    logger.info(f"Total time: {total_time_seconds / 60:.2f} minutes")

    logger.info("=" * 60)
    logger.info("Training completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
