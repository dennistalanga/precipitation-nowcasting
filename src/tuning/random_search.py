import csv
from datetime import datetime
import json
os_imported = True # included in logic below
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from src.utils.config import load_search_environment
from src.utils.logger import get_logger, print_configuration


def load_best_metrics(train_name, model_output_dir="output/models"):
    experiment_path = Path(model_output_dir) / train_name / "experiment.json"
    with open(experiment_path, "r") as f:
        experiment = json.load(f)
        
    best = experiment["model"]["best_metrics"]
    return {
        "best_epoch": experiment["model"]["best_epoch"],
        "val_loss": best["val_loss"],
        "val_mae": best["val_mae"],
        "val_rmse": best["val_rmse"],
        "train_loss": best["train_loss"],
        "train_mae": best["train_mae"],
        "train_rmse": best["train_rmse"],
    }


def get_resume_directory(config, logger):
    resume_cfg = config.get("resume_settings", {})
    if not resume_cfg.get("enabled", False):
        return None, 0

    tuning_root = Path("output/tuning")
    if not tuning_root.exists():
        return None, 0

    specific_path = resume_cfg.get("directory_path")
    if specific_path:
        tuning_run_dir = Path(specific_path)
    else:
        subdirs = [d for d in tuning_root.iterdir() if d.is_dir()]
        if not subdirs:
            logger.info("Resume enabled, but no prior tuning directories exist. Starting fresh.")
            return None, 0
        
        subdirs.sort(key=lambda d: d.name)
        tuning_run_dir = subdirs[-1]

    output_csv = tuning_run_dir / "random_search_results.csv"
    if not output_csv.exists():
        logger.info(f"Target directory {tuning_run_dir} found, but missing random_search_results.csv. Starting fresh.")
        return None, 0

    try:
        with open(output_csv, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header:
                rows = list(reader)
                if rows:
                    completed_runs = int(rows[-1][0]) + 1
                    logger.info(f"Resuming search in: {tuning_run_dir}")
                    logger.info(f"Next trial run index will be: {completed_runs + 1}")
                    return tuning_run_dir, completed_runs
    except Exception as e:
        logger.error(f"Failed parsing CSV for resuming: {e}. Starting fresh.")
        
    return None, 0


def main():
    run_start_time = time.time()
    
    params = load_search_environment()
    config = params["config"]
    search_name = params["search_name"]
    param_map = params["param_map"]
    
    # Logger Setup
    logger = get_logger(name="random_search", log_dir="search", log_file=f"{search_name}.log")
    
    # Path Setup
    out_dir = Path("output/tuning") / search_name
    tuning_run_dir, start_run_idx = get_resume_directory(config, logger)
    is_resuming = (tuning_run_dir is not None)

    if not is_resuming:
        tuning_run_dir = out_dir
        tuning_run_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = tuning_run_dir / "random_search_results.csv"

    logger.info("=" * 60)
    logger.info(f"{'Resuming' if is_resuming else 'Starting'} Hyperparameter Random Search Sweep")
    logger.info("=" * 60)

    settings = config["search_settings"]
    total_target_runs = settings["runs"]
    epochs = settings["epochs"]
    model_type = settings["model"]
    seq_len = settings["sequence_length"]
    pred_steps = settings["predict_steps"]

    print_configuration(config=config, logger=logger)

    csv_headers = ["run", "train_name"] + list(param_map.keys()) + [
        "best_epoch", "train_loss", "train_mae", "train_rmse", "val_loss", "val_mae", "val_rmse"
    ]
    file_mode = "a" if is_resuming else "w"

    with open(output_csv, file_mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not is_resuming:
            writer.writerow(csv_headers)

        for run in range(start_run_idx, total_target_runs):
            hp = {}
            for short_name, full_path in param_map.items():
                hp[short_name] = random.choice(config["distributions"][full_path])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_name = f"{timestamp}_search_{run:03d}"

            train_command = [
                sys.executable, "-m", "src.training.train",
                "--config", "configs/train_baseline.yml",
                "--opts",
                f"train_name={train_name}",
                f"optimization.epochs={epochs}",
                f"model.architecture={model_type}",
                f"dataset.sequence_length={seq_len}",
                f"dataset.predict_steps={pred_steps}",
                f"paths.processed_dir={config['paths']['processed_dir']}",
                f"data_split.train_start={config['data_split']['train_start']}",
                f"data_split.train_end={config['data_split']['train_end']}",
                f"data_split.val_start={config['data_split']['val_start']}",
                f"data_split.val_end={config['data_split']['val_end']}"
            ]

            if "system" in config:
                for sys_k, sys_v in config["system"].items():
                    dest = "model" if sys_k == "num_groups" else "dataset"
                    train_command.append(f"{dest}.{sys_k}={sys_v}")

            for short_name, hp_val in hp.items():
                full_path = param_map[short_name]
                train_command.append(f"{full_path}={hp_val}")

            logger.info("=" * 60)
            logger.info(f"Executing Trial Run {run + 1}/{total_target_runs} [{train_name}]")
            logger.info("=" * 60)
            
            # Isolated environment initialization block to kill parameter bleedthrough
            env_override = os.environ.copy()
            # Forces PyTorch's custom allocator to dump state history completely
            env_override["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            # Forces CUDA driver context to flush caches
            env_override["CUDA_CACHE_DISABLE"] = "0"
            
            try:
                # Synchronous call completely isolates memory and execution scopes
                subprocess.run(train_command, check=True, env=env_override)
            except subprocess.CalledProcessError as e:
                logger.critical(f"Subprocess Execution Fault! Trial {run + 1} crashed with status code {e.returncode}")
                raise e

            metrics = load_best_metrics(train_name)
            
            row_data = [run, train_name]
            for short_name in param_map.keys():
                row_data.append(hp[short_name])
                
            row_data.extend([
                metrics["best_epoch"], metrics["train_loss"], metrics["train_mae"], metrics["train_rmse"],
                metrics["val_loss"], metrics["val_mae"], metrics["val_rmse"]
            ])
            
            writer.writerow(row_data)
            csvfile.flush()
            logger.info(f"Successfully serialized summary parameters for trial {run + 1}\n")

    logger.info("=" * 60)
    logger.info("All trials completed. Generating search analytics matrix...")
    logger.info("=" * 60)
    
    summary_plot_command = [
        sys.executable, "-m", "src.visualization.plot_random_search",
        "--results-dir", str(tuning_run_dir)
    ]
    try:
        subprocess.run(summary_plot_command, check=True)
        logger.info(f"Random search diagnostics plots created inside: {tuning_run_dir / 'plots/'}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Visualization pipeline completed execution with an active error code flag: {e}")
    
    total_time_seconds = time.time() - run_start_time

    logger.info(f"Total time: {total_time_seconds / 60:.2f} minutes")

    logger.info("=" * 60)
    logger.info("Random Search completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
