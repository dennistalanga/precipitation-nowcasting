import csv
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import mlflow
import torch

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
        "trial_runtime_seconds": experiment["runtime"]["total_time_seconds"]
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
    
    run_concluded_successfully = False
    active_run_id = None
    
    params = load_search_environment()
    config = params["config"]
    search_name = params["search_name"]
    param_map = params["param_map"]
    
    # Logger Setup
    logger = get_logger(name="random_search", log_dir="search", log_file=f"{search_name}.log")
    # Strictly prevent MLflow's root logger from intercepting and duplicating console lines
    logger.propagate = False
    
    # Path Setup - temporary local scratchpad workspace
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

    # Initialize mlflow tracking server URI endpoint
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "precipitation-nowcasting"))

    try:
        with mlflow.start_run(run_name=search_name) as parent_run:
            active_run_id = parent_run.info.run_id
            logger.info("=" * 60)
            logger.info(f"🚀 Initialized HPO Parent Sweep Run Node: {active_run_id}")
            logger.info("=" * 60)

            # ─── SECTION A: SYSTEM LIFECYCLE & PROVENANCE TAGS ───
            mlflow.set_tags({
                "mlflow.source.name": "src/tuning/random_search.py",
                "mlflow.source.type": "LOCAL",
                "pipeline_stage": "Hyperparameter_Optimization",
                "mlflow.run_type": "hpo_parent_sweep",
                "mlflow.note.content": "### Hyperparameter Random Search Sweep Suite\nOrchestrates discrete choice distributions, tracks structural parameter boundaries, and hosts nested trial execution frames."
            })

            # ─── SECTION B: GLOBAL HPO STRATEGY & WINDOWS METADATA ───
            mlflow.log_params({
                "hpo_strategy": "RandomSearch",
                "hpo_cv_folds": "1 (Fixed Temporal Validation Split)",
                "hpo_total_target_runs": int(total_target_runs),
                "hpo_epochs_per_trial": int(epochs),
                "model_architecture": str(model_type),
                "dataset_sequence_length": int(seq_len),
                "dataset_predict_steps": int(pred_steps),
                "dataset_processed_dir": str(config["paths"]["processed_dir"]),
                "dataset_train_window": f"{config['data_split']['train_start']}->{config['data_split']['train_end']}",
                "dataset_val_window": f"{config['data_split']['val_start']}->{config['data_split']['val_end']}"
            })

            # ─── SECTION C: DISCRETE SEARCH SPACE CHANNELS BOUNDARIES ───
            # Stringifies the raw list distributions arrays from your config file for the parameters tab
            dist_cfg = config["distributions"]
            mlflow.log_params({
                "space_learning_rate": str(dist_cfg["optimization.learning_rate"]),
                "space_weight_decay": str(dist_cfg["optimization.weight_decay"]),
                "space_batch_size": str(dist_cfg["optimization.batch_size"]),
                "space_base_channels": str(dist_cfg["model.base_channels"]),
                "space_num_groups": str(dist_cfg["model.num_groups"])
            })

            # ─── SECTION D: SYSTEM & REPRODUCIBILITY ENVIRONMENT PARAMETERS ───
            if "system" in config:
                sys_cfg = config["system"]
                mlflow.log_params({
                    "env_num_workers": int(sys_cfg.get("num_workers", 4)),
                    "env_pin_memory": str(sys_cfg.get("pin_memory", True)),
                    "env_prefetch_factor": int(sys_cfg.get("prefetch_factor", 1)),
                    "env_input_channels": int(sys_cfg.get("input_channels", 2)),
                })
            
            # Environment Provenance (Programmatic System Context tracking)
            pytorch_version = torch.__version__
            hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            mlflow.log_params({
                "env_hardware": hardware_name,
                "env_framework": f"PyTorch {pytorch_version}"
            })

            print_configuration(config=config, logger=logger)

            csv_headers = ["run", "train_name"] + list(param_map.keys()) + [
                "best_epoch", "train_loss", "train_mae", "train_rmse", "val_loss", "val_mae", "val_rmse", "trial_runtime"
            ]

            completed_trial_paths = []

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
                        f"data_split.val_end={config['data_split']['val_end']}",
                        f"runtime.hpo_parent_run_id={active_run_id}"
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
                    
                    try:
                        subprocess.run(train_command, check=True, env=None)
                    except subprocess.CalledProcessError as e:
                        logger.critical(f"Subprocess Execution Fault! Trial {run + 1} crashed with status code {e.returncode}")
                        raise e

                    metrics = load_best_metrics(train_name)
                    
                    row_data = [run, train_name]
                    for short_name in param_map.keys():
                        row_data.append(hp[short_name])
                        
                    row_data.extend([
                        metrics["best_epoch"], metrics["train_loss"], metrics["train_mae"], metrics["train_rmse"],
                        metrics["val_loss"], metrics["val_mae"], metrics["val_rmse"], metrics["trial_runtime_seconds"]
                    ])
                    
                    writer.writerow(row_data)
                    csvfile.flush()
                    logger.info(f"Successfully serialized summary parameters for trial {run + 1}\n")

                    trial_dir = Path("output/model") / train_name
                    completed_trial_paths.append(trial_dir)

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

            # ─── SELECTIVE ARTIFACT FILE UPLOAD ───
            # Syncs your flat performance tables to the root level
            if output_csv.exists():
                mlflow.log_artifact(str(output_csv), artifact_path="")

            # Upload only the generated PNG analytics curves into a dedicated 'plots' folder
            plots_folder = tuning_run_dir / "plots"
            if plots_folder.exists():
                mlflow.log_artifacts(str(plots_folder), artifact_path="plots")

            # Flip success marker to authorize scratchpad wiping
            run_concluded_successfully = True

            logger.info("=" * 60)
            logger.info("Random Search completed successfully")
            logger.info("=" * 60)

    except Exception as sweep_error:
        logger.error(f"❌ Hyperparameter optimization sweep crashed: {sweep_error}")
        raise sweep_error

    finally:
        # Calculate trailing durations regardless of execution outcome status
        total_time_seconds = time.time() - run_start_time
        logger.info(f"Total time: {total_time_seconds / 60:.2f} minutes")
        logger.info("=" * 60)

        # ─── FLUSH AND DETACH THE FILE LOGGER HANDLER NATIVELY ───
        # Releases the system write lock so the folder can be deleted without PermissionErrors
        try:
            import logging
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
        except Exception as logger_err:
            print(f"⚠️ Failed to gracefully flush logger file buffers: {logger_err}")

        # ─── LOW-LEVEL CLIENT COMPLETION LOG UPLOAD ───
        # Runs outside the 'with' block so the log file captures the final performance summaries
        if active_run_id:
            try:
                log_file_path = Path("output/logs/search") / f"{search_name}.log"
                if log_file_path.exists():
                    client = mlflow.client.MlflowClient()
                    client.log_artifact(
                        run_id=active_run_id, 
                        local_path=str(log_file_path), 
                        artifact_path="logs"
                    )
                    print(f"🎯 Complete sweep trace log safely archived in MLflow parent run: {active_run_id}")
            except Exception as mlflow_client_err:
                print(f"⚠️ Backend client failed to archive final sweep logs: {mlflow_client_err}")

        # ─── AUTOMATED CLEANUP HOOK ───
        if run_concluded_successfully:
            print("Hyperparameter search sweep finished cleanly. Initiating automated workspace cleanup...")
            try:
                import shutil
                # 1. Loop through and delete all the temporary child trial model directories
                print(f"🧹 Sweeping {len(completed_trial_paths)} local child trial scratchpads...")
                for trial_path in completed_trial_paths:
                    if trial_path.exists():
                        shutil.rmtree(trial_path)
                        print(f"   -> Deleted trial workspace: {trial_path.name}")
                
                # 2. Delete the overarching tuning parent summary folder
                if tuning_run_dir.exists():
                    shutil.rmtree(tuning_run_dir)
                    print(f"🧹 Local parent search scratchpad workspace cleanly deleted: {tuning_run_dir}")
            except Exception as cleanup_err:
                print(f"Search workspace cleanup encountered a soft failure notice: {cleanup_err}")
        else:
            print("⚠️ Optimization sweep was cut short or aborted! Keeping local scratchpad folder for state recovery.")

        print("Random search script execution loop finalized.")
        print("=" * 60)


if __name__ == "__main__":
    main()
