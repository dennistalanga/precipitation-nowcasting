import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import mlflow

from src.models.factory import get_model

# =====================================================================
# 🛠️ SECTION 1: GLOBAL METADATA LOG PARSING & UTILITIES
# =====================================================================

def extract_peak_vram_mb(log_path: Path) -> float:
    """Parses text trace logs for VRAM metrics and standardizes to Megabytes."""
    if not log_path or not log_path.exists():
        return 0.0
    vram_pattern = re.compile(r"Peak GPU Memory Allocated.*:\s*([0-9.]+)\s*(GB|MB)")
    try:
        with open(log_path, "r") as f:
            for line in f:
                match = vram_pattern.search(line)
                if match:
                    val, unit = float(match.group(1)), match.group(2)
                    return val * 1024.0 if unit == "GB" else val
    except Exception:
        pass
    return 0.0


def find_associated_log(run_dir: Path, stage: str) -> Path:
    """Finds the plain-text execution logs matching a runtime directory token."""
    project_root = run_dir.parents[2]
    logs_dir = project_root / "output/logs" / stage
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            if run_dir.name in log_file.name:
                return log_file
    return None

# =====================================================================
# 🏗️ SECTION 2: THE REUSABLE SKELETON MIGRATION ENGINE
# =====================================================================

class UnifiedMigrationEngine:
    def __init__(self, tracking_uri: str, experiment_name: str):
        print(f"Connecting to MLflow Tracking Server Gateway at: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        mlflow.set_experiment(experiment_name)
        
    def _apply_universal_system_provenance(self, stage_tag: str):
        """Applies native system keys and metadata across all run variations."""

        if "Model_Evaluation" in stage_tag:
            source_name = "src/evaluation/evaluate.py"  
        elif "Hyperparameter_Optimization" in stage_tag:
            source_name = "src/tuning/random_search.py"
        else:
            source_name = "src/training/train.py"

        mlflow.set_tags({
            "mlflow.source.name": source_name,
            "mlflow.source.type": "LOCAL",
            "mlflow.user": "dennis",
            "pipeline_stage": stage_tag
        })

    def _log_shared_dataset_lineage(self, config: dict, is_test: bool = False):
        """Maps data-centric space constraints straight into searchable fields."""
        mlflow.log_params({
            "dataset_zone": config["preprocessing"]["zone"],
            "dataset_resolution": f"{config['preprocessing']['resize_height']}x{config['preprocessing']['resize_width']}",
            "dataset_transform": config["preprocessing"]["transform"],
            "dataset_max_clip": config["preprocessing"]["clip_value"]
        })
        
        # Branch variables based on whether we are loading test or train files
        h_param_key = "test" if is_test else "hyperparameters"
        mlflow.log_params({
            "dataset_sequence_length": config[h_param_key]["sequence_length"],
            "dataset_predict_steps": config[h_param_key]["predict_steps"]
        })

    def _log_hardware_provenance(self):
        """Programmatically snapshots runtime execution environment stats."""
        pytorch_version = torch.__version__
        hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        mlflow.log_params({
            "env_hardware": hardware_name,
            "env_framework": f"PyTorch {pytorch_version}"
        })

    def execute_migration_run(self, target_dir: str, run_type: str, parent_id: str = None):
        """The master lifecycle wrapper supervising continuous execution frames."""
        run_path = Path(target_dir)
        log_stage = "evaluate" if run_type == "Model_Evaluation" else ("search" if run_type == "Hyperparameter_Optimization" else "train")
        stdout_log = find_associated_log(run_path, stage=log_stage)
        
        # Handle custom dynamic run naming schemes
        run_name = run_path.name
        if run_type == "Model_Training" and (run_path / "experiment.json").exists():
            with open(run_path / "experiment.json", "r") as f:
                run_name = json.load(f).get("train_name", run_name)
        elif run_type == "Model_Evaluation" and (run_path / "test.json").exists():
            with open(run_path / "test.json", "r") as f:
                run_name = json.load(f).get("test_name", run_name)

        # Structure parent/child nested sessions cleanly
        context = mlflow.start_run(run_id=parent_id) if (run_type == "Model_Evaluation" and parent_id) else None
        nested_flag = True if (run_type == "Model_Evaluation" and parent_id) else False

        if context:
            context.__enter__()

        try:
            with mlflow.start_run(run_name=run_name, nested=nested_flag) as run:
                active_run_id = run.info.run_id
                print(f"🚀 Processing [{run_type}] | Active Run Node UUID: {active_run_id}")

                self._apply_universal_system_provenance(stage_tag=run_type)
                self._log_hardware_provenance()
                
                # Dynamic routine routing
                if run_type == "Model_Training":
                    self._process_single_training(run_path, stdout_log, is_hpo_child=False, run_id=active_run_id)
                elif run_type == "Model_Evaluation":
                    self._process_standalone_evaluation(run_path, stdout_log)
                elif run_type == "Hyperparameter_Optimization":
                    self._process_hpo_parent(run_path, stdout_log)
                    
        finally:
            if context:
                context.__exit__(None, None, None)
            mlflow.end_run()

        print(f"🎉 Clean migration completed successfully for: {run_name}\n")

    # =====================================================================
    # 🎨 SECTION 3: RE-ADDED VARIANT PAYLOAD IMPLEMENTATIONS
    # =====================================================================

    def _process_single_training(self, run_path: Path, log_file: Path, is_hpo_child: bool = False, run_id: int = 0):
        """
        Comprehensive payload logging for deep model training metrics.
        Dynamically shares execution logic between Standalone Training and HPO Trials.
        """
        with open(run_path / "experiment.json", "r") as f: config = json.load(f)
        with open(run_path / "history.json", "r") as f: history = json.load(f)

        self._log_shared_dataset_lineage(config, is_test=False)
        mlflow.set_tag("model_status", "HPO_Trial" if is_hpo_child else "Baseline_Anchor")
        
        # All previously missed architectural and hyperparameter configs re-added
        mlflow.log_params({
            "dataset_train_window": f"{config['dataset']['train_start']}->{config['dataset']['train_end']}",
            "dataset_val_window": f"{config['dataset']['val_start']}->{config['dataset']['val_end']}",
            
            "model_architecture": config["model"]["architecture"],
            "model_base_channels": config["hyperparameters"]["base_channels"],
            "model_num_groups": config["hyperparameters"]["num_groups"],

            "train_epochs": config["hyperparameters"]["epochs"],
            "train_optimizer": config["hyperparameters"]["optimizer"],
            "train_loss_function": config["hyperparameters"]["loss_function"],
            "train_base_lr": config["hyperparameters"]["learning_rate"],
            "train_weight_decay": config["hyperparameters"]["weight_decay"],
            "train_batch_size": config["hyperparameters"]["batch_size"],
            "train_lr_scheduler": config["hyperparameters"]["scheduler"]["type"],
        })
        
        mlflow.log_metric("num_parameters", config["model"]["num_parameters"])
        
        # Comprehensive historical curves streaming loop
        epochs = history["epoch"]
        for i in range(len(epochs)):
            step_idx = int(epochs[i])
            mlflow.log_metrics({
                "train/loss": history["train_loss"][i],
                "train/mae": history["train_mae"][i],
                "train/rmse": history["train_rmse"][i],

                "val/loss": history["val_loss"][i],
                "val/mae": history["val_mae"][i],
                "val/rmse": history["val_rmse"][i],

                "system/gradient_norm": history["gradient_norm"][i],
                "system/epoch_runtime_seconds": history["epoch_runtime_seconds"][i],
                "system/current_lr": history["learning_rate"][i]
            }, step=step_idx)

        # System hardware logging
        total_seconds = config["runtime"]["total_time_seconds"]
        mlflow.log_metric("system/total_runtime_seconds", total_seconds)

        runtime_human_string = f"{total_seconds / 3600:.2f} hours ({int(total_seconds / 60)} mins)"
        mlflow.log_param("env_total_runtime_display", runtime_human_string)

        peak_vram = extract_peak_vram_mb(log_file)
        if peak_vram > 0:
            mlflow.log_metric("system/gpu_memory_usage_megabytes", peak_vram, step=0)
        # Vault complete directory trees assets
        mlflow.log_artifact(str(run_path / "experiment.json"), artifact_path="")
        mlflow.log_artifact(str(run_path / "history.json"), artifact_path="")
        if log_file and log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="logs")
        for file in run_path.glob("*.png"): 
            mlflow.log_artifact(str(file), artifact_path="plots")

        if not is_hpo_child:
            # Fallback to the architecture token name if no explicit registry path config is defined
            registry_name = "baseline_cnn_mse_loss"
            model_desc = "Baseline CNN encoder-decoder with group normalization and MSE loss."

            print(f"Standalone run detected. Packaging model weights for Registry: {registry_name}")
            
            historical_model = get_model(
                model_name=config["model"]["architecture"],
                sequence_length=config["hyperparameters"]["sequence_length"],
                predict_steps=config["hyperparameters"]["predict_steps"],
                input_channels=2, # keep at 2: (frames, masks)
                base_channels=config["hyperparameters"]["base_channels"],
                num_groups=config["hyperparameters"]["num_groups"]
            )

            # best_pt_path = Path(config["model"]["experiment_dir"]) / "checkpoints" / config["model"]["checkpoint_state"]
            best_pt_path = Path(config["model"]["experiment_dir"]) / "checkpoints/_best_epoch.pt"
            # /home/dennis/precipitation-nowcasting/output/models/20260728_111238_baseline_cnn_seq6_pred6_train/checkpoints/_best_epoch.pt
            # 2. Load the binary checkpoint onto the CPU safely to prevent memory collisions [6, 11]
            checkpoint = torch.load(best_pt_path, map_location="cpu")
            historical_model.load_state_dict(checkpoint["model_state_dict"])
            
            # 1. Log the model asset to the run artifacts tier natively
            model_info = mlflow.pytorch.log_model(
                pytorch_model=historical_model,
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
                    description=f"{model_desc} | Parent Run ID: {run_id}"
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
                    value=run_id
                )
                print(f"🎯 Successfully pushed metadata tags to Model Version {registered_model_version}")
                
            except Exception as registry_tag_err:
                print(f"⚠️ Model Registry metadata tag injection dropped: {registry_tag_err}")

        elif (run_path / "checkpoints/_best_epoch.pt").exists():
            print("📌 Skipping model registration to keep Registry clean. Instead log model checkpoint as artifact.")
            mlflow.log_artifact(str(run_path / "checkpoints/_best_epoch.pt"), artifact_path="model_checkpoints")


    def _process_standalone_evaluation(self, run_path: Path, log_file: Path):
        """Comprehensive payload logging for test-set verification matrices."""
        with open(run_path / "test.json", "r") as f: test_config = json.load(f)
        with open(run_path / "calculated_verification_metrics.json", "r") as f: metrics = json.load(f)

        self._log_shared_dataset_lineage(test_config, is_test=True)
        mlflow.log_param("dataset_test_window", f"{test_config['dataset']['test_start']}->{test_config['dataset']['test_end']}")
        mlflow.log_param("test_loss_function", test_config["test"]["loss_function"])

        # --- SECTION 3.5: LOG METEOROLOGICAL VERIFICATION SPACE BOUNDARIES ---
        met_cfg = test_config.get("meteorology", {})
        
        if met_cfg:
            # 1. Log the spatial FSS sliding windows kernel scales as a stringified list array
            if "fss_scales" in met_cfg:
                mlflow.log_param("met_fss_scales", str(met_cfg["fss_scales"]))
            
            # 2. Unpack the nested rain intensity thresholds dictionary into flat searchable keys
            # This ensures your dashboard can filter runs by their operational limits
            thresholds_cfg = met_cfg.get("thresholds", {})
            if thresholds_cfg:
                mlflow.log_params({
                    "met_threshold_light_mm_h": float(thresholds_cfg.get("Light Rain", 0.12)),
                    "met_threshold_moderate_mm_h": float(thresholds_cfg.get("Moderate Rain", 2.5)),
                    "met_threshold_heavy_mm_h": float(thresholds_cfg.get("Heavy Rain", 10.0))
                })
        
        # 1. Top-Level Core Summary Regression Scores
        overall = test_config["results"]["overall"]
        mlflow.log_metrics({
            "test_overall_loss": overall["loss"], 
            "test_overall_mae": overall["mae"],
            "test_overall_rmse": overall["rmse"]
        })

        # 2. Continuous lead-time horizon loops (MAE + RMSE)
        for step_str, values in metrics.get("horizon", {}).items():
            mlflow.log_metrics({
                "test_horizon_loss": values.get("Loss", 0.0),
                "test_horizon_mae": values.get("MAE", 0.0),
                "test_horizon_rmse": values.get("RMSE", 0.0)
            }, step=int(step_str))

        # 3. Categorical Skill Scores loop (POD, FAR, CSI, HSS, SEDS)
        for step_str, thresholds in metrics.get("categorical", {}).items():
            lead_step = int(step_str)
            for t_name, scores in thresholds.items():
                suffix = t_name.replace(" ", "_")
                mlflow.log_metrics({
                    f"test/POD/{suffix}": scores.get("POD", 0.0),
                    f"test/FAR/{suffix}": scores.get("FAR", 0.0),
                    f"test/CSI/{suffix}": scores.get("CSI", 0.0),
                    f"test/HSS/{suffix}": scores.get("HSS", 0.0),
                    f"test/SEDS/{suffix}": scores.get("SEDS", 0.0)
                }, step=lead_step)

        # 4. Fractions Skill Scores (FSS Sliding Matrices Windows)
        for step_str, thresholds in metrics.get("fss", {}).items():
            lead_step = int(step_str)
            for t_name, kernel_configs in thresholds.items():
                suffix = t_name.replace(" ", "_")
                for k_size, fss_val in kernel_configs.items():
                    mlflow.log_metric(f"test/FSS/kernel_{k_size}/{suffix}", float(fss_val), step=lead_step)

        # 5. Object-Oriented Spatial SAL Evaluations
        for step_str, sal_vectors in metrics.get("sal", {}).items():
            mlflow.log_metric("test/SAL/Structure", sal_vectors.get("S", 0.0), step=int(step_str))
            mlflow.log_metric("test/SAL/Amplitude", sal_vectors.get("A", 0.0), step=int(step_str))
            mlflow.log_metric("test/SAL/Location", sal_vectors.get("L", 0.0), step=int(step_str))
        
        total_seconds = test_config["runtime"]["total_time_seconds"]
        mlflow.log_metric("system/total_runtime_seconds", total_seconds)

        runtime_human_string = f"{total_seconds / 3600:.2f} hours ({int(total_seconds / 60)} mins)"
        mlflow.log_param("env_total_runtime_display", runtime_human_string)

        # 6. Comprehensive Vault Artifacts Storage File Infrastructure
        mlflow.log_artifact(str(run_path / "test.json"), artifact_path="")
        mlflow.log_artifact(str(run_path / "calculated_verification_metrics.json"), artifact_path="evaluation")
        
        if log_file and log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="logs")
        if (run_path / "spatial_mae_footprint.npy").exists():
            mlflow.log_artifact(str(run_path / "spatial_mae_footprint.npy"), artifact_path="evaluation")
            
        # Recursive directory sync mapping for plots and predictions trees
        if (run_path / "plots").exists():
            mlflow.log_artifacts(str(run_path / "plots"), artifact_path="plots")
        if (run_path / "predictions").exists():
            mlflow.log_artifacts(str(run_path / "predictions"), artifact_path="predictions")
    
    def _process_hpo_parent(self, run_path: Path, log_file: Path):
        """
        Payload tracking global multi-worker optimization tuning spaces.
        Automatically orchestrates and nests completed individual child trials.
        """
        results_csv = run_path / "random_search_results.csv"
        if not results_csv.exists():
            print(f"⚠️ Warning: Missing summary matrix file inside {run_path.name}. Skipping trial indexing.")
            return

        # Log HPO high-level metrics summary table
        mlflow.log_artifact(str(results_csv), artifact_path="")
        df_tuning = pd.read_csv(results_csv)
        
        # Log global search boundaries metadata
        mlflow.log_params({
            "hpo_strategy": "RandomSearch",
            "hpo_cv_folds": "1 (Fixed Temporal Validation Split)",
            "hpo_total_completed_trials": len(df_tuning),
            "hpo_epochs_per_trial": 20,

            "space_learning_rate": "[0.0003, 0.001, 0.003, 0.01]",
            "space_weight_decay": "[0.0, 0.01, 0.05, 0.1]",
            "space_batch_size": "[16, 32]",
            "space_base_channels": "[16, 24, 32]",
            "space_num_groups": "[2, 4, 8]"
        })
        
        if (run_path / "plots").exists():
            mlflow.log_artifacts(str(run_path / "plots"), artifact_path="plots")
        if log_file and log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="logs")

        # ─── NESTED CHILD LOOP FOR INDIVIDUAL RUN TRIALS ───
        print(f"\nProcessing {len(df_tuning)} sequential tuning child runs...")
        models_root = Path("output/models")
        for _, row in df_tuning.iterrows():
            trial_name = str(row["train_name"])
            trial_dir = models_root / trial_name
            
            if not trial_dir.exists():
                print(f"⚠️ Warning: Checkpoint folder not found for trial {trial_name}. Skipping.")
                continue
                
            # Resolve chronological log timelines for this trial loop instance
            child_log = find_associated_log(trial_dir, stage="train")
            
            # Nest execution wrapper cleanly using start/end markers
            with mlflow.start_run(run_name=trial_name, nested=True) as child_run:
                active_child_run_id = child_run.info.run_id
                print(f" -> Indexing Child Run ID: {active_child_run_id} [{trial_name[-10:]}]")
                self._apply_universal_system_provenance(stage_tag="HPO_TRIAL")
                self._log_hardware_provenance()
                
                # Re-route parameters straight into the shared training payload
                self._process_single_training(
                    run_path=trial_dir, 
                    log_file=child_log, 
                    is_hpo_child=True,
                    run_id=active_child_run_id
                )


# =====================================================================
# 🚀 SECTION 4: UNIFIED RUNTIME CONTROLLER EXECUTION ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    URI = "http://localhost:5000"
    EXP_NAME = "precipitation-nowcasting"
    migrator = UnifiedMigrationEngine(tracking_uri=URI, experiment_name=EXP_NAME)

    # Toggle string switchboard driver
    CHOSEN_STAGE = "Model_Evaluation"  # "Model_Training" | "Model_Evaluation" | "Hyperparameter_Optimization"
    
    if CHOSEN_STAGE == "Model_Training":
        migrator.execute_migration_run(
            target_dir="output/models/20260728_111238_baseline_cnn_seq6_pred6_train",
            run_type="Model_Training"
        )
    elif CHOSEN_STAGE == "Model_Evaluation":
        MAPPED_PARENT_UUID = "6c231d65559f467e8021a5dea840374c" 
        migrator.execute_migration_run(
            target_dir="output/evaluation/20260729_222338_baseline_cnn_seq6_pred6_test",
            # target_dir="output/evaluation/20260730_012552_baseline_cnn_seq6_pred6_test",
            run_type="Model_Evaluation",
            parent_id=MAPPED_PARENT_UUID
        )
    elif CHOSEN_STAGE == "Hyperparameter_Optimization":
        migrator.execute_migration_run(
            target_dir="output/tuning/20260727_181201_random_search_lr-decay-batch_size-base_channels-num_groups",
            run_type="Hyperparameter_Optimization"
        )
