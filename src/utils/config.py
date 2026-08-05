import argparse
from datetime import datetime
import json
from pathlib import Path
import yaml


def load_preprocessing_metadata(processed_dir):
    if not processed_dir:
        return {}
    metadata_path = Path(processed_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}
    

def load_training_metadata(experiment_dir):
    if not experiment_dir:
        return {}
    experiment_file = Path(experiment_dir) / "experiment.json"
    if experiment_file.exists():
        with open(experiment_file, "r") as f:
            return json.load(f)
    return {}


def get_latest_directory(root_dir_path, error_message="No directories discovered."):
    root = Path(root_dir_path)
    if not root.exists():
        raise FileNotFoundError(f"The directory '{root}' does not exist.")
    
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(error_message)
        
    return sorted(subdirs)[-1]


def apply_dynamic_overrides(config, opts_list):
    """
    Parses dynamic arguments like ['query.target=500', 'query.fps=6']
    and injects them natively directly into the loaded nested config dictionary.
    """
    for opt in opts_list:
        if "=" not in opt:
            continue
        key_path, value_str = opt.split("=", 1)
        
        try:
            if "." in value_str and "e" not in value_str.lower():
                value = float(value_str)
            elif value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            elif "e" in value_str.lower() or "." in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            value = value_str

        keys = key_path.split(".")
        current_level = config
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})
        current_level[keys[-1]] = value
        
    return config


def load_preprocessing_environment():
    parser = argparse.ArgumentParser(description="Preprocess Radar Data Archives")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing YAML config")
    parser.add_argument("--opts", nargs="*", default=[], help="Dynamic field.key=value overrides")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = apply_dynamic_overrides(config, args.opts)

    h = config.get("dimensions", {}).get("resize_height", 256)
    w = config.get("dimensions", {}).get("resize_width", 256)
    c = config.get("parameters", {}).get("clip_value", 1500.0)
    
    preprocess_name = config.get("preprocess_name")
    if not preprocess_name:
        preprocess_name = f"resize{h}x{w}_log_clip{c:g}"

    runtime_params = {
        "preprocess_name": preprocess_name,
        "config": config
    }
    return runtime_params 


def load_stratification_environment():
    parser = argparse.ArgumentParser(description="Characterize Processed Radar Archives")
    parser.add_argument("--config", type=str, required=True, help="Path to stratification YAML config")
    parser.add_argument("--opts", nargs="*", default=[], help="Dynamic field.key=value overrides")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = apply_dynamic_overrides(config, args.opts)
    
    strat_name = config.get("strat_name")
    if not strat_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zone = config.get("paths", {}).get("zone", "NW")
        strat_name = f"{timestamp}_{zone}_archive_characterization"

    preprocessing_metadata = load_preprocessing_metadata(config.get("paths", {}).get("processed_dir"))
    runtime_params = {
        "strat_name": strat_name,
        "preprocessing_metadata": preprocessing_metadata,
        "config": config
    }
    return runtime_params


def load_training_environment():
    parser = argparse.ArgumentParser(description="Train Weather Nowcasting Models")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument("--opts", nargs="*", default=[], help="Dynamic field.key=value overrides")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = apply_dynamic_overrides(config, args.opts)

    train_name = config.get("train_name")
    if not train_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = config.get("model", {}).get("architecture", "baseline")
        seq = config.get("dataset", {}).get("sequence_length", 6)
        pred = config.get("dataset", {}).get("predict_steps", 6)
        train_name = f"{timestamp}_{arch}_seq{seq}_pred{pred}_train"
    
    preprocessing_metadata = load_preprocessing_metadata(config.get("paths", {}).get("processed_dir"))
    runtime_params = {
        "train_name": train_name,
        "preprocessing_metadata": preprocessing_metadata,
        "config": config
    }
    return runtime_params


def load_search_environment():
    parser = argparse.ArgumentParser(description="Orchestrate Hyperparameter Random Search Sweeps")
    parser.add_argument("--config", type=str, required=True, help="Path to search space YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    param_map = {}  
    param_labels = []
    for full_path in config["distributions"].keys():
        short_name = full_path.split(".")[-1]
        param_map[short_name] = full_path
        label = short_name.replace("learning_rate", "lr").replace("weight_decay", "decay")
        param_labels.append(label)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_name = f"{timestamp}_random_search_" + "-".join(param_labels)
    
    runtime_params = {
        "search_name": search_name,
        "param_map": param_map,
        "config": config
    }
    return runtime_params


def load_evaluation_environment():
    parser = argparse.ArgumentParser(description="Evaluate Trained Radar Nowcasting Models")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation YAML config")
    parser.add_argument("--opts", nargs="*", default=[], help="Dynamic field.key=value overrides")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = apply_dynamic_overrides(config, args.opts)

    experiment_dir_str = config.get("paths", {}).get("experiment_dir", "latest")
    if not experiment_dir_str or experiment_dir_str == "latest":
        experiment_dir = get_latest_directory(
            root_dir_path="output/models",
            error_message="No experiment folders found inside output/models."
        )
    else:
        experiment_dir = Path(experiment_dir_str)
    
    preprocessing_metadata = load_preprocessing_metadata(config.get("paths", {}).get("processed_dir"))
    training_metadata = load_training_metadata(experiment_dir)

    test_name = config.get("test_name")
    if not test_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_architecture = training_metadata["model"]["architecture"]
        sequence_length = training_metadata["hyperparameters"]["sequence_length"]
        predict_steps = training_metadata["hyperparameters"]["predict_steps"]
        test_name = f"{timestamp}_{model_architecture}_seq{sequence_length}_pred{predict_steps}_test"
    
    runtime_params = {
        "test_name": test_name,
        "experiment_dir": experiment_dir,
        "preprocessing_metadata": preprocessing_metadata,
        "training_metadata": training_metadata,
        "config": config,
    }
    return runtime_params


def load_plotting_environment():
    parser = argparse.ArgumentParser(description="Render Geospatial Weather Radar Forecasts")
    parser.add_argument("--config", type=str, required=True, help="Path to plotting YAML config")
    parser.add_argument("--opts", nargs="*", default=[], help="Dynamic field.key=value overrides")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = apply_dynamic_overrides(config, args.opts)

    test_dir_cfg = config.get("paths", {}).get("test_dir", "latest")
    if not test_dir_cfg or test_dir_cfg == "latest":
        test_dir = get_latest_directory(
            root_dir_path="output/evaluation",
            error_message="No model evaluation run folders discovered in output/evaluation/"
        )
    else:
        test_dir = Path(test_dir_cfg)

    runtime_params = {
        "predictions_dir": test_dir / "predictions",
        "radar_coords": Path(config["paths"]["radar_coords"]),
        "query": config.get("query", {}).get("target", 450),
        "fps": config.get("query", {}).get("fps", 2),
        "test_dir": test_dir,
        "config": config
    }
    return runtime_params
