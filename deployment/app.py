import os
import logging
import torch
import numpy as np
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File

from src.models.factory import get_model  
from src.utils.config import get_latest_directory, load_training_metadata

from deployment.schemas import ModelMetadata, NowcastResponse
from deployment.pipeline import transform_sequence_to_tensor

# Get Uvicorn's native error logger to match the default API style
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Precipitation Nowcasting Service", version="1.0.0")

# Global placeholders for dynamic environment configuration states
MODEL = None
TRAINING_METADATA = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def initialize_inference_environment():
    """Locates the experiment tracking runs and dynamically sets up model weights."""
    global MODEL, TRAINING_METADATA, DEVICE
    
    logger.info("Initializing Live Machine Learning Inference Deployment Engine")
    
    try:
        # Resolve target artifact registry folders (handles docker parameters vs local runs)
        exp_dir_env = os.getenv("EXPERIMENT_DIR", "latest")
        if exp_dir_env == "latest":
            logger.info("Searching 'output/models' to locate the latest active experiment run...")
            experiment_dir = get_latest_directory(
                root_dir_path="output/models",
                error_message="No experiment folders found inside output/models."
            )
        else:
            experiment_dir = Path(exp_dir_env)

        logger.info(f"Targeting experiment artifact registry directory: {experiment_dir.name}")

        # Extract configuration arrays
        TRAINING_METADATA = load_training_metadata(experiment_dir)
        if not TRAINING_METADATA:
            raise FileNotFoundError(f"Could not load valid experiment tracking data from {experiment_dir}")

        # Locate model checkpoints
        checkpoint_path = experiment_dir / "checkpoints" / "_best_epoch.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Weight checkpoint file not found at {checkpoint_path}")

        # Pull structural properties out of the nested configuration map
        model_architecture = TRAINING_METADATA["model"]["architecture"]
        sequence_length = TRAINING_METADATA["hyperparameters"]["sequence_length"]
        predict_steps = TRAINING_METADATA["hyperparameters"]["predict_steps"]
        base_channels = TRAINING_METADATA["hyperparameters"]["base_channels"]
        input_channels = TRAINING_METADATA["hyperparameters"].get("input_channels", 2)
        num_groups = TRAINING_METADATA["hyperparameters"].get("num_groups", 4)

        logger.info(f"Assembling network architecture blueprint: {model_architecture}")
        
        # Initialize model architecture
        MODEL = get_model(
            model_name=model_architecture,
            sequence_length=sequence_length,
            predict_steps=predict_steps,
            base_channels=base_channels,
            input_channels=input_channels,
            num_groups=num_groups
        )
        
        # Bind weights to active execution processor hardware
        checkpoint_data = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Extract weights if wrapped in an epoch checkpoint state dictionary map
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            MODEL.load_state_dict(checkpoint_data["model_state_dict"])
        else:
            MODEL.load_state_dict(checkpoint_data)
            
        MODEL.to(DEVICE)
        MODEL.eval()
        
        num_params = sum(p.numel() for p in MODEL.parameters())
        logger.info(f"Successfully deployed network ({num_params:,} parameters) onto hardware target: {DEVICE}")
        
    except Exception as e:
        logger.error(f"CRITICAL SYSTEM FAILURE: Deployment environment could not start: {str(e)}")
        raise RuntimeError(e)

@app.get("/meta", response_model=ModelMetadata)
def get_service_metadata():
    """Exposes current model architecture parameters and input dimensions to external clients."""
    if not TRAINING_METADATA:
        raise HTTPException(status_code=503, detail="Service configuration state is uninitialized.")
        
    model_cfg = TRAINING_METADATA.get("model", {})
    hyper_cfg = TRAINING_METADATA.get("hyperparameters", {})
    prep_cfg = TRAINING_METADATA.get("preprocessing", {})

    return ModelMetadata(
        model_name=model_cfg.get("architecture", "unknown"),
        sequence_length=hyper_cfg.get("sequence_length", 6),
        predict_steps=hyper_cfg.get("predict_steps", 6),
        input_resolution=[
            model_cfg.get("resize_height", 256),
            model_cfg.get("resize_width", 256)
        ],
        clip_value=prep_cfg.get("clip_value", 1500.0),
        additional_hyperparameters=hyper_cfg
    )

@app.post("/predict", response_model=NowcastResponse)
async def generate_nowcast(file: UploadFile = File(...)):
    """Receives a raw NumPy sequence file [Sequence, H, W] and performs real-time inference."""
    logger.info(f"Received raw execution request payload: '{file.filename}'")
    
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Invalid binary format payload. Must be a .npy file.")
        
    try:
        contents = await file.read()
        raw_sequence = np.load(BytesIO(contents))
        
        # 1. Safely extract configuration scalars from the nested metadata dictionary
        model_cfg = TRAINING_METADATA.get("model", {})
        hyper_cfg = TRAINING_METADATA.get("hyperparameters", {})
        prep_cfg = TRAINING_METADATA.get("preprocessing", {})

        seq_len = hyper_cfg.get("sequence_length", 6)
        h = model_cfg.get("resize_height", 256)
        w = model_cfg.get("resize_width", 256)
        clip_value = prep_cfg.get("clip_value", 1500.0)

        # 2. Pass the explicit, required positional arguments to the tensor bridge
        input_tensor = transform_sequence_to_tensor(
            raw_sequence=raw_sequence, 
            seq_len=seq_len, 
            h=h, 
            w=w, 
            clip_value=clip_value
        )
        input_tensor = input_tensor.to(DEVICE)
        logger.info(f"Completed input array tensor preprocessing conversions. Moving data tensors to {DEVICE}...")
        
        # Execute forward pass within a zero-gradient inference loop
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                prediction = MODEL(input_tensor)
                
        logger.info("Model forward-pass execution completed successfully. Formatting structured predictions...")
        
        # Drop batch layout dimensions from [1, Steps, H, W] down to [Steps, H, W]
        prediction_numpy = prediction.squeeze(0).cpu().numpy()
        
        return NowcastResponse(
            success=True,
            prediction_shape=list(prediction_numpy.shape),
            predictions=prediction_numpy.tolist()
        )
        
    except ValueError as val_err:
        logger.error(f"Validation failure during input stream conversion transformations: {str(val_err)}")
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        logger.error(f"Inference execution engine exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference execution engine failure: {str(e)}")
