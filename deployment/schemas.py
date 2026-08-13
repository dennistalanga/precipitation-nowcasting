from pydantic import BaseModel, Field
from typing import List

class ModelMetadata(BaseModel):
    """Provides system details about the currently loaded experiment model."""
    model_name: str
    sequence_length: int
    predict_steps: int
    input_resolution: List[int] = Field(..., description="[Height, Width] loaded from config")
    clip_value: float

class NowcastResponse(BaseModel):
    """Structured response schema returned back to the API client."""
    success: bool
    prediction_shape: List[int] = Field(..., description="Shape of the output tensor: [Steps, H, W]")
    predictions: List[List[List[float]]] = Field(..., description="Nested float list containing sequence predictions")
