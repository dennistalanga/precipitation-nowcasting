import cv2
import torch
import numpy as np

def resize_array(array: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resizes the continuous radar grid using area interpolation."""
    return cv2.resize(
        array.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

def resize_mask(mask: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resizes the logical binary validity mask using nearest interpolation."""
    return cv2.resize(
        mask.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

def preprocess_live_frame(frame: np.ndarray, clip_value: float, h: int, w: int) -> np.ndarray:
    """
    Transforms a single raw radar snapshot into a combined 2-channel array [Data, Mask].
    Uses parameters loaded dynamically from experiment metadata.
    """
    # 1. Identify missing data tokens (-1) before modification
    mask = (frame != -1).astype(bool)
    frame = frame.astype(np.float32)
    
    # 2. Apply clip limits to protect spatial interpolation stability
    frame = np.clip(frame, 0, clip_value)

    # 3. Handle resolution scaling dynamically
    frame = resize_array(frame, h, w)
    mask = resize_mask(mask, h, w)

    # 4. Apply Logarithmic Transformation to normalize distributions
    global_log_max = np.log1p(clip_value)  
    frame = np.log1p(frame) / global_log_max

    # 5. Package components together into a 2-channel feature tensor slice
    frame_channel = frame[np.newaxis, :, :]
    mask_channel = mask.astype(np.float32)[np.newaxis, :, :]
    
    return np.concatenate((frame_channel, mask_channel), axis=0)

def transform_sequence_to_tensor(raw_sequence: np.ndarray, seq_len: int, h: int, w: int, clip_value: float) -> torch.Tensor:
    """
    Takes an un-preprocessed 3D array [Sequence, Raw_H, Raw_W] and outputs a structured
    5D tensor [1, Sequence, Channels=2, H, W] matching the loaded model config specs.
    """
    if raw_sequence.ndim != 3 or raw_sequence.shape[0] != seq_len:
        raise ValueError(
            f"Expected input sequence length of {seq_len} with 3 dimensions [Seq, H, W]. "
            f"Received shape: {raw_sequence.shape}"
        )

    processed_slices = []
    for slice_idx in range(seq_len):
        two_channel_slice = preprocess_live_frame(raw_sequence[slice_idx], clip_value, h, w)
        processed_slices.append(two_channel_slice)

    # Combine array down to [Sequence, Channels=2, Height, Width]
    sequence_array = np.array(processed_slices, dtype=np.float32)
    
    # Add artificial batch dimension [1, Sequence, Channels, Height, Width]
    return torch.from_numpy(sequence_array).unsqueeze(0)
