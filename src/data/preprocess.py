import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from src.utils.config import load_preprocessing_environment
from src.utils.logger import get_logger, print_configuration


def resize_array(array, target_height, target_width):
    return cv2.resize(
        array.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )


def resize_mask(mask, target_height, target_width):
    return cv2.resize(
        mask.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def preprocess_frame(frame, clip_value=1500, resize_height=256, resize_width=256):
    # Track missing data mask (-1 values) before making changes
    mask = (frame != -1).astype(bool)
    frame = frame.astype(np.float32)
    
    # Hard clip the data first to protect the upcoming INTER_AREA resize
    frame = np.clip(frame, 0, clip_value)

    # Scale spatial matrix geometries
    frame = resize_array(frame, resize_height, resize_width)
    mask = resize_mask(mask, resize_height, resize_width)

    # Apply Logarithmic Transformation to linearize the distribution
    global_log_max = np.log1p(clip_value)  
    frame = np.log1p(frame) / global_log_max

    return frame, mask


def process_file(input_path, out_dir, config, logger):
    logger.info(f"Processing: {input_path}")

    with np.load(input_path, allow_pickle=True) as d:
        data = d["data"]
        dates = d["dates"]
        miss_dates = d['miss_dates']

    h = config["dimensions"]["resize_height"]
    w = config["dimensions"]["resize_width"]
    c = config["parameters"]["clip_value"]

    processed_frames = []
    processed_masks = []

    for frame in data:
        processed_frame, processed_mask = preprocess_frame(
            frame=frame, clip_value=c, resize_height=h, resize_width=w
        )
        processed_frames.append(processed_frame)
        processed_masks.append(processed_mask)

    processed_frames = np.array(processed_frames, dtype=np.float32)
    processed_masks = np.array(processed_masks, dtype=np.bool_)

    logger.info(f"Resize: {h}x{w} | Log Transform | Clip value: {c}")

    # Inject channel dimension to output tensors: [Frames, Channels, Height, Width]
    processed_frames = processed_frames[:, np.newaxis, :, :]
    processed_masks = processed_masks[:, np.newaxis, :, :]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays separately uncompressed
    np.save(out_dir / "data.npy", processed_frames)
    np.save(out_dir / "masks.npy", processed_masks)
    np.save(out_dir / "dates.npy", dates)
    np.save(out_dir / "miss_dates.npy", miss_dates)
    
    logger.info(f"Saved uncompressed directory contents to: {out_dir}\n")



def main():
    run_start_time = time.time()

    params = load_preprocessing_environment()
    config = params["config"]
    preprocess_name = params["preprocess_name"]

    height = config["dimensions"]["resize_height"]
    width = config["dimensions"]["resize_width"]
    clip = config["parameters"]["clip_value"]
    zone = config["metadata"]["zone"]
    raw_dir = Path(config["paths"]["source_dir"])
    processed_dir = (
        Path(config["paths"]["target_dir"]) / zone / preprocess_name
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Logger Setup
    logger = get_logger(name="preprocess", log_dir="preprocess", log_file=f"{preprocess_name}.log")
    logger.info("=" * 60)
    logger.info("Preprocessing started via Infrastructure Config Workflow")
    logger.info("=" * 60)

    print_configuration(config=config, logger=logger)

    # Save preprocessing metadata
    metadata = {
        "zone": zone,
        "resize_height": height,
        "resize_width": width,
        "transform": "log1p",
        "clip_value": clip,
        "missing_value_handling": {
            "original_missing_value": -1,
            "replacement_value": 0,
            "mask_saved": True
        },
        "source_dir": str(raw_dir.absolute()),
        "processed_dir": str(processed_dir.absolute())
    }

    with open(processed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    preprocessing_start_time = time.time()
    zone_dirs = sorted(raw_dir.glob(f"{zone}_rainfall_*"))
    
    # System call to count all matching files
    cmd = f"find {raw_dir}/{zone}_rainfall_*/ -name '*.npz' | wc -l"
    num_files = int(subprocess.check_output(cmd, shell=True).decode().strip())

    if not zone_dirs:
        logger.info(f"Warning: No source directories matching pattern '{zone}_rainfall_*' discovered inside {raw_dir}")
    logger.info(f"Preprocessing {num_files} files...")

    for year_dir in zone_dirs:
        npz_files = sorted(year_dir.rglob("*.npz"))

        for input_path in npz_files:
            output_name = input_path.stem + "_processed_dir"
            output_path = processed_dir / output_name

            process_file(
                input_path=input_path,
                out_dir=output_path,
                config=config,
                logger=logger
            )
    
    preprocessing_time_seconds = time.time() - preprocessing_start_time
    total_time_seconds = time.time() - run_start_time

    logger.info(f"Preprocessing time: {preprocessing_time_seconds / 60:.2f} minutes")
    logger.info(f"Total run time: {total_time_seconds / 60:.2f} minutes")

    logger.info("=" * 60)
    logger.info("Preprocessing completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
