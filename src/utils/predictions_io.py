import json
from pathlib import Path
import numpy as np


def flush_chunk_to_disk(
    batch_inputs,
    batch_preds,
    batch_targets,
    batch_masks,
    batch_dates,
    test_dir,
    chunk_idx,
):
    predictions_dir = Path(test_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    raw_inputs = np.concatenate(batch_inputs, axis=0)
    inputs = raw_inputs[:, :, 0, :, :]
    input_masks = raw_inputs[:, :, 1, :, :].astype(np.bool_)
    
    predictions = np.concatenate(batch_preds, axis=0)
    targets = np.concatenate(batch_targets, axis=0)
    target_masks = np.concatenate(batch_masks, axis=0).astype(np.bool_)
    dates = np.array(batch_dates)

    file_name = f"chunk_{chunk_idx:03d}.npz"
    chunk_path = predictions_dir / file_name

    np.savez_compressed(
        chunk_path,
        inputs=inputs,
        input_masks=input_masks,
        predictions=predictions,
        targets=targets,
        target_masks=target_masks,
        dates=dates,
    )

    return file_name, len(predictions), dates


def load_prediction_sample(predictions_dir, query):

    predictions_dir = Path(predictions_dir)
    manifest_path = predictions_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing required lookup catalog at: {manifest_path}"
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Normalize and resolve the index or timestamp query
    query_str = str(query).strip().replace(" ", "T")
    if query_str.isdigit():
        global_idx = query_str
    else:
        # Standardize the query string format using numpy's datetime formatter
        try:
            standardized_dt = np.datetime64(query_str, 's')
            query_str = str(standardized_dt).replace(" ", "T")
        except ValueError:
            pass # Fall back to the original string if it's an unparseable format

        if query_str not in manifest.get("date_to_global", {}):
            raise KeyError(
                f"Target timestamp '{query}' (standardized to '{query_str}') could not be matched inside prediction logs."
            )
        global_idx = str(manifest["date_to_global"][query_str])

    # Boundary safety verification
    if global_idx not in manifest["global_to_local"]:
        raise IndexError(
            f"Resolved index position {global_idx} falls outside dataset total boundaries."
        )

    file_name, local_idx = manifest["global_to_local"][global_idx]
    target_chunk_file = predictions_dir / file_name

    if not target_chunk_file.exists():
        raise FileNotFoundError(
            f"Missing target file chunk segment: {target_chunk_file}"
        )

    # Read array slice out of targeted compressed archive
    with np.load(target_chunk_file, allow_pickle=True) as data:
        raw_date = data["dates"][local_idx]
        if isinstance(raw_date, (int, float, np.integer, np.floating)):
            resolved_date = np.datetime64(int(raw_date), "s")
        else:
            resolved_date = np.array(raw_date, dtype="datetime64[s]")

        sample = {
            "inputs": data["inputs"][local_idx],
            "input_masks": data["input_masks"][local_idx],
            "predictions": data["predictions"][local_idx],
            "targets": data["targets"][local_idx],
            "target_masks": data["target_masks"][local_idx],
            "date": resolved_date,
            "resolved_global_idx": int(global_idx),
        }

    return sample
