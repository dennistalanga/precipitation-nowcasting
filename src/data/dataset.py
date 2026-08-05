from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def get_candidate_files(processed_dir, start_datetime, end_datetime):

    processed_dir = Path(processed_dir)
    start = datetime.fromisoformat(start_datetime)
    end = datetime.fromisoformat(end_datetime)
    candidate_keys = set()

    current = datetime(start.year, start.month, 1)

    while current <= end:
        for part in (1, 2, 3):
            candidate_keys.add((current.year, current.month, part))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    candidate_keys = sorted(candidate_keys)
    expanded = []

    for year, month, part in candidate_keys:
        expanded.append((year, month, part))

    # Add one archive before as safety padding
    first = candidate_keys[0]
    y, m, p = first
    if p > 1:
        expanded.append((y, m, p - 1))
    elif m > 1:
        expanded.append((y, m - 1, 3))
    else:
        expanded.append((y - 1, 12, 3))

    # Add one archive after as safety padding
    last = candidate_keys[-1]
    y, m, p = last
    if p < 3:
        expanded.append((y, m, p + 1))
    elif m < 12:
        expanded.append((y, m + 1, 1))
    else:
        expanded.append((y + 1, 1, 1))

    files = []
    for year, month, part in sorted(set(expanded)):
        # Update the naming format to reflect the new directory structure
        dirname = (f"rainfall_NW_{year}_{month:02d}.{part}_processed_dir")
        path = processed_dir / dirname
        if path.exists():
            files.append(path)

    return sorted(files)


class RadarDataset(Dataset):

    def __init__(self, processed_dir, start_datetime, end_datetime, sequence_length=6, predict_steps=6):
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps

        self.archive_paths = []
        self.dates = []
        self.start_offsets = []  

        processed_dir = Path(processed_dir)
        files = get_candidate_files(processed_dir, start_datetime, end_datetime)

        start_np = np.datetime64(start_datetime)
        end_np = np.datetime64(end_datetime)

        for file in files:
            dates = np.load(file / "dates.npy", allow_pickle=True)
            range_mask = (dates >= start_np) & (dates <= end_np)
            filtered_dates = dates[range_mask]

            if len(filtered_dates) == 0:
                continue
            
            self.archive_paths.append(file)
            self.dates.append(filtered_dates)
            self.start_offsets.append(int(np.argmax(range_mask)))

        if len(self.dates) == 0:
            raise ValueError("No data found in requested range.")

        self.valid_indices = []
        self.invalid_sequence_count = 0
        expected_delta = np.timedelta64(5, "m")

        for archive_id in range(len(self.dates)):
            current_dates = self.dates[archive_id]

            if archive_id < len(self.dates) - 1:
                combined_dates = np.concatenate((current_dates, self.dates[archive_id + 1]))
            else:
                combined_dates = current_dates

            for filtered_start_idx in range(len(current_dates)):
                target_end = filtered_start_idx + self.sequence_length + self.predict_steps

                if target_end > len(combined_dates):
                    break

                sequence_dates = combined_dates[filtered_start_idx:target_end]
                valid = True

                for i in range(len(sequence_dates) - 1):
                    if (sequence_dates[i + 1] - sequence_dates[i]) != expected_delta:
                        valid = False
                        break

                if not valid:
                    self.invalid_sequence_count += 1
                    continue

                crosses_boundary = (target_end > len(current_dates))
                self.valid_indices.append((archive_id, filtered_start_idx, crosses_boundary))

        self.valid_sequence_count = len(self.valid_indices)
        self.total_candidate_sequences = self.valid_sequence_count + self.invalid_sequence_count

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        archive_id, filtered_start_idx, crosses_boundary = self.valid_indices[idx]

        archive_dir = self.archive_paths[archive_id]
        frames = np.load(archive_dir / "data.npy", mmap_mode="r")
        masks = np.load(archive_dir / "masks.npy", mmap_mode="r")
        dates = self.dates[archive_id]

        offset = self.start_offsets[archive_id]
        real_start_index = offset + filtered_start_idx

        x_end = real_start_index + self.sequence_length
        target_end = x_end + self.predict_steps

        if crosses_boundary:
            next_archive_dir = self.archive_paths[archive_id + 1]
            next_frames = np.load(next_archive_dir / "data.npy", mmap_mode="r")
            next_masks = np.load(next_archive_dir / "masks.npy", mmap_mode="r")
            next_dates = self.dates[archive_id + 1]

            current_file_total_frames = frames.shape[0]
            remaining_frames_needed = target_end - current_file_total_frames

            frames_part1 = frames[real_start_index:].astype(np.float32)
            frames_part2 = next_frames[:remaining_frames_needed].astype(np.float32)
            x_y_frames = np.concatenate((frames_part1, frames_part2), axis=0)

            masks_part1 = masks[real_start_index:].astype(np.float32)
            masks_part2 = next_masks[:remaining_frames_needed].astype(np.float32)
            x_y_masks = np.concatenate((masks_part1, masks_part2), axis=0)

            dates_part1 = dates[filtered_start_idx:]
            dates_part2 = next_dates[:remaining_frames_needed]
            combined_dates = np.concatenate((dates_part1, dates_part2), axis=0)
        else:
            x_y_frames = frames[real_start_index:target_end].astype(np.float32).copy()
            x_y_masks = masks[real_start_index:target_end].astype(np.float32).copy()
            combined_dates = dates[filtered_start_idx:filtered_start_idx + self.sequence_length + self.predict_steps]

        x_frames = x_y_frames[:self.sequence_length]
        x_masks = x_y_masks[:self.sequence_length]

        y_frames = np.squeeze(x_y_frames[self.sequence_length:], axis=1)
        y_masks = np.squeeze(x_y_masks[self.sequence_length:], axis=1)

        x_frames_tensor = torch.from_numpy(x_frames)
        x_masks_tensor = torch.from_numpy(x_masks)
        y_frames_tensor = torch.from_numpy(y_frames)
        y_masks_tensor = torch.from_numpy(y_masks)

        dt_object = combined_dates[self.sequence_length - 1]
        if isinstance(dt_object, np.datetime64):
            naive_dt = dt_object.astype("datetime64[s]")
        else:
            naive_dt = np.datetime64(str(dt_object), 's')

        target_unix_timestamp = int(naive_dt.astype(np.int64))

        return x_frames_tensor, x_masks_tensor, y_frames_tensor, y_masks_tensor, target_unix_timestamp


class StratificationRadarDataset(Dataset):

    def __init__(self, processed_dir: Path, stride=3, clip_value=1500.0):
        self.stride = stride
        self.clip_value = clip_value
        self.archive_dirs = sorted([p.parent for p in processed_dir.glob("*_processed_dir/data.npy")])

    def __len__(self):
        return len(self.archive_dirs)

    def __getitem__(self, idx):
        dir_path = self.archive_dirs[idx]
        try:
            raw_data = np.load(dir_path / "data.npy", mmap_mode='r')
            dates = np.load(dir_path / "dates.npy", allow_pickle=True)
            masks = np.load(dir_path / "masks.npy", mmap_mode='r')
            
            total_frames = len(dates)
            if total_frames == 0:
                return {"valid": False, "dir_name": dir_path.name}

            # Reverse the Log1p Preprocessing Layer back to raw counts
            global_log_max = np.log1p(self.clip_value)
            raw_counts = np.expm1(raw_data.astype(np.float32) * global_log_max)
            
            # Perform unit conversion from 1/100 mm per 5 min to true mm/h
            physical_mm_h = raw_counts * 0.12
            clean_data = np.where(masks, physical_mm_h, 0.0)

            # Apply temporal striding for subsampling
            strided_data = clean_data[::self.stride]
            strided_masks = masks[::self.stride]

            missing_data_ratio = 0.0
            miss_file = dir_path / "miss_dates.npy"
            if miss_file.exists():
                try:
                    miss_dates = np.load(miss_file, allow_pickle=True)
                    missing_data_ratio = float(len(miss_dates) / (total_frames + len(miss_dates)))
                except Exception:
                    pass

            return {
                "valid": True,
                "dir_name": dir_path.name,
                "archive_name": dir_path.name.replace("_processed_dir", ".npz"),
                "total_frames": total_frames,
                "missing_data_ratio": missing_data_ratio,
                "data_tensor": torch.from_numpy(strided_data),
                "mask_tensor": torch.from_numpy(strided_masks.astype(np.bool_))
            }
        except Exception as e:
            return {"valid": False, "dir_name": dir_path.name}
