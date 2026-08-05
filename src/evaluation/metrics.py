import numpy as np
import torch
import torch.nn.functional as F


def _inverse_log_transform(tensor, clip_value=1500.0):
    global_log_max = np.log1p(clip_value)
    tensor_bounded = torch.clamp(tensor, 0.0, 1.0)
    return torch.expm1(tensor_bounded * global_log_max)


def masked_mae(prediction, target, mask):
    prediction, target, mask = prediction.float(), target.float(), mask.float()
    pred_physical = _inverse_log_transform(prediction)
    target_physical = _inverse_log_transform(target)

    error = (pred_physical - target_physical).abs() * mask
    return error.sum() / mask.sum().clamp(min=1)


def masked_rmse(prediction, target, mask):
    prediction, target, mask = prediction.float(), target.float(), mask.float()
    pred_physical = _inverse_log_transform(prediction)
    target_physical = _inverse_log_transform(target)

    error = (pred_physical - target_physical) ** 2 * mask
    mse = error.sum() / mask.sum().clamp(min=1)
    return torch.sqrt(mse)


def masked_mae_per_step(prediction, target, mask):
    prediction, target, mask = prediction.float(), target.float(), mask.float()
    pred_physical = _inverse_log_transform(prediction)
    target_physical = _inverse_log_transform(target)

    absolute_error = torch.abs(pred_physical - target_physical) * mask
    dims = (0, 2, 3)
    return absolute_error.sum(dim=dims) / mask.sum(dim=dims).clamp(min=1)


def masked_rmse_per_step(prediction, target, mask):
    prediction, target, mask = prediction.float(), target.float(), mask.float()
    pred_physical = _inverse_log_transform(prediction)
    target_physical = _inverse_log_transform(target)

    squared_error = (pred_physical - target_physical) ** 2 * mask
    dims = (0, 2, 3)
    return torch.sqrt(squared_error.sum(dim=dims) / mask.sum(dim=dims).clamp(min=1))


def compute_batch_contingency(prediction, target, mask, threshold):
    """
    Calculates operational Hits, False Alarms, Misses, and Correct Negatives
    across a target intensity threshold. Returns flat tensor sums.
    """
    pred_p = _inverse_log_transform(prediction.float())
    target_p = _inverse_log_transform(target.float())
    mask_b = mask.float() > 0.5

    p_bin = pred_p >= threshold
    t_bin = target_p >= threshold

    hits = torch.sum(p_bin & t_bin & mask_b, dim=(0, 2, 3))
    false_alarms = torch.sum(p_bin & ~t_bin & mask_b, dim=(0, 2, 3))
    misses = torch.sum(~p_bin & t_bin & mask_b, dim=(0, 2, 3))
    correct_negatives = torch.sum(~p_bin & ~t_bin & mask_b, dim=(0, 2, 3))

    return torch.stack([hits, false_alarms, misses, correct_negatives]) # Shape: (4, P)


def compute_batch_fss(prediction, target, mask, scale, threshold):
    """
    Computes Fractions Skill Score (FSS) over a neighborhood bounding box scale
    utilizing parallelized 2D average pooling.
    """
    pred_p = _inverse_log_transform(prediction.float())
    target_p = _inverse_log_transform(target.float())
    mask_b = mask.float() > 0.5

    p_bin = (pred_p >= threshold).float() * mask_b.float()
    t_bin = (target_p >= threshold).float() * mask_b.float()

    # Reshape from (B, P, H, W) to (B*P, 1, H, W) to run deep 2D average pooling in one pass
    b, p, h, w = p_bin.shape
    p_bin_flat = p_bin.view(b * p, 1, h, w)
    t_bin_flat = t_bin.view(b * p, 1, h, w)

    pad = scale // 2
    p_padded = F.pad(p_bin_flat, (pad, pad, pad, pad), mode='constant', value=0.0)
    t_padded = F.pad(t_bin_flat, (pad, pad, pad, pad), mode='constant', value=0.0)

    f_pred = F.avg_pool2d(p_padded, kernel_size=scale, stride=1).view(b, p, h, w)
    f_true = F.avg_pool2d(t_padded, kernel_size=scale, stride=1).view(b, p, h, w)

    num = torch.mean((f_pred - f_true) ** 2, dim=(2, 3)) # Shape: (B, P)
    denom = torch.mean(f_pred ** 2, dim=(2, 3)) + torch.mean(f_true ** 2, dim=(2, 3))

    fss_per_sequence = torch.where(denom == 0, torch.where(num == 0, 1.0, 0.0), 1.0 - (num / denom))
    return fss_per_sequence.mean(dim=0) # Returns shape: (P,)


def compute_batch_sal(prediction, target, mask, threshold=1.0):
    """
    Calculates Structure (S), Amplitude (A), and Location (L) metrics 
    by computing batch-wide coordinate grid spatial moments.
    """
    pred_p = _inverse_log_transform(prediction.float()) * mask.float()
    target_p = _inverse_log_transform(target.float()) * mask.float()

    # 1. Amplitude (A) Calculation
    mean_p = pred_p.sum(dim=(2, 3)) # Shape: (B, P)
    mean_t = target_p.sum(dim=(2, 3))
    A_vals = torch.where((mean_p + mean_t) == 0, 0.0, (mean_p - mean_t) / (0.5 * (mean_p + mean_t)))

    # 2. Location (L) Calculation via Meshgrids
    b, p, h, w = pred_p.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=pred_p.device), torch.arange(w, device=pred_p.device), indexing='ij')

    # Expand grids to match batch layout: (1, 1, H, W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).float()
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).float()

    total_p = pred_p.sum(dim=(2, 3)) + 1e-7  # Squeeze spatial bounds immediately
    total_t = target_p.sum(dim=(2, 3)) + 1e-7

    # Multi-dimensional element-wise parallel division
    c_p_y = (pred_p * grid_y).sum(dim=(2, 3)) / total_p
    c_p_x = (pred_p * grid_x).sum(dim=(2, 3)) / total_p
    c_t_y = (target_p * grid_y).sum(dim=(2, 3)) / total_t
    c_t_x = (target_p * grid_x).sum(dim=(2, 3)) / total_t

    max_dist = np.sqrt(h**2 + w**2)
    L_vals = torch.sqrt((c_p_y - c_t_y)**2 + (c_p_x - c_t_x)**2) / max_dist

    # 3. Structure (S) Calculation
    max_p = torch.max(pred_p.view(b, p, -1), dim=-1)[0] + 1e-7
    max_t = torch.max(target_p.view(b, p, -1), dim=-1)[0] + 1e-7
    v_p = pred_p.sum(dim=(2, 3)) / max_p
    v_t = target_p.sum(dim=(2, 3)) / max_t
    S_vals = torch.where((v_p + v_t) == 0, 0.0, (v_p - v_t) / (0.5 * (v_p + v_t)))

    return torch.stack([S_vals.mean(dim=0), A_vals.mean(dim=0), L_vals.mean(dim=0)]) # Shape: (3, P)


def compute_batch_correlation_sums(prediction, target, mask, light_rain_threshold=0.12):
    """
    Computes the running sums, sum of squares, and sum of products 
    to calculate global Pearson correlation.
    """
    pred_p = _inverse_log_transform(prediction.float())
    target_p = _inverse_log_transform(target.float())
    
    # Dynamically match your active configuration limits
    rain_mask = (target_p > light_rain_threshold) & (mask.float() > 0.5)

    if not torch.any(rain_mask):
        # Return 6 elements to perfectly match the active stack footprint dimensions below
        return torch.zeros(6, device=prediction.device, dtype=torch.float64)

    x = pred_p[rain_mask].double()
    y = target_p[rain_mask].double()

    return torch.stack([
        x.sum(),          # Index 0: sum_x
        y.sum(),          # Index 1: sum_y
        (x ** 2).sum(),   # Index 2: sum_x2
        (y ** 2).sum(),   # Index 3: sum_y2
        (x * y).sum(),    # Index 4: sum_xy
        torch.tensor(x.numel(), device=prediction.device, dtype=torch.float64) # Index 5: n_pixels
    ])
