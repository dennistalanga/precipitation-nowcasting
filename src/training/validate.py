import sys

import torch

from src.evaluation.metrics import masked_mae, masked_rmse
from src.utils.logger import get_log_prefix


def evaluate_train(
    denom,
    total_loss,
    total_mae,
    total_rmse,
    total_grad_norm,
):

    return {
        "loss": total_loss / denom,
        "mae": total_mae / denom,
        "rmse": total_rmse / denom,
        "gradient_norm": total_grad_norm
    }


def evaluate_val(model, loader, device, criterion, logger, epoch):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_valid_pixels = 0

    # Determine if AMP should be active based on the device type
    use_amp = (device.type == 'cuda')

    num_batches = len(loader)

    with torch.no_grad():

        for batch_idx, (x, x_mask, y, y_mask, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            x_mask = x_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)
            x = torch.cat([x, x_mask], dim=2) # dim=2 accounts for the Batch dimension (Batch, Time, Channel, H, W)

            # Enforce mixed precision forward pass
            with torch.autocast(device_type='cuda', enabled=use_amp):
                preds = model(x)
                loss = criterion(preds, y, y_mask)

            # pixel-level weighting
            valid_pixels = y_mask.sum().item()
            total_valid_pixels += valid_pixels
            total_loss += loss.item() * valid_pixels

            detached_preds = preds.detach()
            total_mae += masked_mae(detached_preds, y, y_mask).item() * valid_pixels
            total_rmse += masked_rmse(detached_preds, y, y_mask).item() * valid_pixels

            # Clear line first in case a progress bar was present
            print("\r\033[K", end="", flush=True)
            # Generate live logger-style prefix
            log_prefix = get_log_prefix(logger.name)
            
            # Print dynamic line: \r resets cursor, \033[K clears trailing text
            progress_str = f"\r\033[K{log_prefix}Epoch {epoch+1} Validation: Batch {batch_idx + 1}/{num_batches}"
            sys.stdout.write(progress_str)
            sys.stdout.flush()

    # Remove the line completely when the loop ends
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    denom = max(total_valid_pixels, 1)

    return {
        "loss": total_loss / denom,
        "mae": total_mae / denom,
        "rmse": total_rmse / denom
    }
