import argparse
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")  # Forces non-interactive background file rendering
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import get_latest_directory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training history metrics from local logs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_baseline.yml",
        help="Path to training config to resolve root output paths if dir is missing",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Path to the targeted experiment folder containing history.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Explicit output path for the saved image file",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def add_best_epoch(ax, best_epoch):
    ax.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Best epoch",
    )


def add_early_stopping(ax, history, best_epoch):
    if len(history["epoch"]) <= best_epoch:
        return
    ax.axvspan(
        best_epoch,
        history["epoch"][-1],
        color="lightgray",
        alpha=0.3,
        label="After best epoch",
    )


def main():
    args = parse_args()

    # 1. Resolve experiment directory cleanly
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        root_models_dir = config.get("paths", {}).get(
            "model_output_dir", "output/models"
        )
        experiment_dir = get_latest_directory(
            root_dir_path=root_models_dir,
            error_message=f"No run folders found inside: {root_models_dir}",
        )

    # 2. Confidently load guaranteed JSON pairings
    experiment = load_json(experiment_dir / "experiment.json")
    history = load_json(experiment_dir / "history.json")

    epochs = history["epoch"]
    best_epoch = experiment["model"]["best_epoch"]

    fig = plt.figure(figsize=(14, 13))
    gs_master = fig.add_gridspec(4, 2, height_ratios=[1.5, 0.85, 0.85, 0.8])

    ax_loss_macro = fig.add_subplot(gs_master[0, 0])
    ax_loss_micro = fig.add_subplot(gs_master[0, 1])
    ax_metrics = fig.add_subplot(gs_master[1:3, 0])
    ax_lr = fig.add_subplot(gs_master[1, 1])
    ax_grad = fig.add_subplot(gs_master[2, 1])
    ax_runtime = fig.add_subplot(gs_master[3, :])
    gs_master.update(hspace=0.4, wspace=0.25)

    #  Macroscopic loss profile (All Epochs)
    ax_loss_macro.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax_loss_macro.plot(epochs, history["val_loss"], marker="o", label="Validation")
    add_best_epoch(ax_loss_macro, best_epoch)
    
    ax_loss_macro.scatter(
        best_epoch,
        history["val_loss"][best_epoch - 1],
        color="red",
        s=80,
        zorder=5,
        label="Best validation",
    )
    add_early_stopping(ax_loss_macro, history, best_epoch)
    ax_loss_macro.set_title("Macroscopic Loss Profile (Inc. Epoch 1)")
    ax_loss_macro.set_xlabel("Epoch")
    ax_loss_macro.set_ylabel("Masked MSE")
    ax_loss_macro.grid(True)
    ax_loss_macro.legend()

    # Microscopic loss profile (Epoch 2+)
    if len(epochs) > 1:
        micro_epochs = list(epochs)
        train_loss_micro = [None] + list(history["train_loss"][1:])
        val_loss_micro = [None] + list(history["val_loss"][1:])
        ax_loss_micro.plot(micro_epochs, train_loss_micro, marker="o", label="Train")
        ax_loss_micro.plot(micro_epochs, val_loss_micro, marker="o", label="Validation")
        add_best_epoch(ax_loss_micro, best_epoch)
        if best_epoch > 1:
            ax_loss_micro.scatter(
                best_epoch,
                history["val_loss"][best_epoch - 1],
                color="red",
                s=80,
                zorder=5,
                label="Best validation",
            )
        add_early_stopping(ax_loss_micro, history, best_epoch)
        ax_loss_micro.set_xlim(0.8, max(epochs) + 0.5)
        ax_loss_micro.set_title("Microscopic Loss Profile (Epoch 2+ Zoom)")
    else:
        ax_loss_micro.text(
            0.5,
            0.5,
            "Zoom unavailable\n(Requires > 1 Epoch)",
            ha="center",
            va="center",
        )
    ax_loss_micro.set_xlabel("Epoch")
    ax_loss_micro.set_ylabel("Masked MSE")
    ax_loss_micro.grid(True)
    ax_loss_micro.legend()

    # Evaluation metrics
    ax_metrics.plot(epochs, history["train_mae"], marker="o", label="Train MAE")
    ax_metrics.plot(epochs, history["val_mae"], marker="o", label="Validation MAE")
    ax_metrics.plot(
        epochs, history["train_rmse"], marker="s", linestyle="--", label="Train RMSE"
    )
    ax_metrics.plot(
        epochs, history["val_rmse"], marker="s", linestyle="--", label="Validation RMSE"
    )
    add_best_epoch(ax_metrics, best_epoch)
    add_early_stopping(ax_metrics, history, best_epoch)
    ax_metrics.set_title("Validation Metrics Evaluation")
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_ylabel("Rainfall (normalized)")
    ax_metrics.grid(True)
    ax_metrics.legend()

    # Learning rate history
    if "learning_rate" in history:
        ax_lr.plot(epochs, history["learning_rate"], marker="o", linewidth=2)
        add_best_epoch(ax_lr, best_epoch)
        add_early_stopping(ax_lr, history, best_epoch)
        ax_lr.set_yscale("log")
        ax_lr.set_xlim(0.8, max(epochs) + 0.5)
        ax_lr.set_ylabel("Learning Rate")
    else:
        ax_lr.text(
            0.5, 0.5, "Learning rate history unavailable", ha="center", va="center"
        )
        ax_lr.set_xticks([])
        ax_lr.set_yticks([])
    ax_lr.set_title("Learning Rate History")
    ax_lr.set_xlabel("Epoch")
    ax_lr.grid(True)

    # Gradient stability
    if "gradient_norm" in history:
        gradient_norm = np.array(history["gradient_norm"])
        ax_grad.plot(epochs, gradient_norm, marker="o", linewidth=2)
        ax_grad.axhline(
            gradient_norm.mean(),
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {gradient_norm.mean():.4f}",
        )
        add_best_epoch(ax_grad, best_epoch)
        add_early_stopping(ax_grad, history, best_epoch)
        ax_grad.set_xlim(0.8, max(epochs) + 0.5)
        ax_grad.legend()
    else:
        ax_grad.text(
            0.5, 0.5, "Gradient norm unavailable", ha="center", va="center"
        )
        ax_grad.set_xticks([])
        ax_grad.set_yticks([])
    ax_grad.set_title("Gradient Stability Norm")
    ax_grad.set_xlabel("Epoch")
    ax_grad.set_ylabel("L2 Norm")
    ax_grad.grid(True)

    # Epoch runtime
    if "epoch_runtime_seconds" in history:
        runtimes = np.array(history["epoch_runtime_seconds"])
        ax_runtime.bar(epochs, runtimes, width=0.7)
        ax_runtime.axhline(
            runtimes.mean(),
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {runtimes.mean():.2f}s",
        )
        add_best_epoch(ax_runtime, best_epoch)
        add_early_stopping(ax_runtime, history, best_epoch)
        ax_runtime.legend()
    else:
        ax_runtime.text(
            0.5, 0.5, "Runtime history unavailable", ha="center", va="center"
        )
        ax_runtime.set_xticks([])
        ax_runtime.set_yticks([])
    ax_runtime.set_title("Compute Allocation Runtime Per Epoch")
    ax_runtime.set_xlabel("Epoch")
    ax_runtime.set_ylabel("Seconds")
    ax_runtime.grid(True)

    # Dynamic title & metadata formatting
    hp = experiment["hyperparameters"]
    base_title = (
        f'Experiment Tracking Profile: {experiment["train_name"]}\n'
        f'Architecture Stack: {experiment["model"]["architecture"]} | '
    )
    hp_tokens = [
        f"{k.replace('learning_rate', 'LR').replace('weight_decay', 'Decay')}={v}"
        for k, v in hp.items()
    ]

    chunked_lines = [
        " | ".join(hp_tokens[i : i + 4]) for i in range(0, len(hp_tokens), 4)
    ]
    full_title_string = base_title + "\n".join(chunked_lines)
    
    fig.suptitle(full_title_string, fontsize=12, y=0.96, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.935])

    output_path = Path(args.output) if args.output else experiment_dir / "training_history.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"Saved training analysis graphic successfully to {output_path}")


if __name__ == "__main__":
    main()
