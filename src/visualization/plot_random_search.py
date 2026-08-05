import argparse
import json
from pathlib import Path
import sys

import matplotlib
import yaml

matplotlib.use("Agg")  # Secure headless background rendering for unattended loops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import get_latest_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Visual Summaries for Hyperparameter Sweeps"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to the directory containing random_search_results.csv. "
        "If omitted, the latest folder in output/tuning/ will be used.",
    )
    return parser.parse_args()


def format_label(name):
    """
    Cleans parameter strings for plot aesthetics on-the-fly.
    Example: 'learning_rate' -> 'Learning Rate'
    """
    return " ".join([word.capitalize() for word in name.split("_")])


def plot_single_parameter_vs_loss(
    df, param_name, output_dir
):
    plt.figure()

    is_log = (
        "lr" in param_name.lower()
        or "decay" in param_name.lower()
        or "learning_rate" in param_name.lower()
    )
    clean_label = format_label(param_name)

    if is_log:
        nonzero_mask = df[param_name] > 0
        zero_mask = df[param_name] == 0

        nonzero_df = df[nonzero_mask]
        zero_df = df[zero_mask]

        if not nonzero_df.empty:
            plt.scatter(
                nonzero_df[param_name],
                nonzero_df["val_loss"],
                label=f"{clean_label} > 0",
            )
            plt.xscale("log")
        if not zero_df.empty:
            min_x = (
                nonzero_df[param_name].min() * 0.1 if not nonzero_df.empty else 1e-5
            )
            plt.scatter(
                [min_x] * len(zero_df),
                zero_df["val_loss"],
                color="tab:blue",
                marker="x",
                label=f"{clean_label} = 0.0",
            )

        if not zero_df.empty and not nonzero_df.empty:
            plt.legend(loc="best")

        xlabel_suffix = " (Log Scale)"
    else:
        plt.scatter(df[param_name], df["val_loss"])
        xlabel_suffix = ""

    plt.xlabel(f"{clean_label}{xlabel_suffix}")
    plt.ylabel("Validation Loss")
    plt.title(f"Convergence Mapping: {clean_label} Performance")
    plt.tight_layout()

    save_path = output_dir / f"{param_name}_vs_loss.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_analytics_plots(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    df_trials = df.copy()
    df_ranked = df.sort_values("val_loss").reset_index(drop=True)
    num_trials = len(df_ranked)

    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True

    # 1) Validation Loss over Trials
    plt.figure()
    plt.plot(np.arange(num_trials), df_trials["val_loss"], marker="o")
    plt.xlabel("Trial Run")
    plt.ylabel("Best Validation Loss")
    plt.title("Random Search History")
    plt.tight_layout()
    plt.savefig(output_dir / "trial_vs_loss.png", dpi=300)
    plt.close()

    # 2) Best Validation Loss so Far
    best_so_far = df_trials["val_loss"].cummin()
    plt.figure()
    plt.plot(np.arange(len(best_so_far)), best_so_far, linewidth=2)
    plt.xlabel("Trial Run")
    plt.ylabel("Minimum Validation Loss Achieved")
    plt.title("Best Validation Loss Found During Sweep Sequence")
    plt.tight_layout()
    plt.savefig(output_dir / "best_so_far.png", dpi=300)
    plt.close()

    # 3) Top 10 Configurations
    top10 = df_ranked.head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(top10)), top10["val_loss"])
    plt.xticks(np.arange(len(top10)), top10["train_name"], rotation=90)
    plt.ylabel("Validation Loss")
    plt.title("Top 10 Hyperparameter Run Profiles")
    plt.tight_layout()
    plt.savefig(output_dir / "top10.png", dpi=300)
    plt.close()

    # 4) Scatter Plots for Active Parameters
    non_param_cols = {
        "run",
        "train_name",
        "best_epoch",
        "train_loss",
        "train_mae",
        "train_rmse",
        "val_loss",
        "val_mae",
        "val_rmse",
    }
    hyperparameters = [col for col in df.columns if col not in non_param_cols]

    for param in hyperparameters:
        plot_single_parameter_vs_loss(
            df=df_ranked, param_name=param, output_dir=output_dir
        )

    # 5) Distribution Histogram
    plt.figure()
    dynamic_bins = int(max(5, min(30, np.ceil(2 * (num_trials ** (1 / 3))))))
    plt.hist(df_ranked["val_loss"], bins=dynamic_bins, edgecolor="darkblue")
    plt.xlabel("Validation Loss")
    plt.ylabel("Model Frequency Count")
    plt.title(
        f"Validation Loss Distribution ({num_trials} Total Trials, {dynamic_bins} Bins)"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "loss_histogram.png", dpi=300)
    plt.close()

    # 6) Correlation Matrix Heatmap
    tracking_targets = hyperparameters + [
        "val_loss",
        "val_mae",
        "val_rmse",
        "best_epoch",
    ]
    valid_targets = [col for col in tracking_targets if col in df_ranked.columns]

    numeric = df_ranked[valid_targets]
    corr = numeric.corr()

    plt.figure(figsize=(8, 7))
    plt.imshow(corr, interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Parameter Importance Multi-Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()

    # Resolve active sweep directory cleanly
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Configuration fallback route to match train patterns
        root_tuning_dir = "output/tuning"
        results_dir = get_latest_directory(
            root_dir_path=root_tuning_dir,
            error_message=f"No sweep folders located inside: {root_tuning_dir}",
        )

    csv_file = results_dir / "random_search_results.csv"
    if not csv_file.exists():
        print(f"CRITICAL: {csv_file.name} missing at {results_dir}. Cannot plot.")
        sys.exit(1)

    df = pd.read_csv(csv_file)
    output_dir = results_dir / "plots"

    print(f"Reading tracking structures out of: {csv_file}")
    print(f"Generating parameter matrices inside output folder: {output_dir}")

    generate_analytics_plots(df=df, output_dir=output_dir)
    print("Sweep visualizations finalized successfully. Canvas targets closed.")


if __name__ == "__main__":
    main()
