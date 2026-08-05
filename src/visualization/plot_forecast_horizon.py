import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Secure headless mode for automated pipeline loops
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Performance Degradation Curves Across Forecast Horizons")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to the test run folder containing test.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)

    with open(test_dir / "test.json", "r") as f:
        test = json.load(f)

    horizon = test["results"]["forecast_horizon"]

    minutes = horizon["minutes"]
    loss = horizon["loss"]
    mae = horizon["mae"]
    rmse = horizon["rmse"]

    print()
    print("Forecast Horizon Metrics")
    print("-" * 50)

    for m, l, a, r in zip(minutes, loss, mae, rmse):
        print(
            f"+{m:2d} min | "
            f"Loss={l:.6f} | "
            f"MAE={a:.6f} | "
            f"RMSE={r:.6f}"
        )

    plots_dir = test_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Unify visual aesthetics
    plt.rcParams["axes.grid"] = True

    # 1) Plot Loss Curve Separately
    plt.figure(figsize=(7, 4.5))
    plt.plot(minutes, loss, marker="o", linewidth=2, color="tab:blue", label="Loss (MSE)")
    plt.xlabel("Forecast Horizon (minutes)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Forecast Horizon Performance: Loss Degradation")
    plt.xticks(minutes)
    plt.tight_layout()
    
    loss_path = plots_dir / "horizon_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved Loss figure to {loss_path}")

    # 2) Plot MAE Curve Separately
    plt.figure(figsize=(7, 4.5))
    plt.plot(minutes, mae, marker="s", linewidth=2, color="tab:orange", label="MAE")
    plt.xlabel("Forecast Horizon (minutes)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Forecast Horizon Performance: MAE Degradation")
    plt.xticks(minutes)
    plt.tight_layout()
    
    mae_path = plots_dir / "horizon_mae.png"
    plt.savefig(mae_path, dpi=300)
    plt.close()
    print(f"Saved MAE figure to {mae_path}")

    # 3) Plot RMSE Curve Separately
    plt.figure(figsize=(7, 4.5))
    plt.plot(minutes, rmse, marker="^", linewidth=2, color="tab:green", label="RMSE")
    plt.xlabel("Forecast Horizon (minutes)")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.title("Forecast Horizon Performance: RMSE Degradation")
    plt.xticks(minutes)
    plt.tight_layout()
    
    rmse_path = plots_dir / "horizon_rmse.png"
    plt.savefig(rmse_path, dpi=300)
    plt.close()
    print(f"Saved RMSE figure to {rmse_path}")
    print()


if __name__ == "__main__":
    main()
