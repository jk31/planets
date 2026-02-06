import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "agents" / "simulation_results.csv"
DEFAULT_GRAPHICS_DIR = BASE_DIR / "graphics" / "agents"


def analyze_performance(data_file: Path, output_dir: Path) -> None:
    if not data_file.exists():
        print(f"Data file {data_file} not found. Please run scripts/agent/simulation.py first.")
        raise SystemExit(1)

    results = pd.read_csv(data_file)
    grid_summary = results.groupby(["reg", "k"])["reward_received"].mean().reset_index()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    pivot_data = grid_summary.pivot(index="reg", columns="k", values="reward_received")
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax0)
    ax0.set_title("Overall Average Reward: Regularization vs Exploration Multiplier (k)", fontsize=14)
    ax0.set_ylabel("Regularization")
    ax0.set_xlabel("Exploration Multiplier (k)")

    learning_data = results.groupby(["reg", "k", "trial"])["reward_received"].mean().reset_index()
    reg_values = sorted(learning_data["reg"].unique())
    k_values = sorted(learning_data["k"].unique())
    palette_k = sns.color_palette("rocket_r", n_colors=len(k_values))

    gs_inner = gs[1].subgridspec(1, len(reg_values))

    for i, reg in enumerate(reg_values):
        ax = fig.add_subplot(gs_inner[i])
        subset = learning_data[learning_data["reg"] == reg]
        sns.lineplot(
            data=subset,
            x="trial",
            y="reward_received",
            hue="k",
            palette=palette_k,
            ax=ax,
            legend=(i == len(reg_values) - 1),
        )
        ax.set_title(f"Reg={reg}", fontsize=12)
        ax.set_ylabel("Avg Reward" if i == 0 else "")
        ax.set_xlabel("Trial Number")
        ax.grid(True, alpha=0.3)
        if i == len(reg_values) - 1:
            ax.legend(title="k value", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle("Consolidated Performance Analysis: Heatmap and Learning Curves", fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "performance_analysis.pdf"
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Plots saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze agent reward performance.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_FILE, help="Path to agent simulation CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GRAPHICS_DIR,
        help="Directory where plots are saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_performance(args.input, args.output_dir)
