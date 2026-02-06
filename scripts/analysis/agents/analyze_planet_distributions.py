import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "agents" / "simulation_results.csv"
DEFAULT_GRAPHICS_DIR = BASE_DIR / "graphics" / "agents"


def analyze_planet_distributions(data_file: Path, output_dir: Path) -> None:
    if not data_file.exists():
        print(f"Data file {data_file} not found. Please run scripts/agent/simulation.py first.")
        raise SystemExit(1)

    results = pd.read_csv(data_file)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    planet_counts = results["choice_planet_label"].value_counts().sort_index().reset_index()
    planet_counts.columns = ["Planet", "Count"]
    sns.barplot(data=planet_counts, x="Planet", y="Count", hue="Planet", palette="viridis", ax=ax0, legend=False)
    ax0.set_title("Overall Distribution of Selected Planets", fontsize=14)
    ax0.set_ylabel("Count")
    ax0.set_xlabel("Planet")

    ax1 = fig.add_subplot(gs[0, 1])
    results["trial_bin"] = (results["trial"] - 1) // 10 * 10
    bin_dist = results.groupby(["trial_bin", "choice_planet_label"]).size().unstack(fill_value=0)
    bin_dist_pct = bin_dist.div(bin_dist.sum(axis=1), axis=0)
    bin_dist_pct.plot(kind="area", stacked=True, ax=ax1, colormap="viridis", alpha=0.7)
    ax1.set_title("Planet Selection Frequency over Trials (Binned)", fontsize=14)
    ax1.set_ylabel("Proportion")
    ax1.set_xlabel("Trial Bin")
    ax1.legend(title="Planet", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    k_dist = results.groupby(["k", "choice_planet_label"]).size().unstack(fill_value=0)
    k_dist_pct = k_dist.div(k_dist.sum(axis=1), axis=0)
    k_dist_pct.plot(kind="bar", stacked=True, ax=ax2, colormap="viridis", alpha=0.8)
    ax2.set_title("Planet Selection by Exploration Multiplier (k)", fontsize=14)
    ax2.set_ylabel("Proportion")
    ax2.set_xlabel("k value")
    ax2.legend(title="Planet", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax3 = fig.add_subplot(gs[1, 1])
    reg_dist = results.groupby(["reg", "choice_planet_label"]).size().unstack(fill_value=0)
    reg_dist_pct = reg_dist.div(reg_dist.sum(axis=1), axis=0)
    reg_dist_pct.plot(kind="bar", stacked=True, ax=ax3, colormap="viridis", alpha=0.8)
    ax3.set_title("Planet Selection by Regularization", fontsize=14)
    ax3.set_ylabel("Proportion")
    ax3.set_xlabel("Regularization")
    ax3.legend(title="Planet", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle("Planet Selection Distributions", fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "planet_distribution_analysis.pdf"
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Plots saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selected-planet distributions for agent simulations.")
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
    analyze_planet_distributions(args.input, args.output_dir)
