import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "agents" / "simulation_results.csv"
DEFAULT_GRAPHICS_DIR = BASE_DIR / "graphics" / "agents"


def analyze_exploration(data_file: Path, output_dir: Path) -> None:
    if not data_file.exists():
        print(f"Data file {data_file} not found. Please run scripts/agent/simulation.py first.")
        raise SystemExit(1)

    results = pd.read_csv(data_file)
    sns.set_theme(style="whitegrid")
    reg_values = sorted(results["reg"].unique())
    k_values = sorted(results["k"].unique())

    fig, axes = plt.subplots(1, len(reg_values), figsize=(5 * len(reg_values), 5), sharey=True)
    axes = [axes] if len(reg_values) == 1 else axes
    palette_k = sns.color_palette("rocket_r", n_colors=len(k_values))

    for i, reg in enumerate(reg_values):
        ax = axes[i]
        subset = results[results["reg"] == reg]
        sns.lineplot(
            data=subset,
            x="trial",
            y="exploration",
            hue="k",
            palette=palette_k,
            errorbar=("ci", 95),
            ax=ax,
            legend=(i == len(reg_values) - 1),
        )
        ax.set_title(f"Reg={reg}", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("P(Explore)" if i == 0 else "")
        ax.grid(True, alpha=0.3)
        if i == len(reg_values) - 1:
            ax.legend(title="k value", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle("Consolidated Exploration Behavior Analysis (95% CI)", fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "exploration_analysis.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Analysis complete. Plot saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze agent exploration behavior.")
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
    analyze_exploration(args.input, args.output_dir)
