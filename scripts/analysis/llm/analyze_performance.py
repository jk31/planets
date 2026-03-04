import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "llm" / "simulation_results.csv"
DEFAULT_GRAPHICS_DIR = BASE_DIR / "graphics" / "llm"


def analyze_performance(data_file: Path, output_dir: Path) -> None:
    if not data_file.exists():
        print(f"Data file {data_file} not found. Please run scripts/llm/run_simulation.py first.")
        raise SystemExit(1)

    results = pd.read_csv(data_file)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    line_color = sns.color_palette("rocket_r", n_colors=1)[0]

    sns.lineplot(
        data=results,
        x="trial",
        y="reward_received",
        color=line_color,
        linewidth=2,
        errorbar=("ci", 95),
        ax=ax,
    )

    ax.set_title("LLM Learning Curve", fontsize=14)
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Avg Reward")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Consolidated Performance Analysis: Learning Curve (95% CI)", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "performance_analysis.pdf"
    fig.savefig(output_file)
    plt.close(fig)
    print(f"LLM performance plot saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LLM reward performance.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_FILE, help="Path to LLM simulation CSV")
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
