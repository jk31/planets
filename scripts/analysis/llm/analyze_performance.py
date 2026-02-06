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

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=results,
        x="trial",
        y="reward_received",
        color="blue",
        linewidth=2,
        label="LLM Agent",
    )

    plt.axhline(y=50, color="red", linestyle="--", alpha=0.6, label="Baseline (Safe Planet D)")
    plt.title("LLM Agent Learning Curve: Reward per Trial (with 95% CI)", fontsize=16)
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("Average Reward Received", fontsize=12)
    plt.ylim(bottom=0)
    plt.legend(loc="lower right")

    max_trial = int(results["trial"].to_numpy().max())
    avg_final = float(results.loc[results["trial"] == max_trial, "reward_received"].mean())
    plt.annotate(
        f"Final Avg: {avg_final:.1f}",
        xy=(float(max_trial), float(avg_final)),
        xytext=(float(max_trial - 10), float(avg_final + 10)),
        arrowprops={"facecolor": "black", "shrink": 0.05, "width": 1, "headwidth": 5},
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "performance_analysis.pdf"
    plt.savefig(output_file)
    plt.close()
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
