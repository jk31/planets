import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "llm" / "simulation_results.csv"
DEFAULT_GRAPHICS_DIR = BASE_DIR / "graphics" / "llm"


def analyze_exploration(data_file: Path, output_dir: Path) -> None:
    if not data_file.exists():
        print(f"Data file {data_file} not found. Please run scripts/llm/run_simulation.py first.")
        raise SystemExit(1)

    results = pd.read_csv(data_file)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=results,
        x="trial",
        y="exploration",
        color="orange",
        linewidth=2,
        label="LLM Agent",
    )

    plt.title("LLM Agent Exploration Behavior: Probability of Exploration over Time", fontsize=16)
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("P(Explore)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    initial_expl = float(results.loc[results["trial"] <= 5, "exploration"].to_numpy().mean())
    final_expl = float(results.loc[results["trial"] >= 45, "exploration"].to_numpy().mean())

    plt.text(5, 0.9, f"Initial Explore Rate: {initial_expl:.1%}", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8})
    plt.text(35, 0.1, f"Final Explore Rate: {final_expl:.1%}", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8})

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "exploration_analysis.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"LLM exploration plot saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LLM exploration behavior.")
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
    analyze_exploration(args.input, args.output_dir)
