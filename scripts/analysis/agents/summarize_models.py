import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_FILE = BASE_DIR / "data" / "agents" / "simulation_results.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "graphics" / "agents" / "model_summary.csv"


def summarize_agent_file(file_path: Path) -> pd.DataFrame:
    results = pd.read_csv(file_path)
    required_columns = {"simulation_id", "reg", "k", "reward_received", "exploration"}
    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")

    per_sim = results.groupby(["reg", "k", "simulation_id"], as_index=False).agg(
        mean_reward_per_trial=("reward_received", "mean"),
        exploration_rate=("exploration", "mean"),
    )

    summary = (
        per_sim.groupby(["reg", "k"], as_index=False)
        .agg(
            n_simulations=("simulation_id", "nunique"),
            mean_reward_per_trial=("mean_reward_per_trial", "mean"),
            sd_reward_per_trial=("mean_reward_per_trial", "std"),
            mean_exploration_rate=("exploration_rate", "mean"),
            sd_exploration_rate=("exploration_rate", "std"),
        )
        .sort_values(["reg", "k"])
        .reset_index(drop=True)
    )

    return summary


def format_value(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.3f}"


def summarize_models(input_file: Path, output_file: Path | None) -> None:
    if not input_file.exists():
        print(f"Input file {input_file} not found.")
        raise SystemExit(1)

    summary = summarize_agent_file(input_file)

    display_summary = summary.copy()
    for column in (
        "mean_reward_per_trial",
        "sd_reward_per_trial",
        "mean_exploration_rate",
        "sd_exploration_rate",
    ):
        display_summary[column] = display_summary[column].map(format_value)

    print("Linear agent summary by (reg, k) (simulation-level metrics)")
    print(display_summary.to_string(index=False))

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_file, index=False)
        print(f"\nSummary saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate linear-agent simulation results by (reg, k) and report mean/sd "
            "performance and exploration rates."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to agent simulation CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Optional path to save the summary as CSV",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Only print the summary table; do not write a CSV file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = None if args.no_output else args.output
    summarize_models(args.input, output_path)
