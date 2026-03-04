import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "llm"
DEFAULT_PATTERN = "llm_simulation_results*.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "graphics" / "llm" / "model_summary.csv"


def infer_model_name(file_path: Path) -> str:
    stem = file_path.stem
    for prefix in ("llm_simulation_results_", "llm_simulation_results-"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def summarize_model_file(file_path: Path) -> dict:
    results = pd.read_csv(file_path)
    required_columns = {"simulation_id", "reward_received", "exploration"}
    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")

    per_sim = results.groupby("simulation_id", as_index=False).agg(
        mean_reward_per_trial=("reward_received", "mean"),
        exploration_rate=("exploration", "mean"),
    )

    return {
        "model": infer_model_name(file_path),
        "file": file_path.name,
        "n_simulations": int(per_sim.shape[0]),
        "mean_reward_per_trial": per_sim["mean_reward_per_trial"].mean(),
        "sd_reward_per_trial": per_sim["mean_reward_per_trial"].std(ddof=1),
        "mean_exploration_rate": per_sim["exploration_rate"].mean(),
        "sd_exploration_rate": per_sim["exploration_rate"].std(ddof=1),
    }


def format_value(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.3f}"


def summarize_models(input_dir: Path, pattern: str, output_file: Path | None) -> None:
    if not input_dir.exists():
        print(f"Input directory {input_dir} not found.")
        raise SystemExit(1)

    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"No files found in {input_dir} matching pattern '{pattern}'.")
        raise SystemExit(1)

    rows = [summarize_model_file(file_path) for file_path in files]
    summary = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

    display_summary = summary.copy()
    for column in (
        "mean_reward_per_trial",
        "sd_reward_per_trial",
        "mean_exploration_rate",
        "sd_exploration_rate",
    ):
        display_summary[column] = display_summary[column].map(format_value)

    print("LLM model summary (simulation-level metrics)")
    print(display_summary.to_string(index=False))

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_file, index=False)
        print(f"\nSummary saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate LLM simulation CSV files by model and report mean/sd performance "
            "and exploration rates."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with LLM simulation CSV files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob pattern used to select model CSV files",
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
    summarize_models(args.input_dir, args.pattern, output_path)
