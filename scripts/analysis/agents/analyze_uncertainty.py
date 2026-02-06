import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "agents" / "simulation_results.csv"
DEFAULT_GRAPHICS_DIR = BASE_DIR / "graphics" / "agents"


def analyze_uncertainty(data_file: Path, output_dir: Path) -> None:
    if not data_file.exists():
        print(f"Data file {data_file} not found. Please run scripts/agent/simulation.py first.")
        raise SystemExit(1)

    results = pd.read_csv(data_file)

    def get_chosen_sigma(row: pd.Series) -> float:
        idx = int(row["choice_arm_index"])
        return row[f"agent_sigma_arm_{idx}"]

    results["chosen_sigma"] = results.apply(get_chosen_sigma, axis=1)
    sigma_data = results.groupby(["agent_name", "reg", "k", "trial"])["chosen_sigma"].mean().reset_index()

    feature_names = ["mercury", "krypton", "nobelium"]
    for feat in feature_names:
        results[f"chosen_unc_{feat}"] = results.apply(
            lambda row: row[f"feature_uncertainty_{feat}_arm_{int(row['choice_arm_index'])}"], axis=1
        )

    feat_unc_columns = [f"chosen_unc_{name}" for name in feature_names]
    feat_data = results.groupby(["trial"])[feat_unc_columns].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.lineplot(
        data=sigma_data,
        x="trial",
        y="chosen_sigma",
        hue="reg",
        style="k",
        ax=axes[0],
    )
    axes[0].set_title("Evolution of Uncertainty (Chosen Arm)", fontsize=14)
    axes[0].set_xlabel("Trial Number")
    axes[0].set_ylabel("Mean Sigma (Prediction Uncertainty)")
    axes[0].grid(True, alpha=0.3)

    feat_data_melted = feat_data.melt(id_vars="trial", var_name="Feature", value_name="Uncertainty")
    feat_data_melted["Feature"] = feat_data_melted["Feature"].str.replace("chosen_unc_", "").str.capitalize()

    sns.lineplot(
        data=feat_data_melted,
        x="trial",
        y="Uncertainty",
        hue="Feature",
        ax=axes[1],
    )
    axes[1].set_title("Diagonal Uncertainty Contributions (Chosen Arm)", fontsize=14)
    axes[1].set_xlabel("Trial Number")
    axes[1].set_ylabel("Variance Contribution (x_i^2 * A_inv[i,i])")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "uncertainty_analysis.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Analysis complete. Plot saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze agent uncertainty evolution.")
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
    analyze_uncertainty(args.input, args.output_dir)
