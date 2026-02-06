import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "agents" / "simulation_results.csv"
DEFAULT_GRAPHICS_DIR = BASE_DIR / "graphics" / "agents"


def visualize_weight_evolution(csv_path: Path, output_file: Path) -> None:
    df = pd.read_csv(csv_path)

    reg_values = sorted(df["reg"].unique())
    k_values = sorted(df["k"].unique())

    ground_truth = {
        "A": {"Intercept": 50, "Mercury": 15, "Krypton": -15, "Nobelium": 0},
        "B": {"Intercept": 50, "Mercury": 0, "Krypton": 15, "Nobelium": -15},
        "C": {"Intercept": 50, "Mercury": -15, "Krypton": 0, "Nobelium": 15},
        "D": {"Intercept": 50, "Mercury": 0, "Krypton": 0, "Nobelium": 0},
    }

    features = ["Intercept", "Mercury", "Krypton", "Nobelium"]
    planets = ["A", "B", "C", "D"]
    colors = {"Intercept": "black", "Mercury": "red", "Krypton": "green", "Nobelium": "blue"}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_file) as pdf:
        for reg in reg_values:
            for k in k_values:
                agent_df = df[(df["reg"] == reg) & (df["k"] == k)].copy()
                if agent_df.empty:
                    continue

                print(f"Processing weights for reg={reg}, k={k}...")

                planet_data = []
                for _, row in agent_df.iterrows():
                    trial = row["trial"]
                    sim_id = row["simulation_id"]
                    for arm_idx in range(4):
                        planet_label = row[f"mapping_arm_{arm_idx}"]
                        record = {"trial": trial, "sim_id": sim_id, "planet": planet_label}
                        for feature in features:
                            col_name = f"w_{feature.lower()}_arm_{arm_idx}"
                            record[feature] = row[col_name]
                        planet_data.append(record)

                remapped_df = pd.DataFrame(planet_data)
                avg_weights = remapped_df.groupby(["planet", "trial"])[features].mean().reset_index()

                fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
                axes = axes.flatten()

                for idx, planet in enumerate(planets):
                    ax = axes[idx]
                    planet_df = avg_weights[avg_weights["planet"] == planet]
                    for feature in features:
                        ax.plot(planet_df["trial"], planet_df[feature], label=feature, color=colors[feature], linewidth=2)
                        ax.axhline(y=ground_truth[planet][feature], color=colors[feature], linestyle="--", alpha=0.5)
                    ax.set_title(f"Planet {planet}")
                    ax.grid(True, alpha=0.3)
                    if idx >= 2:
                        ax.set_xlabel("Trial")
                    if idx % 2 == 0:
                        ax.set_ylabel("Weight Value")

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.95))
                plt.suptitle(f"Weight Evolution: reg={reg}, k={k}\n(Dashed lines = Ground Truth)", fontsize=16)
                plt.tight_layout(rect=(0, 0, 0.92, 0.95))

                pdf.savefig(fig)
                plt.close(fig)

    print(f"All weight visualizations saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze convergence of agent linear weights.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_FILE, help="Path to agent simulation CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GRAPHICS_DIR / "weight_evolution_analysis.pdf",
        help="Path to output PDF",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        print(f"Data file {args.input} not found. Please run scripts/agent/simulation.py first.")
        raise SystemExit(1)
    visualize_weight_evolution(args.input, output_file=args.output)
