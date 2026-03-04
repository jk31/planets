import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "agents" / "simulation_results.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "graphics" / "agents" / "simulations"


def visualize_all_simulations(file_path: Path, output_dir: Path) -> None:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    simulations = df[["agent_name", "simulation_id", "reg", "k"]].drop_duplicates()
    total_sims = len(simulations)
    print(f"Found {total_sims} simulations. Generating plots...")

    for _, row in simulations.iterrows():
        agent_name = row["agent_name"]
        sim_id = row["simulation_id"]
        reg = row["reg"]
        k = row["k"]

        sim_data = df[(df["agent_name"] == agent_name) & (df["simulation_id"] == sim_id)].copy()
        sim_data = sim_data.set_index("trial").sort_index().reset_index()

        mapping = {}
        for arm_idx in range(4):
            col = f"mapping_arm_{arm_idx}"
            mapping[arm_idx] = sim_data[col].values[0] if col in sim_data.columns else "?"

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.step(
            sim_data["trial"],
            sim_data["choice_arm_index"],
            where="post",
            color="gray",
            alpha=0.5,
            label="Selected Arm",
        )

        optimal = sim_data[sim_data["is_optimal"] == 1]
        suboptimal = sim_data[sim_data["is_optimal"] == 0]

        ax.scatter(
            optimal["trial"],
            optimal["choice_arm_index"],
            c="green",
            marker="o",
            label="Optimal Choice",
            s=20,
            zorder=3,
        )
        ax.scatter(
            suboptimal["trial"],
            suboptimal["choice_arm_index"],
            c="red",
            marker="x",
            label="Suboptimal Choice",
            s=20,
            zorder=3,
        )

        ax.set_yticks(range(4))
        ax.set_yticklabels([f"Arm {i} ({mapping[i]})" for i in range(4)])
        ax.set_xlabel("Trial")
        ax.set_ylabel("Selected Arm (Planet)")
        ax.set_title(f"Arm Selection over Time\nAgent: {agent_name} | Sim ID: {sim_id}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        mean_reward = sim_data["reward_received"].mean()
        textstr = "\n".join(
            (
                f"Regularization: {reg}",
                f"Exploration Mul (k): {k}",
                f"Total Trials: {len(sim_data)}",
                f"Mean Reward: {mean_reward:.2f}",
            )
        )

        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
        ax.text(1.02, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)
        ax.legend(loc="upper right")

        plt.tight_layout(rect=(0, 0, 0.85, 1))
        output_file = output_dir / f"reg_{reg}_k_{k}_sim_{sim_id}.pdf"
        plt.savefig(output_file)
        plt.close(fig)

    print(f"All {total_sims} plots saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-simulation arm selection timelines.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_FILE, help="Path to agent simulation CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-simulation PDFs are written",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_all_simulations(args.input, output_dir=args.output_dir)
