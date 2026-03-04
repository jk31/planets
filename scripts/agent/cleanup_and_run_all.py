import argparse
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data" / "agents"
GRAPHICS_DIR = BASE_DIR / "graphics" / "agents"


def clean_dir(directory: Path) -> None:
    if directory.exists():
        print(f"Cleaning {directory}...")
        for path in directory.iterdir():
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as exc:
                print(f"Failed to delete {path}. Reason: {exc}")
    else:
        print(f"Directory {directory} does not exist. Creating it...")
        directory.mkdir(parents=True, exist_ok=True)


def get_python_executable() -> Path:
    venv_python = BASE_DIR / "venv" / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else Path(sys.executable)


def run_script(script_path: Path, args: list[str] | None = None) -> bool:
    args = args or []
    script_name = script_path.name
    print(f"\n>>> Running {script_name} {' '.join(args)} <<<")

    cmd = [str(get_python_executable()), str(script_path), *args]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with exit code {result.returncode}")
        return False

    print(f"Success: {script_name} completed.")
    return True


def run_agents_pipeline(agent_sims: int, agent_trials: int) -> None:
    clean_dir(DATA_DIR)
    clean_dir(GRAPHICS_DIR)

    simulation_ok = run_script(
        SCRIPTS_DIR / "agent" / "simulation.py",
        [
            "--n_simulations",
            str(agent_sims),
            "--n_trials",
            str(agent_trials),
            "--output",
            str(DATA_DIR / "simulation_results.csv"),
        ],
    )
    if not simulation_ok:
        return

    agent_analysis_scripts = [
        SCRIPTS_DIR / "analysis" / "agents" / "summarize_models.py",
        SCRIPTS_DIR / "analysis" / "agents" / "analyze_performance.py",
        SCRIPTS_DIR / "analysis" / "agents" / "analyze_exploration.py",
        SCRIPTS_DIR / "analysis" / "agents" / "analyze_uncertainty.py",
        SCRIPTS_DIR / "analysis" / "agents" / "analyze_weights.py",
        SCRIPTS_DIR / "analysis" / "agents" / "analyze_planet_distributions.py",
        SCRIPTS_DIR / "analysis" / "agents" / "visualize_simulation_arms.py",
    ]

    for script_path in agent_analysis_scripts:
        if script_path.exists():
            run_script(script_path)
        else:
            print(f"Skipping: {script_path} not found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and run the normal-agent simulation pipeline.")
    parser.add_argument("--simulations", type=int, default=20, help="Agent simulations per reg/k configuration")
    parser.add_argument("--trials", type=int, default=50, help="Agent trials per simulation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agents_pipeline(agent_sims=args.simulations, agent_trials=args.trials)
    print("\n[DONE] Agent cleanup and execution tasks finished.")
