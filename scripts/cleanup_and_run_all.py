import os
import shutil
import subprocess
import sys

def clean_dir(directory):
    if os.path.exists(directory):
        print(f"Cleaning {directory}...")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory {directory} does not exist. Creating it...")
        os.makedirs(directory)

def run_script(script_path, args=None):
    if args is None:
        args = []
    script_name = os.path.basename(script_path)
    print(f"\n>>> Running {script_name} {' '.join(args)} <<<")
    # Use the venv python if available
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(BASE_DIR, 'venv', 'Scripts', 'python.exe')
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    cmd = [python_exe, script_path] + args
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with exit code {result.returncode}")
    else:
        print(f"Success: {script_name} completed.")

if __name__ == "__main__":
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
    SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

    # 1. Clean folders
    clean_dir(DATA_DIR)
    clean_dir(GRAPHICS_DIR)

    # 2. Run simulation.py
    simulation_script = os.path.join(SCRIPTS_DIR, 'simulation.py')
    run_script(simulation_script, ["--n_simulations", "20", "--n_trials", "150"])

    # 3. Run consolidated analytical scripts
    analytical_scripts = [
        'analyze_performance.py',
        'analyze_exploration.py',
        'analyze_weights.py',
        'analyze_planet_distributions.py',
        'visualize_simulation_arms.py'
    ]

    for script_name in analytical_scripts:
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        if os.path.exists(script_path):
            run_script(script_path)
        else:
            print(f"Skipping: {script_name} not found.")

    print("\n[DONE] All cleaning and consolidated execution tasks finished.")
