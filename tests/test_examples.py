#!/usr/bin/env python3
import subprocess
from tqdm import tqdm  # Import tqdm for the progress bar


def run_script(script_name):
    try:
        # Run the script using subprocess
        result = subprocess.run(["python", script_name], capture_output=True, text=True)

        # Check if the script ran successfully
        if result.returncode == 0:
            print(f"SUCCESS: {script_name} ran successfully.")
        else:
            print(f"FAILURE: {script_name} encountered an error.")
            print(f"Error Output:\n{result.stderr}")
    except Exception as e:
        print(f"ERROR: Failed to run {script_name}. Exception: {e}")


def run_examples(scripts):
    # Wrap the script list with tqdm to show progress
    for script in tqdm(scripts, desc="Running Scripts", unit="script"):
        run_script(script)


if __name__ == "__main__":
    scripts_to_run = [
        "../examples/InsetFedPatch_60GHz.py",
        "../examples/InsetFedPatch_2_45GHz.py",
        "../examples/ProbeFedPatch_2_45GHz.py",
    ]
    run_examples(scripts_to_run)
