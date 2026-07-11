#!/usr/bin/env python3
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_script(script_name):
    try:
        console.rule(f"[bold]{script_name}")
        process = subprocess.Popen(
            ["python", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            console.print(line, end="")

        process.wait()

        if process.returncode == 0:
            console.print(Panel(f"[green]SUCCESS:[/] {script_name} ran successfully."))
        else:
            console.print(
                Panel(
                    f"[red]FAILURE:[/] {script_name} encountered an error.",
                    border_style="red",
                )
            )
    except Exception as e:
        console.print(
            Panel(f"[red]ERROR:[/] Failed to run {script_name}. Exception: {e}")
        )


def run_examples(scripts):
    for script in scripts:
        run_script(script)


if __name__ == "__main__":
    scripts_to_run = [
        "../examples/InsetFedPatch_24GHz.py",
        "../examples/MicrostripLine.py",
        # FEM backend (requires the getdp binary on PATH or SIMPLEEMS_GETDP_BIN)
        "../examples/InsetFedPatch_24GHz_FEM.py",
        "../examples/MicrostripLine_FEM.py",
    ]
    run_examples(scripts_to_run)
