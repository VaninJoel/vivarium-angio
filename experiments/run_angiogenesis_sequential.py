"""
Sequential runner for multiple angiogenesis simulations with process isolation.

This demonstrates how to run multiple simulations one after another,
using subprocess to ensure complete CC3D state isolation between runs.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_simulation_in_subprocess(params_dict):
    """
    Run a single simulation in a fresh subprocess.

    This ensures complete isolation of CC3D's global state between runs.
    """
    exp_name = params_dict.get('angiogenesis_process', {}).get('exp_name', 'unnamed')

    # Create a temporary Python script to run this simulation
    script_content = f"""
from vivarium.core.engine import Engine
from vivarium_angio.composites.angiogenesis_composer import AngiogenesisComposer

params = {params_dict}

print(f"Running simulation: {{params['angiogenesis_process']['exp_name']}}")

composer = AngiogenesisComposer(params)
composite = composer.generate()
engine = Engine(composite=composite)
engine.update(100.0)

print(f"Simulation {{params['angiogenesis_process']['exp_name']}} completed successfully")
"""

    # Write temporary script
    temp_script = Path(f"_temp_run_{exp_name}.py")
    temp_script.write_text(script_content)

    try:
        print(f"\n{'='*60}")
        print(f"Starting simulation: {exp_name}")
        print(f"{'='*60}")

        start_time = time.time()

        # Run in subprocess
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ {exp_name} completed in {elapsed:.2f}s")
            print(result.stdout)
            return True
        else:
            print(f"✗ {exp_name} FAILED (exit code {result.returncode})")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    finally:
        # Cleanup temp script
        if temp_script.exists():
            temp_script.unlink()


def main():
    """Run multiple simulations sequentially."""

    configs = [
        {
            'angiogenesis_process': {
                'exp_name': 'default_run',
            }
        },
        {
            'angiogenesis_process': {
                'exp_name': 'jem_8_run',
                'jem': 8.0,
            }
        },
        {
            'angiogenesis_process': {
                'exp_name': 'jem_4_run',
                'jem': 4.0,
            }
        },
    ]

    print("="*60)
    print("SEQUENTIAL ANGIOGENESIS SIMULATIONS")
    print(f"Running {len(configs)} simulations sequentially...")
    print("="*60)

    results = []
    start_time = time.time()

    for config in configs:
        success = run_simulation_in_subprocess(config)
        results.append(success)

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print(f"All simulations completed in {elapsed:.2f} seconds")
    print(f"Success: {sum(results)}/{len(results)}")
    print("="*60)

    if all(results):
        print("\n✓ All simulations succeeded!")
        print("\nData saved to: temp_store.zarr/experiments/")
        for config in configs:
            exp_name = config['angiogenesis_process']['exp_name']
            print(f"  - {exp_name}/")
    else:
        print("\n✗ Some simulations failed")


if __name__ == '__main__':
    main()
