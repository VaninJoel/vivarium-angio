"""
Parallel runner for multiple angiogenesis simulations using multiprocessing.

This demonstrates the integrated architecture's ability to run multiple simulations
concurrently while maintaining process isolation to avoid CC3D global state conflicts.
"""

from multiprocessing import Process
import sys
import time


def run_single_simulation(params):
    """
    Run a single angiogenesis simulation in an isolated process.

    This function is designed to be called via multiprocessing.Process,
    ensuring complete isolation of CC3D's global state.
    """
    from vivarium.core.engine import Engine
    from vivarium_angio.composites.angiogenesis_composer import AngiogenesisComposer

    exp_name = params.get('angiogenesis_process', {}).get('exp_name', 'unnamed')
    print(f"\n[{exp_name}] Starting simulation in process {Process().pid}...")

    start_time = time.time()

    try:
        # Create composer and engine
        composer = AngiogenesisComposer(params)
        composite = composer.generate()
        engine = Engine(composite=composite)

        # Run simulation
        print(f"[{exp_name}] Running for 100 steps...")
        engine.update(100.0)

        elapsed = time.time() - start_time
        print(f"[{exp_name}] ✓ Completed in {elapsed:.2f} seconds")

        # Get final data
        data = engine.emitter.get_data()
        last_timestep = max(data.keys())
        last_data = data[last_timestep]

        print(f"[{exp_name}] Final timestep: {last_timestep}")
        print(f"[{exp_name}] Output keys: {last_data.get('outputs', {}).keys()}")

        return True

    except Exception as e:
        print(f"[{exp_name}] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_parallel_simulations():
    """
    Launch multiple angiogenesis simulations in parallel processes.

    This demonstrates:
    1. Process isolation prevents CC3D global state conflicts
    2. Simulations run truly in parallel (utilizing multiple cores)
    3. Each simulation writes incrementally to shared zarr store
    """

    # Define simulation configurations
    simulation_configs = [
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
    print("PARALLEL ANGIOGENESIS SIMULATIONS")
    print(f"Launching {len(simulation_configs)} simulations in parallel...")
    print("="*60)

    # Create processes
    processes = []
    for config in simulation_configs:
        p = Process(target=run_single_simulation, args=(config,))
        processes.append(p)

    # Start all processes
    start_time = time.time()
    for p in processes:
        p.start()
        print(f"Started process PID: {p.pid}")

    # Wait for all to complete
    print("\nWaiting for all simulations to complete...")
    for p in processes:
        p.join()

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print(f"All simulations completed in {elapsed:.2f} seconds")
    print("="*60)

    # Check results
    success_count = sum(1 for p in processes if p.exitcode == 0)
    print(f"\nResults: {success_count}/{len(processes)} simulations succeeded")

    if success_count == len(processes):
        print("\n✓ All simulations completed successfully!")
        print("\nData saved to: temp_store.zarr")
        print("  - default_run/")
        print("  - jem_8_run/")
        print("  - jem_4_run/")
    else:
        print("\n✗ Some simulations failed. Check output above for details.")


if __name__ == '__main__':
    run_parallel_simulations()
