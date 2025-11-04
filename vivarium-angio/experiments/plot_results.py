import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_experiment_comparison():
    """
    Creates a side-by-side plot of cell fields from multiple experiments,
    comparing the initial (step 10) and final (step 100) states.
    """
    store_path = 'temp_store.zarr'
    try:
        root = zarr.open(store_path, mode='r')
    except zarr.errors.PathNotFoundError:
        print(f"Error: Zarr store not found at '{store_path}'.")
        print("Please run a simulation (e.g., 'python run_angiogenesis_parallel.py') first.")
        return

    if 'experiments' not in root:
        print("Error: 'experiments' group not found in the zarr store.")
        return

    experiments = root['experiments']
    
    # Define the experiments and order
    exp_keys = {
        'default_run': 2.0,
        'jem_4_run': 4.0,
        'jem_8_run': 8.0,
    }
    
    # Sort experiments by jem value for a logical plot layout
    sorted_exps = sorted(exp_keys.items(), key=lambda item: item[1])
    
    if not all(name in experiments for name, jem in sorted_exps):
        print("One or more required experiments (default_run, jem_4_run, jem_8_run) not found.")
        return

    # Create a 2xN grid for initial vs. final states
    num_exps = len(sorted_exps)
    fig, axes = plt.subplots(2, num_exps, figsize=(5 * num_exps, 10.5))

    # Define custom colormap: 0 (Medium) -> white, 1 (EC) -> dark red
    colors = ['white', '#8B0000']
    cmap = ListedColormap(colors)

    print(f"Generating 2x{num_exps} plot for initial (step 10) vs. final (step 100) states...")

    for col, (exp_name, jem_val) in enumerate(sorted_exps):
        exp_group = experiments[exp_name]
        
        # --- Plot initial state (Step 10) ---
        ax_initial = axes[0, col]
        if '10' in exp_group:
            data_initial = exp_group['10']['data'][:]
            cell_types_initial = data_initial[:, :, :, 0]
            ax_initial.imshow(cell_types_initial, cmap=cmap, interpolation='none')
            ax_initial.set_title(f"jem = {jem_val}\nStep 10 (Initial)")
        else:
            ax_initial.set_title(f"jem = {jem_val}\n(No data at step 10)")
            ax_initial.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        ax_initial.set_xticks([])
        ax_initial.set_yticks([])

        # --- Plot final state (Step 100) ---
        ax_final = axes[1, col]
        if '100' in exp_group:
            data_final = exp_group['100']['data'][:]
            cell_types_final = data_final[:, :, :, 0]
            ax_final.imshow(cell_types_final, cmap=cmap, interpolation='none')
            ax_final.set_title(f"Step 100 (Final)")
        else:
            ax_final.set_title(f"Step 100\n(No data)")
            ax_final.text(0.5, 0.5, 'No data', ha='center', va='center')

        ax_final.set_xticks([])
        ax_final.set_yticks([])

    fig.suptitle('Effect of EC-Medium Adhesion (jem) on Network Morphology', fontsize=16, y=0.97)
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle
    
    output_filename = 'aggregation_comparison_detailed.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved successfully as '{output_filename}'")

if __name__ == "__main__":
    plot_experiment_comparison()
