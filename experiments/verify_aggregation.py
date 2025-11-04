"""
Verification script to measure cell aggregation using cluster counting.
"""

import zarr
import numpy as np
from scipy.ndimage import label

def verify_aggregation():
    """
    Analyzes the zarr store to count cell clusters as a measure of aggregation.
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
    
    print("="*60)
    print("CELL AGGREGATION VERIFICATION")
    print("="*60)
    print("Analyzing cell clustering at timestep 10...")

    aggregation_results = {}

    exp_names = sorted(experiments.keys())

    for exp_name in exp_names:
        exp_group = experiments[exp_name]
        
        if '10' not in exp_group:
            print(f"\n- {exp_name}: Timestep 10 data not found. Skipping.")
            continue

        # Get parameters (which are nested in a 'params' dictionary)
        params_wrapper = dict(exp_group.attrs)
        params = params_wrapper.get('params', {})
        jem = params.get('jem', 'default')

        # Get cell data
        data = exp_group['10']['data'][:] 
        cell_types = data[:,:,:,0]
        
        # Create a binary image: 1 for EC cells, 0 for medium
        binary_cells = np.where(cell_types == 1, 1, 0)
        
        # Label connected components (clusters)
        labeled_array, num_clusters = label(binary_cells)
        
        aggregation_results[exp_name] = {
            'jem': jem,
            'num_clusters': num_clusters
        }

    if not aggregation_results:
        print("\nNo experiments with data at timestep 10 were found.")
        return

    print("\n--- Aggregation Results ---")
    # Sort results by jem value for clear comparison
    sorted_results = sorted(aggregation_results.items(), key=lambda item: item[1]['jem'])

    for exp_name, result in sorted_results:
        print(f"  - Experiment: {exp_name}")
        print(f"    jem value: {result['jem']}")
        print(f"    Number of cell clusters: {result['num_clusters']}")

    print("\n--- Interpretation ---")
    # Compare runs to draw a conclusion
    if len(sorted_results) > 1:
        jem_values = [res[1]['jem'] for res in sorted_results]
        cluster_counts = [res[1]['num_clusters'] for res in sorted_results]

        # Check if cluster count decreases as jem increases
        if all(cluster_counts[i] >= cluster_counts[i+1] for i in range(len(cluster_counts)-1)):
            print("Conclusion: As 'jem' increases, the number of clusters decreases.")
            print("This indicates that a HIGHER 'jem' value promotes MORE cell aggregation.")
            print("This aligns with the physical model where higher interface energy with the")
            print("medium drives cells together.")
        # Check if cluster count increases as jem increases
        elif all(cluster_counts[i] <= cluster_counts[i+1] for i in range(len(cluster_counts)-1)):
            print("Conclusion: As 'jem' increases, the number of clusters increases.")
            print("This indicates that a HIGHER 'jem' value promotes LESS cell aggregation (spreading).")
            print("This contradicts the expected physical model but matches the README's interpretation.")
        else:
            print("Conclusion: The relationship between 'jem' and aggregation is not monotonic.")
            print("Further analysis is needed.")
    else:
        print("Only one experiment was analyzed. Cannot compare aggregation trends.")

    print("\n" + "="*60)

if __name__ == "__main__":
    verify_aggregation()
