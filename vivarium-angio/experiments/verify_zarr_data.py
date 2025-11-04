"""
Verification script to check zarr store contents and validate parameter differences.
Run this in the angio_GUI3 conda environment.
"""

import zarr
import numpy as np

# Open the zarr store
store_path = 'temp_store.zarr'
root = zarr.open(store_path, mode='r')

print("="*60)
print("ZARR STORE STRUCTURE")
print("="*60)

# List experiments
experiments = root['experiments']
print(f"\nFound {len(experiments)} experiments:")
for exp_name in experiments.keys():
    print(f"  - {exp_name}")

print("\n" + "="*60)
print("EXPERIMENT PARAMETERS")
print("="*60)

# Check parameters for each experiment
for exp_name in experiments.keys():
    exp_group = experiments[exp_name]
    params = dict(exp_group.attrs)
    print(f"\n{exp_name}:")
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")

    # List available timesteps
    timesteps = sorted([int(k) for k in exp_group.keys() if k.isdigit()])
    print(f"  Available timesteps: {timesteps}")

print("\n" + "="*60)
print("DATA VERIFICATION AT STEP 10")
print("="*60)

# Verify data at step 10 for each experiment
for exp_name in experiments.keys():
    exp_group = experiments[exp_name]

    if '10' in exp_group:
        data = exp_group['10']['data'][:]
        print(f"\n{exp_name}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Channels: [type, id, VEGF]")

        # Check cell types (channel 0)
        cell_types = data[:,:,:,0]
        unique_types = np.unique(cell_types)
        print(f"  Unique cell types: {unique_types} (0=Medium, 1=EC)")

        # Count cells (non-zero IDs in channel 1)
        cell_ids = data[:,:,:,1]
        num_cells = len(np.unique(cell_ids)) - 1  # -1 to exclude medium (id=0)
        print(f"  Number of cells (unique IDs): {num_cells}")

        # Analyze VEGF field (channel 2)
        vegf = data[:,:,:,2]
        print(f"  VEGF field stats:")
        print(f"    Min: {vegf.min():.6f}")
        print(f"    Max: {vegf.max():.6f}")
        print(f"    Mean: {vegf.mean():.6f}")
        print(f"    Std: {vegf.std():.6f}")

        # Show VEGF at a sample location
        sample_vegf = vegf[100, 100, 0]
        print(f"    VEGF at center (100,100): {sample_vegf:.6f}")
    else:
        print(f"\n{exp_name}: No step 10 data found")

print("\n" + "="*60)
print("COMPARING EXPERIMENTS (PARAMETER IMPACT)")
print("="*60)

# Compare VEGF fields between experiments to verify parameter effects
exp_names = list(experiments.keys())

# Get jem values for each experiment
jem_values = {}
for exp_name in exp_names:
    exp_group = experiments[exp_name]
    params = dict(exp_group.attrs)
    jem_values[exp_name] = params.get('jem', 'unknown')

print(f"\nParameter values (jem - EC-Medium adhesion):")
for exp_name, jem in sorted(jem_values.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0):
    print(f"  {exp_name}: jem={jem}")

if all('10' in experiments[name] for name in exp_names):
    print("\nVEGF field comparison at step 10:")

    vegf_data = {}
    for exp_name in exp_names:
        data = experiments[exp_name]['10']['data'][:]
        vegf = data[:,:,:,2]
        vegf_data[exp_name] = vegf
        jem = jem_values[exp_name]
        print(f"  {exp_name} (jem={jem}): mean={vegf.mean():.6f}, std={vegf.std():.6f}")

    # Pairwise comparisons
    print("\nPairwise differences:")
    exp_list = list(exp_names)
    for i in range(len(exp_list)):
        for j in range(i+1, len(exp_list)):
            exp1, exp2 = exp_list[i], exp_list[j]
            diff = np.abs(vegf_data[exp1] - vegf_data[exp2])
            print(f"  {exp1} vs {exp2}:")
            print(f"    Max diff: {diff.max():.6f}")
            print(f"    Mean diff: {diff.mean():.6f}")

            if diff.max() > 0.001:
                print(f"    ✓ Simulations are DIFFERENT (parameter effect detected)")
            else:
                print(f"    ✗ WARNING: Simulations appear identical!")

print("\n" + "="*60)
print("CELL DISTRIBUTION COMPARISON")
print("="*60)

if all('10' in experiments[name] for name in exp_names):
    print("\nCell counts at step 10:")

    for exp_name in exp_names:
        data = experiments[exp_name]['10']['data'][:]
        cell_ids = data[:,:,:,1]
        num_cells = len(np.unique(cell_ids)) - 1

        # Calculate cell density (EC pixels / total pixels)
        cell_types = data[:,:,:,0]
        ec_pixels = np.sum(cell_types == 1)
        total_pixels = cell_types.size
        density = ec_pixels / total_pixels

        jem = jem_values[exp_name]
        print(f"  {exp_name} (jem={jem}):")
        print(f"    Cells: {num_cells}")
        print(f"    EC pixels: {ec_pixels}")
        print(f"    Density: {density:.4f}")

print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)

success = True
issues = []

# Check 1: All experiments have data
for exp_name in exp_names:
    if '10' not in experiments[exp_name]:
        success = False
        issues.append(f"Missing data for {exp_name}")

# Check 2: Parameters are different
unique_jems = set(jem_values.values())
if len(unique_jems) <= 1 and 'unknown' not in unique_jems:
    success = False
    issues.append("All experiments have same jem parameter")

# Check 3: VEGF fields are different
if all('10' in experiments[name] for name in exp_names) and len(exp_names) >= 2:
    all_same = True
    exp_list = list(exp_names)
    for i in range(len(exp_list)):
        for j in range(i+1, len(exp_list)):
            exp1, exp2 = exp_list[i], exp_list[j]
            data1 = experiments[exp1]['10']['data'][:]
            data2 = experiments[exp2]['10']['data'][:]
            vegf1 = data1[:,:,:,2]
            vegf2 = data2[:,:,:,2]
            diff = np.abs(vegf1 - vegf2)
            if diff.max() > 0.001:
                all_same = False
                break
        if not all_same:
            break

    if all_same:
        success = False
        issues.append("VEGF fields are identical across experiments")

if success:
    print("\n✅ VERIFICATION PASSED")
    print("   - All experiments have data")
    print("   - Parameters are different across experiments")
    print("   - Simulation results show parameter effects")
    print("\n   The integrated architecture is working correctly!")
else:
    print("\n❌ VERIFICATION FAILED")
    for issue in issues:
        print(f"   - {issue}")

print("\n" + "="*60)
