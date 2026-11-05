import torch
import numpy as np
from scipy.spatial import cKDTree

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_PATH = "/kaggle/input/datasets/fayezfarhan/atom3alloypt/mcmd_training_data.pt"
ATOM_NAMES = {0: 'Co', 1: 'Cr', 2: 'Ni'} # 0, 1, 2 mapping for argmax
NUM_NEIGHBORS = 12  
TARGET_TEMP = 750

print(f"📂 Loading dataset to find the final LAMMPS frame at {TARGET_TEMP}K...")
dataset = torch.load(DATASET_PATH, weights_only=False)

# ==========================================
# 2. FIND THE FINAL 750K FRAME
# ==========================================
target_frames = []
for g in dataset:
    if getattr(g, 'temp', None) == TARGET_TEMP:
        target_frames.append(g)

if not target_frames:
    raise ValueError(f"Could not find any frames with temp == {TARGET_TEMP} in the dataset!")

# Grab the frame with the maximum step (the final LAMMPS equilibrium state)
try:
    final_lammps_frame = max(target_frames, key=lambda x: getattr(x, 'step', -1))
    step_num = final_lammps_frame.step
except AttributeError:
    # Fallback if 'step' isn't explicitly saved, just take the last one appended
    final_lammps_frame = target_frames[-1]
    step_num = "Unknown (Last Appended)"

print(f"✅ Found final LAMMPS frame at Step {step_num}!")

# ==========================================
# 3. PREP DATA FOR SRO CALCULATION
# ==========================================
# Convert one-hot to integer types
types = torch.argmax(final_lammps_frame.x, dim=1).numpy()
pos = final_lammps_frame.pos.numpy()

# Approximate the periodic box boundaries from the atomic coordinates
min_x, max_x = np.min(pos[:, 0]) - 0.1, np.max(pos[:, 0]) + 0.1
min_y, max_y = np.min(pos[:, 1]) - 0.1, np.max(pos[:, 1]) + 0.1
min_z, max_z = np.min(pos[:, 2]) - 0.1, np.max(pos[:, 2]) + 0.1
boxsize = np.array([max_x - min_x, max_y - min_y, max_z - min_z])

# ==========================================
# 4. SRO CALCULATOR (WITH PBC FIX)
# ==========================================
def calculate_sro(types, pos, boxsize):
    unique_types = sorted(list(ATOM_NAMES.keys()))
    n_types = len(unique_types)
    total_atoms = len(types)
    
    concentrations = {t: np.sum(types == t) / total_atoms for t in unique_types}
    
    # The PBC wrapping fix for negative coordinates
    pos_wrapped = pos % boxsize
    tree = cKDTree(pos_wrapped, boxsize=boxsize)
    
    _, indices = tree.query(pos_wrapped, k=NUM_NEIGHBORS + 1)
    neighbor_indices = indices[:, 1:] 
    
    pair_counts = {i: {j: 0 for j in unique_types} for i in unique_types}
    
    for i in range(total_atoms):
        center_type = types[i]
        for neighbor_idx in neighbor_indices[i]:
            neighbor_type = types[neighbor_idx]
            pair_counts[center_type][neighbor_type] += 1
            
    sro_matrix = np.zeros((n_types, n_types))
    for i_idx, i_type in enumerate(unique_types):
        total_i_neighbors = np.sum(types == i_type) * NUM_NEIGHBORS
        for j_idx, j_type in enumerate(unique_types):
            p_ij = pair_counts[i_type][j_type] / total_i_neighbors
            c_j = concentrations[j_type]
            sro_matrix[i_idx, j_idx] = 1.0 - (p_ij / c_j)
            
    return sro_matrix, unique_types

# ==========================================
# 5. EXECUTE & PRINT
# ==========================================
sro_matrix, types_list = calculate_sro(types, pos, boxsize)

print(f"\n{'-'*40}")
print(f" LAMMPS GROUND TRUTH SRO (750K, Step {step_num}) ")
print(f"{'-'*40}")
header = "      " + " ".join([f"{ATOM_NAMES[t]:>8}" for t in types_list])
print(header)
for i, t_i in enumerate(types_list):
    row_str = f"{ATOM_NAMES[t_i]:<4} |"
    for j, t_j in enumerate(types_list):
        val = sro_matrix[i, j]
        row_str += f"{val:>8.4f} "
    print(row_str)