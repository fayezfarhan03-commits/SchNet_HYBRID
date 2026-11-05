import os
import glob
import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import cKDTree  # Replaced torch_cluster with SciPy
from tqdm import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = "/kaggle/working/MCMD-GCNN/Dump data and Potential Energy data"
OUTPUT_FILE = "/kaggle/working/mcmd_training_data.pt"

NUM_TYPES = 3 
CUTOFF_RADIUS = 3.0 

def parse_dump_file(filepath):
    """Extracts atom types and coordinates from a LAMMPS dump file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    start_idx = 0
    for i, line in enumerate(lines):
        if "ITEM: ATOMS" in line:
            start_idx = i + 1
            break
            
    types, positions = [], []
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) < 5: continue
        
        # Convert LAMMPS types (1, 2, 3) to PyTorch indexing (0, 1, 2)
        atom_type = int(parts[1]) - 1 
        pos = [float(parts[2]), float(parts[3]), float(parts[4])]
        types.append(atom_type)
        positions.append(pos)
        
    return torch.tensor(types, dtype=torch.long), torch.tensor(positions, dtype=torch.float)

def create_edges_scipy(pos_tensor, radius):
    """Calculates atomic bonds using SciPy instead of torch_cluster."""
    tree = cKDTree(pos_tensor.numpy())
    pairs = tree.query_pairs(r=radius)
    
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
        
    # Unpack the pairs
    src, dst = zip(*pairs)
    
    # Graphs need to be undirected (A bonded to B means B bonded to A)
    row = list(src) + list(dst)
    col = list(dst) + list(src)
    
    return torch.tensor([row, col], dtype=torch.long)

# --- 2. MAIN PROCESSING LOOP ---
all_graphs = []
temperature_folders = glob.glob(os.path.join(DATA_DIR, "T=*K"))

print("Starting data processing pipeline...")

for temp_folder in temperature_folders:
    temp_val = temp_folder.split("T=")[-1].replace("K", "")
    print(f"\nProcessing temperature: {temp_val}K")
    
    label_file = os.path.join(temp_folder, f"Potential Energy vs step T_{temp_val}.txt")
    if not os.path.exists(label_file):
        print(f"  Warning: Label file missing for {temp_val}K, skipping.")
        continue
        
    label_data = np.loadtxt(label_file)
    dump_files = glob.glob(os.path.join(temp_folder, "dump.mc.*"))
    dump_files = sorted(dump_files, key=lambda x: int(x.split(".mc.")[-1]))
    
    for dump_file in tqdm(dump_files, desc=f"Building Graphs ({temp_val}K)"):
        step_number = int(dump_file.split(".mc.")[-1])
        
        matching_rows = np.where(label_data[:, 0] == step_number)[0]
        if len(matching_rows) == 0:
            continue
            
        correct_row = matching_rows[-1] 
        total_energy = label_data[correct_row, 1]
        energy_per_atom = total_energy / 13500.0
        y = torch.tensor([[energy_per_atom]], dtype=torch.float)
        
        # Parse Structure & Build One-Hot Features
        types, pos = parse_dump_file(dump_file)
        x = torch.zeros((len(types), NUM_TYPES), dtype=torch.float)
        x.scatter_(1, types.unsqueeze(1), 1.0) 
        
        # Build Edges with SciPy fallback
        edge_index = create_edges_scipy(pos, CUTOFF_RADIUS)
        
        # Create Graph
        graph = Data(x=x, edge_index=edge_index, y=y, pos=pos)
        graph.temp = int(temp_val)
        graph.step = step_number
        all_graphs.append(graph)

# --- 3. SAVE ---
print(f"\nProcessing complete. Total structures extracted: {len(all_graphs)}")
torch.save(all_graphs, OUTPUT_FILE)
print(f"Saved binary dataset to: {OUTPUT_FILE}")