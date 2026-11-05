import torch
import numpy as np
import random
import math
import time
import os
import subprocess
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SchNet

# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_ATOMS = 13500
kB = 8.617333262e-5  
TEMPERATURE = 750

# --- 750K HYBRID CONTROLS ---
MACRO_CYCLES = 10      #10,300,25,1,2000,16        
ML_CHUNKS_PER_MACRO = 1000      
SWAP_CHUNK_SIZE = 1           # Swapping out 30 atoms at once to push through ML noise
NOISE_SCALE_FACTOR = 1.0       # 750K thermodynamics
LAMMPS_RELAX_STEPS = 500      # Relaxation steps for large SRO clusters
BATCH_SIZE = 16                 # Evaluates 8 alternate states simultaneously (VRAM Safe)

LAMMPS_EXEC = "lmp" 
POTENTIAL_FILE = "/kaggle/input/datasets/fayezfarhan/eamfile/NiCoCr.lammps.eam"
TRAJ_FILE = "Hybrid_750K_True_SRO.lammpstrj"

print("Initializing Batched 750K Hybrid SchNet-LAMMPS Engine...")

# ==========================================
# 2. LOAD SCHNET & DATA
# ==========================================
model = SchNet(
    hidden_channels=128, num_filters=128, num_interactions=3,
    num_gaussians=50, cutoff=5.0, max_num_neighbors=32, readout='add'
).to(device)
model.load_state_dict(torch.load("/kaggle/input/models/fayezfarhan/schnet-mea-model/pytorch/default/1/best_schnet_model.pth"))
model.eval()

dataset = torch.load("/kaggle/input/datasets/fayezfarhan/atom3alloypt/mcmd_training_data.pt", weights_only=False)

# Find 750K Step 200
start_frame = None
for g in dataset:
    if getattr(g, 'temp', None) == TEMPERATURE and getattr(g, 'step', None) == 200:
        start_frame = g
        break
if start_frame is None: start_frame = dataset[0]

# Initialize state
current_pos = start_frame.pos.clone().numpy()
current_types = torch.argmax(start_frame.x, dim=1).numpy() # 0=Co, 1=Cr, 2=Ni
box_bounds = [np.min(current_pos[:,0])-0.1, np.max(current_pos[:,0])+0.1,
              np.min(current_pos[:,1])-0.1, np.max(current_pos[:,1])+0.1,
              np.min(current_pos[:,2])-0.1, np.max(current_pos[:,2])+0.1]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_ml_energy(types_array, pos_array):
    """Calculates total box energy using SchNet for a single graph."""
    z_tensor = torch.tensor(types_array, dtype=torch.long)
    pos_tensor = torch.tensor(pos_array, dtype=torch.float)
    graph = Data(z=z_tensor, pos=pos_tensor, num_nodes=NUM_ATOMS)
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        return float(model(batch.z, batch.pos, batch.batch).item()) # Pure float guarantee

def run_lammps_relaxation(step, types_array, pos_array, bounds):
    """Writes data, runs LAMMPS NPT, and reads back the relaxed coordinates and energy."""
    data_file = "temp_relax.data"
    in_file = "temp_relax.in"
    dump_file = "temp_relaxed.dump"
    
    with open(data_file, 'w') as f:
        f.write(f"LAMMPS Hybrid Data Step {step}\n\n")
        f.write(f"{NUM_ATOMS} atoms\n3 atom types\n\n")
        f.write(f"{bounds[0]:.5f} {bounds[1]:.5f} xlo xhi\n")
        f.write(f"{bounds[2]:.5f} {bounds[3]:.5f} ylo yhi\n")
        f.write(f"{bounds[4]:.5f} {bounds[5]:.5f} zlo zhi\n\n")
        f.write("Atoms\n\n")
        for i in range(NUM_ATOMS):
            f.write(f"{i+1} {types_array[i]+1} {pos_array[i][0]:.5f} {pos_array[i][1]:.5f} {pos_array[i][2]:.5f}\n")

    lammps_script = f"""
    units metal
    atom_style atomic
    boundary p p p
    read_data {data_file}
    
    pair_style eam/alloy
    pair_coeff * * {POTENTIAL_FILE} Co Cr Ni
    
    thermo 100
    thermo_style custom step pe
    
    fix 1 all npt temp {TEMPERATURE} {TEMPERATURE} 0.1 iso 0 0 1 drag 1.8
    run {LAMMPS_RELAX_STEPS}
    
    variable final_pe equal pe
    print "FINAL_LAMMPS_ENERGY: ${{final_pe}}"
    
    dump 1 all custom 1 {dump_file} id type x y z
    dump_modify 1 sort id
    run 0
    """
    with open(in_file, 'w') as f: f.write(lammps_script)
    
    result = subprocess.run([LAMMPS_EXEC, "-in", in_file], capture_output=True, text=True)
    
    lammps_total_e = 0.0
    for line in result.stdout.split('\n'):
        if "FINAL_LAMMPS_ENERGY:" in line:
            lammps_total_e = float(line.split(":")[1].strip())
            
    new_pos = np.zeros((NUM_ATOMS, 3))
    new_bounds = [0]*6
    with open(dump_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "ITEM: BOX BOUNDS" in line:
            xb = lines[i+1].split(); new_bounds[0], new_bounds[1] = float(xb[0]), float(xb[1])
            yb = lines[i+2].split(); new_bounds[2], new_bounds[3] = float(yb[0]), float(yb[1])
            zb = lines[i+3].split(); new_bounds[4], new_bounds[5] = float(zb[0]), float(zb[1])
        elif "ITEM: ATOMS" in line:
            for j in range(NUM_ATOMS):
                parts = lines[i+1+j].split()
                atom_id = int(parts[0]) - 1 
                new_pos[atom_id] = [float(parts[2]), float(parts[3]), float(parts[4])]
            break
            
    return new_pos, new_bounds, lammps_total_e

def append_to_trajectory(step, types_array, pos_array, bounds):
    """Saves the current frame to a .lammpstrj file for SRO analysis."""
    with open(TRAJ_FILE, 'a') as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{NUM_ATOMS}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{bounds[0]:.5f} {bounds[1]:.5f}\n")
        f.write(f"{bounds[2]:.5f} {bounds[3]:.5f}\n")
        f.write(f"{bounds[4]:.5f} {bounds[5]:.5f}\n")
        f.write("ITEM: ATOMS id type x y z\n")
        for i in range(NUM_ATOMS):
            f.write(f"{i+1} {types_array[i]+1} {pos_array[i][0]:.5f} {pos_array[i][1]:.5f} {pos_array[i][2]:.5f}\n")

# ==========================================
# 4. THE HYBRID EXECUTION LOOP
# ==========================================
if os.path.exists(TRAJ_FILE): os.remove(TRAJ_FILE)
append_to_trajectory(0, current_types, current_pos, box_bounds)

print(f"\nStarting Batched Hybrid Engine | Temperature: {TEMPERATURE}K")
current_energy = get_ml_energy(current_types, current_pos)
print(f"Start | ML Energy: {current_energy / NUM_ATOMS:.6f} eV/atom")

start_time = time.time()

for macro in range(1, MACRO_CYCLES + 1):
    # ------------------------------------------------
    # PHASE 1: THE BATCHED ML SPRINTER (VRAM Safe)
    # ------------------------------------------------
    accepted_chunks = 0
    attempts_this_macro = 0
    MAX_ATTEMPTS = 25000  
    
    pos_tensor = torch.tensor(current_pos, dtype=torch.float)
    
    while accepted_chunks < ML_CHUNKS_PER_MACRO and attempts_this_macro < MAX_ATTEMPTS:
        batch_graphs = []
        proposed_types_list = []
        
        for _ in range(BATCH_SIZE):
            attempts_this_macro += 1
            prop_types = current_types.copy()
            
            for _ in range(SWAP_CHUNK_SIZE):
                idx1, idx2 = random.sample(range(NUM_ATOMS), 2)
                while prop_types[idx1] == prop_types[idx2]:
                    idx1, idx2 = random.sample(range(NUM_ATOMS), 2)
                prop_types[idx1], prop_types[idx2] = prop_types[idx2], prop_types[idx1]
                
            proposed_types_list.append(prop_types)
            z_tensor = torch.tensor(prop_types, dtype=torch.long)
            batch_graphs.append(Data(z=z_tensor, pos=pos_tensor, num_nodes=NUM_ATOMS))
            
        pyg_batch = Batch.from_data_list(batch_graphs).to(device)
        with torch.no_grad():
            batch_energies = model(pyg_batch.z, pyg_batch.pos, pyg_batch.batch).cpu().numpy()
            
        # Memory Cleanup: Prevent OOM by clearing VRAM
        del pyg_batch
        torch.cuda.empty_cache()
            
        for b in range(BATCH_SIZE):
            # Forcing the numpy array element into a pure Python float
            new_energy = float(batch_energies[b].item() if hasattr(batch_energies[b], 'item') else batch_energies[b])
            scaled_delta_e = (new_energy - current_energy) * NOISE_SCALE_FACTOR
            
            if scaled_delta_e < 0 or random.random() < math.exp(-scaled_delta_e / (kB * TEMPERATURE)):
                current_types = proposed_types_list[b]
                current_energy = new_energy
                accepted_chunks += 1
                break 
                
    acc_rate = (accepted_chunks / attempts_this_macro) * 100 if attempts_this_macro > 0 else 0
    print(f"ML Sprint {macro}/{MACRO_CYCLES} | Acc: {accepted_chunks}/{ML_CHUNKS_PER_MACRO} | Rate: {acc_rate:.2f}% | E: {current_energy/NUM_ATOMS:.6f} eV/atom")
    
    if accepted_chunks < 10:
        print("Crystal is completely frozen. Reached absolute ground state. Stopping early.")
        break
    
    # ------------------------------------------------
    # PHASE 2: LATTICE RELAXATION (LAMMPS)
    # ------------------------------------------------
    print(f"   Handing off to LAMMPS for relaxation...")
    current_pos, box_bounds, lammps_total_e = run_lammps_relaxation(macro, current_types, current_pos, box_bounds)
    
    # ------------------------------------------------
    # PHASE 3: SIDE-BY-SIDE EVALUATION & SAVING
    # ------------------------------------------------
    new_ml_total_e = get_ml_energy(current_types, current_pos)
    current_energy = new_ml_total_e  
    
    append_to_trajectory(macro, current_types, current_pos, box_bounds)
    
    print(f"   COMPARISON (Macro {macro}):")
    print(f"      LAMMPS True Energy : {lammps_total_e / NUM_ATOMS:.6f} eV/atom")
    print(f"      SchNet Pred Energy : {new_ml_total_e / NUM_ATOMS:.6f} eV/atom")
    print(f"      Delta (Error)      : {abs((lammps_total_e - new_ml_total_e) / NUM_ATOMS):.6f} eV/atom\n")

total_time = (time.time() - start_time) / 60
print(f"750K Hybrid Simulation complete in {total_time:.2f} minutes.")