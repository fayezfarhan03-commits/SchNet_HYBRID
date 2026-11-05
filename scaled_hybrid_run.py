import torch
import numpy as np
import random
import math
import time
import os
import subprocess
import sys
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SchNet

# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_ATOMS = 30400             
kB = 8.617333262e-5  
TEMPERATURE = 750

MACRO_CYCLES = 10              
ML_CHUNKS_PER_MACRO = 1000      
SWAP_CHUNK_SIZE = 1           
NOISE_SCALE_FACTOR = 1.0       
LAMMPS_RELAX_STEPS = 500      

# Pushed to 7 to max out the 15GB T4 VRAM
BATCH_SIZE = 7                

# LAMMPS Executable (Removed MPI to prevent Kaggle root-user blocking)
LAMMPS_CMD = ["lmp"]

# Update these paths before running
START_DATA_FILE = "/kaggle/input/datasets/fayezfarhan/relaxed30kdata/start_30k_relaxed.data"
POTENTIAL_FILE = "/kaggle/input/datasets/fayezfarhan/eamfile/NiCoCr.lammps.eam"
MODEL_WEIGHTS = "/kaggle/input/models/fayezfarhan/schnet-mea-model/pytorch/default/1/best_schnet_model.pth"
TRAJ_FILE = "Hybrid_30k_750K_True_SRO.lammpstrj"

print("Initializing Batched 30k Hybrid Engine...")

# ==========================================
# 2. LOAD SCHNET & PARSE 30K LAMMPS DATA
# ==========================================
model = SchNet(
    hidden_channels=128, num_filters=128, num_interactions=3,
    num_gaussians=50, cutoff=5.0, max_num_neighbors=32, readout='add'
).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True))
model.eval()

def load_lammps_data(filepath):
    print(f"Loading 30k structure from: {filepath}")
    with open(filepath, 'r') as f: lines = f.readlines()
    bounds, atom_data, reading_atoms = [0.0]*6,