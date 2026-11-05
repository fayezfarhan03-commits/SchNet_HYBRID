import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# ==========================================
# 1. REPRODUCIBILITY & DEVICE SETUP
# ==========================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. DATASET LOADING & ADAPTATION
# ==========================================
DATASET_PATH = "/kaggle/input/datasets/fayezfarhan/atom3alloypt/mcmd_training_data.pt"
NUM_ATOMS = 13500  

print("Loading dataset into memory...")
dataset = torch.load(DATASET_PATH, weights_only=False)

print("Converting GCN format to SchNet format...")
# SchNet requires 'z' (atomic indices) instead of 'x' (one-hot encoding)
for data in dataset:
    # Converts [1,0,0], [0,1,0], [0,0,1] into 0, 1, 2
    data.z = torch.argmax(data.x, dim=1).long()
    data.num_nodes = NUM_ATOMS

# 80/20 Train-Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# ==========================================
# 3. DATALOADERS
# ==========================================
OPTIMIZED_BATCH_SIZE = 4  

train_loader = DataLoader(
    train_dataset, 
    batch_size=OPTIMIZED_BATCH_SIZE, 
    shuffle=True,
    pin_memory=True,  
    num_workers=2     
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=OPTIMIZED_BATCH_SIZE, 
    shuffle=False,
    pin_memory=True,
    num_workers=2
)

print(f"Data ready: {len(train_dataset)} Train | {len(test_dataset)} Test (Batch Size: {OPTIMIZED_BATCH_SIZE})")

# ==========================================
# 4. SCHNET MODEL SETUP
# ==========================================
print("Building SchNet architecture...")

model = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=3,
    num_gaussians=50,
    cutoff=5.0,           
    max_num_neighbors=32, 
    readout='add'         
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# ==========================================
# 5. TRAINING & TESTING FUNCTIONS
# ==========================================
def train():
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Output shape: [batch_size, 1] (Total Box Energy)
        pred_total_energy = model(batch.z, batch.pos, batch.batch)
        
        # Convert to Energy per Atom
        pred_energy_per_atom = pred_total_energy / NUM_ATOMS
        
        # Ensure target shape matches exactly
        target = batch.y.view(-1, 1) 
        
        loss = criterion(pred_energy_per_atom, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            pred_total_energy = model(batch.z, batch.pos, batch.batch)
            pred_energy_per_atom = pred_total_energy / NUM_ATOMS
            
            target = batch.y.view(-1, 1)
            loss = criterion(pred_energy_per_atom, target)
            total_loss += loss.item()
            
    return total_loss / len(test_loader)

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
epochs = 100 
train_losses = []
test_losses = []
best_test_loss = float('inf')

print(f"\nStarting training loop for {epochs} epochs...")
start_time = time.time()

for epoch in range(1, epochs + 1):
    epoch_start = time.time()
    
    train_loss = train()
    test_loss = test()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    # Save the best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "/kaggle/working/best_schnet_model.pth")
        saved_flag = "[Best model saved]"
    else:
        saved_flag = ""
        
    epoch_time = (time.time() - epoch_start) / 60
    print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f} | Time: {epoch_time:.2f}m {saved_flag}")

total_time = (time.time() - start_time) / 60
print(f"\nTraining complete. Total time: {total_time:.1f} minutes.")
print("Best weights saved at: /kaggle/working/best_schnet_model.pth")

# ==========================================
# 7. PLOT LEARNING CURVES
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (eV/atom)')
plt.title('SchNet Learning Curve (NiCoCr SRO Prediction)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()