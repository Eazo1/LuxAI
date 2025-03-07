import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import wandb
# import obstruction_and_relic_matrix
# import gen_map_from_game
# import seed_to_matrix

#Note that GOM stands for Grand Obstruction Matrix

class MemoryMappedDataset(Dataset):
    def __init__(self, mmap_data, device):
        self.data = mmap_data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a tensor in the shape stored in the npy file.
        raw_data = self.data[idx][:,:,0]  # Choose just the first piece of information
        raw_data = torch.tensor(raw_data, dtype=torch.long)  # Now (49,49)
        one_hot = F.one_hot(raw_data, num_classes=4).permute(2,0,1).float()  # Now (4,49,49)
        return one_hot

    

# data_path = 'C:/Users/ahmad/Desktop/lux/GOM_seed_dataset/'

# # Packing all the npy files into a list, and separating them into training and validation sets
# data_files = os.listdir(data_path)
# np.random.shuffle(data_files)  # Shuffle the files

# # Split into training (80%) and validation (20%) sets
# split_idx = int(0.8 * len(data_files))
# train_files = data_files[:split_idx]
# valid_files = data_files[split_idx:]

# # Get full paths
# train_data_ = [os.path.join(data_path, f) for f in train_files]
# valid_data_ = [os.path.join(data_path, f) for f in valid_files]

# # Load the data using memory mapping (for large datasets)
# train_data_mmap = [np.load(f, mmap_mode='r') for f in train_data_]
# valid_data_mmap = [np.load(f, mmap_mode='r') for f in valid_data_]

# # Create datasets and loaders
# train_dataset = MemoryMappedDataset(train_data_mmap, device=None)
# valid_dataset = MemoryMappedDataset(valid_data_mmap, device=None)

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# Define residual stack
class ResidualBlock(nn.Module):
    def __init__(self, in_and_out_channels, residual_in_and_out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_and_out_channels,
            out_channels=residual_in_and_out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=residual_in_and_out_channels,
            out_channels=in_and_out_channels,
            kernel_size=1,
            stride=1)
    
    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return x + h

class ResidualStack(nn.Module):
    def __init__(self, in_and_out_channels, num_residual_layers, residual_in_and_out_channels):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList(
            [ResidualBlock(in_and_out_channels, residual_in_and_out_channels)
             for _ in range(num_residual_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_hiddens // 2, out_channels=num_hiddens // 4, kernel_size=4, stride=2, padding=1)
        self.conv2b = nn.Conv2d(in_channels=num_hiddens // 4, out_channels=latent_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1, padding=1)
        
        # Initialize residual stack as a module
        self.residual_stack = ResidualStack(latent_dim, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x)) 
        x = self.conv3(x)
        # Use the residual stack
        x = self.residual_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, input_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        
        # Initialize residual stack as a module
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        
        self.conv_trans1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=4, kernel_size=4, stride=2, padding=1)
        #self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=2, kernel_size=4, stride=2, padding=1)


    def forward(self, x, target_size):
        x = F.relu(self.conv1(x))
        x = self.residual_stack(x)  # Use the residual stack
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        # Dynamically resize to match original input size
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        #x = F.softmax(x, dim=1)
        return x

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim=latent_dim)
        self.decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, input_dim=latent_dim)

    def forward(self, x):
        original_size = x.shape[-2:]  # Extracts (H, W) from input
        z = self.encoder(x)
        x_recon = self.decoder(z, original_size)  # Pass original size to decoder
        return x_recon
    
    ###
    def reconstruct(self, x):
        with torch.no_grad():
            prob_output = self.forward(x)
            discrete_output = torch.argmax(prob_output, dim=1)
        return discrete_output

####################################################################################################################

# Setup parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 10
num_hiddens = 256
num_residual_layers = 2
num_residual_hiddens = 32
learning_rate = 2e-4
num_training_updates = 1700

'''wandb.login(key="7391c065d23aad000052bc1f7a3a512445ae83d0")
wandb.init(
    project="GOM_AE",
    config={
        "example_1_original_dim": len(train_dataset[0][0].flatten()),
        "latent_dim": latent_dim,
        "architecture": "AE",
        "num_training_updates": num_training_updates,
        "initial_learning_rate": learning_rate,
    },
    reinit=True,
)
'''
# wandb.watch_called = False  # Re-run the model without restarting the runtime
# # Instantiate AE and optimizer
# autoencoder = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim).to(device)
# optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# # Make sure no training happens when the model is in evaluation mode

# # Training loop
# criterion = nn.CrossEntropyLoss()
# train_losses = []
# iteration = 0
# autoencoder.train()
# print("Starting training...")
# while iteration < num_training_updates:
#     for goms in train_loader:
#         goms = goms.to(device)

#         optimizer.zero_grad()
#         recon = autoencoder(goms)
        
#         #print("Unique values in recon:", torch.unique(recon))
        
#         target = torch.argmax(goms, dim=1)  # Shape: (batch, H, W)
#         loss = criterion(recon, target)
#         loss.backward()
#         optimizer.step()

#         train_losses.append(loss.item())
#         #wandb.log({
#             #"train/loss": loss.item(),
#             #"losses/train": loss.item()
#         #})  

#         iteration += 1

#         if iteration % 100 == 0:
#             print(f"Reconstructed shape: {recon.shape}, Target shape: {target.shape}")
#             print(f"Iteration {iteration}, training loss: {loss.item():.4f}")
#         if iteration >= num_training_updates:
#             break
          
#     # Validation loop
#     autoencoder.eval()
#     with torch.no_grad():
#         total_val_loss = 0.0
#         num_batches = 0
#         for val_goms in valid_loader:
#             # Apply the same dimension fix as above.
#             val_goms = val_goms.to(device)
            
#             recon_val = autoencoder(val_goms)
#             target_val = torch.argmax(val_goms, dim=1)  
#             loss_val = criterion(recon_val, target_val)
#             total_val_loss += loss_val.item()
#             num_batches += 1
#         avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
#         #wandb.log({
#         #    "validation/loss": avg_val_loss,
#         #    "losses/val": avg_val_loss
#         #})
        
#         print(f"Validation loss: {avg_val_loss:.4f}")
#     autoencoder.train()

# # Evaluation (for visualization)
# autoencoder.eval()
# with torch.no_grad():
#     for goms in valid_loader:
#         # Again, fix the shape if needed.
#         if goms.dim() == 3:
#             goms = goms.unsqueeze(1)
#         goms = goms.to(device)
#         recon_goms = autoencoder.reconstruct(goms)
#         break

#Plot the first few matrices and their reconstructions
'''original_goms = torch.argmax(goms, dim=1)
num_plots = 4
fig, axes = plt.subplots(2, num_plots, figsize=(15, 5))
for i in range(num_plots):
    axes[0, i].imshow(original_goms[i].cpu().numpy(), cmap='viridis')
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon_goms[i].cpu().numpy(), cmap='viridis')
    axes[1, i].set_title('Reconstruction')
    axes[1, i].axis('off')

plt.tight_layout()
wandb.log({"Reconstructed GOMs": wandb.Image(fig)}, step=None)
plt.savefig('C:/Users/ahmad/Desktop/lux/' + 'ex_reconstructions.png')
plt.show()
plt.close(fig)'''


# #Save the model
# save_directory = 'C:/Users/ahmad/Desktop/lux/'
# model_save_path = os.path.join(save_directory, 'GOM_autoencoder_model.pth')
# torch.save(autoencoder.state_dict(), model_save_path)
# print("Model saved to", model_save_path)

# wandb.finish()
