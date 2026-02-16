import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: 512x512 -> latent_dim
        self.encoder = nn.Sequential(
            # 512 -> 256
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 256 -> 128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Latent space
        self.fc_encode = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 16 * 16)
        
        # Decoder: latent_dim -> 512x512
        self.decoder = nn.Sequential(
            # 16 -> 32
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 32 -> 64
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64 -> 128
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128 -> 256
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 256 -> 512
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc_encode(x)
        return x
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z

# Dataset (same as CNN training)
class FrameDataset(Dataset):
    def __init__(self, data_dir, condition_to_label=None):
       
        self.condition_to_label = condition_to_label or {
            'anesthetic': 0,
            'stimulant': 1,
            'control': 2
        }
        
        self.frames = []
        self.labels = []
        
        data_dir = Path(data_dir)
        
        print(f"Loading data from {data_dir}...")
        
        # Scan all condition folders
        for condition, label in self.condition_to_label.items():
            condition_dir = data_dir / condition
            if not condition_dir.exists():
                continue
            
            tif_files = sorted(condition_dir.glob('*.tif'))
            
            for tif_file in tqdm(tif_files, desc=f"Loading {condition}"):
                try:
                    stack = tifffile.imread(tif_file)
                    
                    # Extract all frames
                    if len(stack.shape) == 3:  # Multi-page stack
                        for frame in stack:
                            self.frames.append(frame)
                            self.labels.append(label)
                    elif len(stack.shape) == 2:  # Single frame
                        self.frames.append(stack)
                        self.labels.append(label)
                except Exception as e:
                    print(f"Error loading {tif_file.name}: {e}")
        
        print(f"Loaded {len(self.frames)} frames")
        
        # Count per class
        for condition, label in self.condition_to_label.items():
            count = sum(1 for l in self.labels if l == label)
            if count > 0:
                print(f"  {condition}: {count} frames")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        frame = np.expand_dims(frame, axis=0)
        
        return torch.from_numpy(frame).float(), label

def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    
    for images, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def visualize_reconstructions(model, dataloader, device, save_path, num_samples=8):
    model.eval()
    
    # Get a batch
    images, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructed, _ = model(images)
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i, 0].cpu(), cmap='gray')
        axes[0, i].set_title('Original', fontsize=9)
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i, 0].cpu(), cmap='gray')
        axes[1, i].set_title('Reconstructed', fontsize=9)
        axes[1, i].axis('off')
    
    plt.suptitle('Autoencoder Reconstructions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("\nAUTOENCODER TRAINING")
    
    # Configuration
    config = {
        'num_epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.001,
        'latent_dim': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'early_stopping_patience': 15
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    device = torch.device(config['device'])
    print(f"Using device: {device}\n")
    
    # Paths
    project_dir = Path('.')
    processed_dir = project_dir / 'data' / 'processed'
    results_dir = project_dir / 'results' / 'models'
    figures_dir = project_dir / 'results' / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if splits exist
    if not processed_dir.exists():
        print("ERROR: Processed data not found!")
        print("Run 02_create_splits.py first to create train/val/test splits")
        return
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = FrameDataset(processed_dir / 'train')
    val_dataset = FrameDataset(processed_dir / 'val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print()
    
    # Creating model
    model = ConvAutoencoder(
        input_channels=1,
        latent_dim=config['latent_dim']
    ).to(device)
    
    print(f"Model: ConvAutoencoder")
    print(f"Latent dimension: {config['latent_dim']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("TRAINING")
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print results
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {new_lr:.6f}", end="")
        if new_lr < old_lr:
            print(f" (reduced from {old_lr:.6f})")
        else:
            print()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = results_dir / 'autoencoder_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'history': history
            }, model_path)
            print(f" Best model saved! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f" No improvement ({patience_counter}/{config['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n Early stopping triggered after {epoch+1} epochs")
            break
        
        # Visualize reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0:
            recon_path = figures_dir / f'autoencoder_reconstruction_epoch{epoch+1}.png'
            visualize_reconstructions(model, val_loader, device, recon_path)
            print(f"  📊 Reconstructions saved: {recon_path}")
        
        print()
    
    # Save final model
    final_model_path = results_dir / 'autoencoder_final.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    print("TRAINING COMPLETE")
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Final validation loss: {val_loss:.6f}")
    print(f"Models saved in: {results_dir}")
    print(f"autoencoder_best.pth - Best model (val loss: {best_val_loss:.6f})")
    print(f"autoencoder_final.pth - Final model")
    
    # Save training history
    history_path = results_dir / 'autoencoder_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    
    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.axhline(y=best_val_loss, color='r', linestyle='--', alpha=0.5, 
               label=f'Best: {best_val_loss:.6f}')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=11)
    ax.set_title('Autoencoder Training', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = figures_dir / 'autoencoder_training_curve.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Training curve saved: {fig_path}")
    
    plt.close()
    
    # Final reconstructions
    final_recon_path = figures_dir / 'autoencoder_final_reconstructions.png'
    visualize_reconstructions(model, val_loader, device, final_recon_path, num_samples=8)
    print(f"Final reconstructions saved: {final_recon_path}")
    
    print("\nReady for feature extraction!")

if __name__ == "__main__":
    main()