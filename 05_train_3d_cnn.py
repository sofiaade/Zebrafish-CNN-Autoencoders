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

# 3D CNN Model
class ZebrafishCNN3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, sequence_length=16):
        super(ZebrafishCNN3D, self).__init__()
        
        self.sequence_length = sequence_length
        
        # 3D Convolutional layers
        self.features = nn.Sequential(
            # Input: (B, 1, T, H, W) - T=temporal, H=height, W=width
            # Block 1
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # Reduce all dimensions
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            
            # Block 4
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 4, 4))
        )
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(512, num_classes)
    
    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        features = self.fc[:-1](x)  # Before dropout
        return features
    
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x

# Dataset for 3D CNN (processes sequences)
class SequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=16, stride=8, condition_to_label=None):
        self.sequence_length = sequence_length
        self.stride = stride
        self.condition_to_label = condition_to_label or {
            'anesthetic': 0,
            'stimulant': 1,
            'control': 2
        }
        
        self.sequences = []
        self.labels = []
        
        data_dir = Path(data_dir)
        
        print(f"Loading sequences from {data_dir}...")
        print(f"Sequence length: {sequence_length}, Stride: {stride}")
        
        # Scan all condition folders
        for condition, label in self.condition_to_label.items():
            condition_dir = data_dir / condition
            if not condition_dir.exists():
                continue
            
            tif_files = sorted(condition_dir.glob('*.tif'))
            
            for tif_file in tqdm(tif_files, desc=f"Loading {condition}"):
                try:
                    stack = tifffile.imread(tif_file)
                    
                    if len(stack.shape) != 3:
                        print(f"Skipping {tif_file.name}: not a stack")
                        continue
                    
                    num_frames = stack.shape[0]
                    
                    # Extract overlapping sequences
                    for start_idx in range(0, num_frames - sequence_length + 1, stride):
                        end_idx = start_idx + sequence_length
                        sequence = stack[start_idx:end_idx]
                        
                        self.sequences.append(sequence)
                        self.labels.append(label)
                        
                except Exception as e:
                    print(f"Error loading {tif_file.name}: {e}")
        
        print(f"Loaded {len(self.sequences)} sequences")
        
        # Count per class
        for condition, label in self.condition_to_label.items():
            count = sum(1 for l in self.labels if l == label)
            if count > 0:
                print(f"  {condition}: {count} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Normalize to [0, 1]
        sequence = sequence.astype(np.float32)
        sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min() + 1e-8)
        
        # Add channel dimension: (T, H, W) -> (1, T, H, W)
        sequence = np.expand_dims(sequence, axis=0)
        
        return torch.from_numpy(sequence).float(), label

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating", leave=False):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def main():
    print("3D CNN TRAINING (ZebrafishCNN3D)")
    print("Spatiotemporal Feature Extraction")
    
    # Configuration
    config = {
        'num_epochs': 100,
        'batch_size': 4,  
        'learning_rate': 0.0005, 
        'num_classes': 3,
        'sequence_length': 16,
        'stride': 8, 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'early_stopping_patience': 15
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("3D CNN training on CPU may be slow!")
        print()
    
    # Paths
    project_dir = Path('.')
    processed_dir = project_dir / 'data' / 'processed'
    results_dir = project_dir / 'results' / 'models'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if splits exist
    if not processed_dir.exists():
        print("ERROR: Processed data not found!")
        print("Run 02_create_splits.py first to create train/val/test splits")
        return
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SequenceDataset(
        processed_dir / 'train',
        sequence_length=config['sequence_length'],
        stride=config['stride']
    )
    
    val_dataset = SequenceDataset(
        processed_dir / 'val',
        sequence_length=config['sequence_length'],
        stride=config['stride']
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No sequences extracted!")
        print("This could mean:")
        print("  • Your .tif files don't have enough frames")
        print(f"  • Need at least {config['sequence_length']} frames per file")
        print("  • Try reducing sequence_length in config")
        return
    
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
    
    # Create model
    model = ZebrafishCNN3D(
        input_channels=1, 
        num_classes=config['num_classes'],
        sequence_length=config['sequence_length']
    ).to(device)
    
    print(f"Model: ZebrafishCNN3D")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("TRAINING")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {new_lr:.6f}", end="")
        if new_lr < old_lr:
            print(f" (reduced from {old_lr:.6f})")
        else:
            print()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = results_dir / 'cnn3d_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config,
                'history': history
            }, model_path)
            print(f" Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Save final model
    final_model_path = results_dir / 'cnn3d_final.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    print("TRAINING COMPLETE")
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Models saved in: {results_dir}")
    print(f"  • cnn3d_best.pth - Best model (val acc: {best_val_acc:.2f}%)")
    print(f"  • cnn3d_final.pth - Final model")
    
    # Save training history
    history_path = results_dir / 'cnn3d_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('3D CNN - Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation', linewidth=2)
    ax2.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_val_acc:.2f}%')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('3D CNN - Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figures_dir = project_dir / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / 'cnn3d_training_curves.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {fig_path}")
    
    plt.close()
    
    print("\n Ready for spatiotemporal feature extraction!")

if __name__ == "__main__":
    main()