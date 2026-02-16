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

# Advanced 2D CNN Model
class ZebrafishCNN2D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, feature_dim=512):
        super(ZebrafishCNN2D, self).__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 512x512 -> 256x256
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 256x256 -> 128x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 32x32 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self.feature_dim = feature_dim
    
    def extract_features(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
    
    def forward(self, x):
        features = self.extract_features(x)
        output = self.classifier(features)
        return output

# Dataset (same as simple CNN)
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
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
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
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def main():
    print("ADVANCED 2D CNN TRAINING (ZebrafishCNN2D)")
    
    # Configuration
    config = {
        'num_epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_classes': 3,
        'feature_dim': 512,
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
    results_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create model
    model = ZebrafishCNN2D(
        input_channels=1, 
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim']
    ).to(device)
    
    print(f"Model: ZebrafishCNN2D")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler - FIXED: removed 'verbose' parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("="*80)
    print("TRAINING")
    print("="*80 + "\n")
    
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
            model_path = results_dir / 'cnn2d_best.pth'
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
            print(f" No improvement ({patience_counter}/{config['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Save final model
    final_model_path = results_dir / 'cnn2d_final.pth'
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
    print(f"  • cnn2d_best.pth - Best model (val acc: {best_val_acc:.2f}%)")
    print(f"  • cnn2d_final.pth - Final model")
    
    # Save training history
    history_path = results_dir / 'cnn2d_training_history.json'
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
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation', linewidth=2)
    ax2.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_val_acc:.2f}%')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figures_dir = project_dir / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / 'cnn2d_training_curves.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {fig_path}")
    
    plt.close()
    
    print("\nReady for feature extraction!")
if __name__ == "__main__":
    main()