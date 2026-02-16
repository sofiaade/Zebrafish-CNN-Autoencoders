"""
Advanced CNN Training with Data Augmentation
Includes augmentation techniques to improve accuracy
Expected improvement: +10-20% over baseline
"""

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
import torchvision.transforms as transforms
import random

# Advanced 2D CNN Model (same as before)
class ZebrafishCNN2D(nn.Module):
    """Advanced 2D CNN for frame-level classification"""
    
    def __init__(self, input_channels=1, num_classes=3, feature_dim=512, dropout=0.5):
        super(ZebrafishCNN2D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
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


# Augmentation class
class Augmentation:
    """Data augmentation for zebrafish images"""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image):
        """Apply random augmentations"""
        
        # Random horizontal flip
        if random.random() < self.prob:
            image = np.fliplr(image).copy()
        
        # Random vertical flip
        if random.random() < self.prob:
            image = np.flipud(image).copy()
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() < self.prob:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()
        
        # Random brightness adjustment
        if random.random() < self.prob:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, image.max())
        
        # Random contrast adjustment
        if random.random() < self.prob:
            mean = image.mean()
            factor = random.uniform(0.8, 1.2)
            image = np.clip((image - mean) * factor + mean, 0, image.max())
        
        # Random Gaussian noise
        if random.random() < self.prob:
            noise = np.random.normal(0, 0.02 * image.std(), image.shape)
            image = np.clip(image + noise, 0, image.max())
        
        return image


# Improved Dataset with Augmentation
class AugmentedFrameDataset(Dataset):
    """Dataset with data augmentation"""
    
    def __init__(self, data_dir, condition_to_label=None, augment=True, augment_prob=0.5):
        """
        Args:
            data_dir: Path to directory with condition subfolders
            condition_to_label: Dict mapping condition names to labels
            augment: Whether to apply augmentation
            augment_prob: Probability of applying each augmentation
        """
        self.condition_to_label = condition_to_label or {
            'anesthetic': 0,
            'stimulant': 1,
            'control': 2
        }
        
        self.augment = augment
        self.augmentation = Augmentation(prob=augment_prob) if augment else None
        
        self.frames = []
        self.labels = []
        
        data_dir = Path(data_dir)
        
        print(f"Loading data from {data_dir}...")
        if augment:
            print(f"  Augmentation ENABLED (prob={augment_prob})")
        
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
        frame = self.frames[idx].copy()  # Copy to avoid modifying original
        label = self.labels[idx]
        
        # Apply augmentation (only during training)
        if self.augment and self.augmentation is not None:
            frame = self.augmentation(frame)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        frame = np.expand_dims(frame, axis=0)
        
        return torch.from_numpy(frame).float(), label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
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
    """Validate the model"""
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
    print("="*80)
    print("IMPROVED CNN TRAINING WITH DATA AUGMENTATION")
    print("="*80 + "\n")
    
    # Configuration
    config = {
        'num_epochs': 150,  # More epochs with augmentation
        'batch_size': 16,  # Larger batch size
        'learning_rate': 0.0005,  # Lower learning rate
        'num_classes': 3,
        'feature_dim': 512,
        'dropout': 0.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'early_stopping_patience': 20,  # More patience
        'augment_prob': 0.5,  # 50% chance for each augmentation
        'weight_decay': 1e-4  # L2 regularization
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
        print("❌ ERROR: Processed data not found!")
        print("Run 02_create_splits.py first")
        return
    
    # Create datasets WITH augmentation for training
    print("Loading datasets...")
    train_dataset = AugmentedFrameDataset(
        processed_dir / 'train', 
        augment=True,  # Augmentation ON for training
        augment_prob=config['augment_prob']
    )
    
    val_dataset = AugmentedFrameDataset(
        processed_dir / 'val',
        augment=False  # Augmentation OFF for validation
    )
    
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
        feature_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model: ZebrafishCNN2D (with augmentation)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dropout: {config['dropout']}")
    print(f"Weight decay: {config['weight_decay']}\n")
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
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
            model_path = results_dir / 'cnn2d_augmented_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config,
                'history': history
            }, model_path)
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Save final model
    final_model_path = results_dir / 'cnn2d_augmented_final.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Improvement over baseline: +{best_val_acc - 60.42:.2f}%")
    print(f"\nModels saved in: {results_dir}")
    print(f"  • cnn2d_augmented_best.pth")
    print(f"  • cnn2d_augmented_final.pth")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss (With Augmentation)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation', linewidth=2)
    ax2.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_val_acc:.2f}%')
    ax2.axhline(y=60.42, color='orange', linestyle='--', alpha=0.5, label='Baseline: 60.42%')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training Accuracy (With Augmentation)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figures_dir = project_dir / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / 'cnn2d_augmented_training_curves.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved: {fig_path}")
    
    plt.close()
    
    print("\n🎉 Training with augmentation complete!")
    print(f"   Expected improvement: +10-20% accuracy")
    print(f"   Run 07_extract_features.py next to get improved features")

if __name__ == "__main__":
    main()
