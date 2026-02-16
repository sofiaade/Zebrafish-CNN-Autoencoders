"""
Feature Visualization Script
Visualize and interpret what the CNN learned
- Filter visualizations
- Activation maps
- Saliency maps
- Feature importance analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tifffile
from tqdm import tqdm

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# Load CNN model architecture
class ZebrafishCNN2D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, feature_dim=512):
        super(ZebrafishCNN2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
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


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = ZebrafishCNN2D(
        input_channels=1,
        num_classes=config.get('num_classes', 3),
        feature_dim=config.get('feature_dim', 512)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def visualize_filters(model, save_dir, layer_name='First Conv Layer'):
    # Get first conv layer weights
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    
    if first_conv is None:
        print("No conv layer found!")
        return
    
    # Get weights
    weights = first_conv.weight.data.cpu().numpy()
    num_filters = min(32, weights.shape[0])  # Show up to 32 filters
    
    # Normalize weights for visualization
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Plot
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_filters):
        ax = axes[i]
        filter_img = weights_norm[i, 0]  # First channel
        ax.imshow(filter_img, cmap='gray')
        ax.set_title(f'Filter {i+1}', fontsize=8)
        ax.axis('off')
    
    # Hide remaining subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{layer_name} - Learned Filters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / 'learned_filters.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Filters visualization saved: {save_path}")


def get_activation_maps(model, image_tensor, device):    
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks on specific layers
    hooks = []
    layer_names = []
    
    for i, (name, module) in enumerate(model.features.named_modules()):
        if isinstance(module, nn.Conv2d):
            layer_names.append(f'Conv_{i}')
            hooks.append(module.register_forward_hook(hook_fn(f'Conv_{i}')))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor.to(device))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations, layer_names


def visualize_activation_maps(model, image, device, save_dir, sample_name='sample'):    
    # Prepare image
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float()
    
    # Get activations
    activations, layer_names = get_activation_maps(model, image_tensor, device)
    
    # Select specific layers to visualize
    layers_to_viz = [layer_names[0], layer_names[2], layer_names[4]]  # Early, mid, late
    
    for layer_name in layers_to_viz:
        if layer_name not in activations:
            continue
        
        activation = activations[layer_name].cpu().numpy()[0]  # Remove batch dim
        num_channels = min(16, activation.shape[0])  # Show up to 16 channels
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Activation maps
        for i in range(min(num_channels, 15)):
            ax = axes[i+1]
            act_map = activation[i]
            ax.imshow(act_map, cmap='hot')
            ax.set_title(f'Channel {i+1}', fontsize=9)
            ax.axis('off')
        
        # Hide remaining subplots
        for i in range(num_channels + 1, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Activation Maps - {layer_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = save_dir / f'{sample_name}_{layer_name}_activations.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  {layer_name} activations saved")


def compute_saliency_map(model, image, device, target_class=None):
    
    # Prepare image
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    image_tensor.requires_grad = True
    
    # Forward pass
    output = model(image_tensor)
    
    # If target class not specified, use predicted class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients
    saliency = image_tensor.grad.data.abs().squeeze().cpu().numpy()
    
    return saliency, target_class


def visualize_saliency_maps(model, images, labels, device, save_dir):
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    
    # Selecting one sample per condition
    samples_per_condition = {}
    for label in range(3):
        indices = np.where(labels == label)[0]
        if len(indices) > 0:
            samples_per_condition[label] = images[indices[0]]
    
    for label, image in samples_per_condition.items():
        saliency, predicted_class = compute_saliency_map(model, image, device)
        
        # Normalize saliency for visualization
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Create overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Saliency map
        axes[1].imshow(saliency_norm, cmap='hot')
        axes[1].set_title('Saliency Map', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, cmap='gray', alpha=0.6)
        axes[2].imshow(saliency_norm, cmap='hot', alpha=0.4)
        axes[2].set_title('Overlay', fontsize=11, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'{condition_names[label]} Sample (Predicted: {condition_names[predicted_class]})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = save_dir / f'saliency_map_{condition_names[label].lower()}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Saliency map saved: {save_path}")


def main():
    print("FEATURE VISUALIZATION & INTERPRETATION")
    
    # Paths
    project_dir = Path('.')
    models_dir = project_dir / 'results' / 'models'
    features_dir = project_dir / 'results' / 'features'
    viz_dir = project_dir / 'results' / 'visualization'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_path = models_dir / 'cnn2d_best.pth'
    
    if not model_path.exists():
        print(" ERROR: Trained CNN model not found!")
        print("  Run 04_train_advanced_cnn.py first")
        return
    
    print("Loading model...")
    model, device = load_model(model_path)
    print(f"  Model loaded from: {model_path.name}")
    print(f"  Using device: {device}\n")
    
    # Visualize learned filters
    print("VISUALIZING LEARNED FILTERS")
    visualize_filters(model, viz_dir)
    print()
    
    # Load some sample images for activation maps
    print("VISUALIZING ACTIVATION MAPS")

    # Load test data
    test_dir = project_dir / 'data' / 'processed' / 'test'
    
    # Get sample images from each condition
    sample_images = []
    sample_labels = []
    
    condition_mapping = {'anesthetic': 0, 'stimulant': 1, 'control': 2}
    
    for condition, label in condition_mapping.items():
        condition_dir = test_dir / condition
        if not condition_dir.exists():
            continue
        
        tif_files = list(condition_dir.glob('*.tif'))
        if len(tif_files) > 0:
            stack = tifffile.imread(tif_files[0])
            if len(stack.shape) == 3:
                sample_images.append(stack[0])  # First frame
            else:
                sample_images.append(stack)
            sample_labels.append(label)
    
    if len(sample_images) == 0:
        print(" No sample images found in test set")
    else:
        # Visualize activation maps for each sample
        for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
            condition_name = ['anesthetic', 'stimulant', 'control'][label]
            print(f"Processing {condition_name} sample...")
            visualize_activation_maps(model, image, device, viz_dir, 
                                    sample_name=f'{condition_name}')
    print()
    
    # Saliency maps
    print("\n GENERATING SALIENCY MAPS")
    
    if len(sample_images) > 0:
        sample_images_np = np.array(sample_images)
        sample_labels_np = np.array(sample_labels)
        visualize_saliency_maps(model, sample_images_np, sample_labels_np, 
                               device, viz_dir)
    print()
    
    print("\n VISUALIZATION COMPLETE")
    print(f"\n All visualizations saved in: {viz_dir}/")
    print("\n Feature visualization complete!")

if __name__ == "__main__":
    main()
