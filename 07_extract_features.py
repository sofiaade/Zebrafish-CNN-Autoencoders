import torch
import torch.nn as nn
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import json

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


class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        
        self.fc_encode = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 16 * 16)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
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


def load_model(model_path, model_type='cnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    if model_type == 'cnn':
        model = ZebrafishCNN2D(
            input_channels=1,
            num_classes=config.get('num_classes', 3),
            feature_dim=config.get('feature_dim', 512)
        )
    elif model_type == 'autoencoder':
        model = ConvAutoencoder(
            input_channels=1,
            latent_dim=config.get('latent_dim', 128)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def extract_features_from_directory(data_dir, cnn_model, autoencoder_model, device, 
                                    condition_to_label=None):
    condition_to_label = condition_to_label or {
        'anesthetic': 0,
        'stimulant': 1,
        'control': 2
    }
    
    data_dir = Path(data_dir)
    
    all_cnn_features = []
    all_ae_features = []
    all_labels = []
    all_filenames = []
    
    print(f"Extracting features from {data_dir}...")
    
    for condition, label in condition_to_label.items():
        condition_dir = data_dir / condition
        if not condition_dir.exists():
            print(f" Skipping {condition}: folder not found")
            continue
        
        tif_files = sorted(condition_dir.glob('*.tif'))
        if len(tif_files) == 0:
            print(f"Skipping {condition}: no .tif files")
            continue
        
        print(f"Processing {condition}: {len(tif_files)} files")
        
        for tif_file in tqdm(tif_files, desc=f"  {condition}", leave=False):
            try:
                # Load stack
                stack = tifffile.imread(tif_file)
                
                # Handle both stacks and single frames
                if len(stack.shape) == 2:
                    stack = np.expand_dims(stack, axis=0)
                
                # Process each frame
                for frame_idx, frame in enumerate(stack):
                    # Normalize
                    frame = frame.astype(np.float32)
                    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                    
                    # Add batch and channel dimensions
                    frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # Extract CNN features
                        cnn_feat = cnn_model.extract_features(frame_tensor)
                        all_cnn_features.append(cnn_feat.cpu().numpy())
                        
                        # Extract autoencoder features
                        _, ae_feat = autoencoder_model(frame_tensor)
                        all_ae_features.append(ae_feat.cpu().numpy())
                        
                        # Store label and filename
                        all_labels.append(label)
                        all_filenames.append(f"{tif_file.name}_frame{frame_idx}")
                
            except Exception as e:
                print(f"Error processing {tif_file.name}: {e}")
    
    # Convert to arrays
    cnn_features = np.vstack(all_cnn_features)
    ae_features = np.vstack(all_ae_features)
    labels = np.array(all_labels)
    
    return cnn_features, ae_features, labels, all_filenames


def main():

    print("\nFEATURE EXTRACTION")
 
    # Paths
    project_dir = Path('.')
    models_dir = project_dir / 'results' / 'models'
    processed_dir = project_dir / 'data' / 'processed'
    features_dir = project_dir / 'results' / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for trained models
    cnn_path = models_dir / 'cnn2d_best.pth'
    ae_path = models_dir / 'autoencoder_best.pth'
    
    # Fallback to simple CNN if advanced not available
    if not cnn_path.exists():
        cnn_path = models_dir / 'cnn_best.pth'
        print(" Using SimpleCNN (cnn_best.pth) - ZebrafishCNN2D not found")
        print("   For better features, train advanced CNN with 04_train_advanced_cnn.py\n")
    
    if not cnn_path.exists():
        print("ERROR: No trained CNN model found!")
        print("Run 03_train_cnn.py or 04_train_advanced_cnn.py first")
        return
    
    if not ae_path.exists():
        print("ERROR: No trained autoencoder found!")
        print("Run 06_train_autoencoder.py first")
        return
    
    print(f"Loading models...")
    print(f"  CNN: {cnn_path.name}")
    print(f"  Autoencoder: {ae_path.name}\n")
    
    # Load models
    cnn_model, device = load_model(cnn_path, model_type='cnn')
    ae_model, _ = load_model(ae_path, model_type='autoencoder')
    
    print(f"Using device: {device}\n")
    
    # Extract features from each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = processed_dir / split
        
        if not split_dir.exists():
            print(f"⚠️  Skipping {split}: folder not found")
            continue
        
        print(f"\nEXTRACTING {split.upper()} FEATURES")
        # Extract features
        cnn_features, ae_features, labels, filenames = extract_features_from_directory(
            split_dir, cnn_model, ae_model, device
        )
        
        # Combine features
        combined_features = np.concatenate([cnn_features, ae_features], axis=1)
        
        # Print summary
        print(f"\n{split.upper()} Summary:")
        print(f"  Total frames: {len(labels)}")
        print(f"  CNN features: {cnn_features.shape}")
        print(f"  Autoencoder features: {ae_features.shape}")
        print(f"  Combined features: {combined_features.shape}")
        
        # Count per condition
        for label_id, condition in enumerate(['anesthetic', 'stimulant', 'control']):
            count = np.sum(labels == label_id)
            if count > 0:
                print(f"    {condition}: {count} frames")
        
        # Save features
        np.save(features_dir / f'{split}_cnn_features.npy', cnn_features)
        np.save(features_dir / f'{split}_ae_features.npy', ae_features)
        np.save(features_dir / f'{split}_combined_features.npy', combined_features)
        np.save(features_dir / f'{split}_labels.npy', labels)
        
        # Save filenames
        with open(features_dir / f'{split}_filenames.txt', 'w') as f:
            f.write('\n'.join(filenames))
        
        print(f"\n Features saved to {features_dir}/")
        print()
    
    # Save metadata
    metadata = {
        'cnn_model': str(cnn_path.name),
        'autoencoder_model': str(ae_path.name),
        'cnn_feature_dim': int(cnn_features.shape[1]),
        'ae_feature_dim': int(ae_features.shape[1]),
        'combined_feature_dim': int(combined_features.shape[1]),
        'conditions': ['anesthetic', 'stimulant', 'control'],
        'label_mapping': {
            'anesthetic': 0,
            'stimulant': 1,
            'control': 2
        }
    }
    
    with open(features_dir / 'feature_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("FEATURE EXTRACTION COMPLETE")

if __name__ == "__main__":
    main()
