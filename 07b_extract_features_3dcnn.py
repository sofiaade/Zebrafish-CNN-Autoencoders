import torch
import torch.nn as nn
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import json

# MODEL ARCHITECTURE
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
        return self.classifier(self.extract_features(x))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Layer names must match the actual trained model from 06_train_autoencoder.py
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.fc_encode = nn.Linear(512 * 16 * 16, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 512 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_encode(x)

    def forward(self, x):
        z = self.encode(x)
        x = self.fc_decode(z)
        x = x.view(x.size(0), 512, 16, 16)
        return self.decoder(x)


class ZebrafishCNN3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, feature_dim=512):
        super(ZebrafishCNN3D, self).__init__()

        # Architecture reverse-engineered from checkpoint keys:
        # features.0  = Conv3d(1->32)      [3,3,3]
        # features.1  = BatchNorm3d(32)
        # features.2  = ReLU
        # features.3  = MaxPool3d
        # features.4  = Conv3d(32->64)     [3,3,3]
        # features.5  = BatchNorm3d(64)
        # features.6  = ReLU
        # features.7  = MaxPool3d
        # features.8  = Conv3d(64->128)    [3,3,3]
        # features.9  = BatchNorm3d(128)
        # features.10 = ReLU
        # features.11 = MaxPool3d
        # features.12 = Conv3d(128->256)   [3,3,3]
        # features.13 = BatchNorm3d(256)
        # features.14 = ReLU
        # features.15 = AdaptiveAvgPool3d  (no weights)
        # fc.1        = Linear(8192->512)  i.e. 256*4*4*... wait:
        #   256 channels * AdaptiveAvgPool3d output -> 8192
        #   8192 / 256 = 32, so pool output = (?,4,4) with temporal dim kept
        #   actual pool = AdaptiveAvgPool3d((2,4,4)) -> 256*2*4*4=8192

        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3,3,3), padding=(1,1,1)),  # 0
            nn.BatchNorm3d(32),                                                    # 1
            nn.ReLU(inplace=True),                                                 # 2
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),                    # 3
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),              # 4
            nn.BatchNorm3d(64),                                                    # 5
            nn.ReLU(inplace=True),                                                 # 6
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),                    # 7
            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),             # 8
            nn.BatchNorm3d(128),                                                   # 9
            nn.ReLU(inplace=True),                                                 # 10
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),                    # 11
            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1)),            # 12
            nn.BatchNorm3d(256),                                                   # 13
            nn.ReLU(inplace=True),                                                 # 14
            nn.AdaptiveAvgPool3d((2, 4, 4))                                        # 15 -> 256*2*4*4=8192
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                              # 0
            nn.Linear(256 * 2 * 4 * 4, feature_dim),  # 1  8192->512
            nn.ReLU(inplace=True),                     # 2
            nn.Dropout(0.5)                            # 3
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def extract_features(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self.classifier(self.extract_features(x))

# DATA LOADING

def load_frames_from_condition(condition_dir):
    frames = []
    filenames = []

    tif_files = sorted(Path(condition_dir).glob('*.tif'))

    for tif_file in tif_files:
        try:
            stack = tifffile.imread(tif_file)
            if len(stack.shape) == 3:
                for i, frame in enumerate(stack):
                    frames.append(frame)
                    filenames.append(f"{tif_file.name}_frame{i:03d}")
            elif len(stack.shape) == 2:
                frames.append(stack)
                filenames.append(f"{tif_file.name}_frame000")
        except Exception as e:
            print(f"  Error loading {tif_file.name}: {e}")

    return frames, filenames


def normalize_frame(frame):
    frame = frame.astype(np.float32)
    mn, mx = frame.min(), frame.max()
    return (frame - mn) / (mx - mn + 1e-8)


def frames_to_sequences(frames, labels, filenames, seq_len=16, stride=8):
    sequences = []
    seq_labels = []
    seq_filenames = []

    i = 0
    while i + seq_len <= len(frames):
        seq = frames[i : i + seq_len]
        lbl_window = labels[i : i + seq_len]

        # Majority vote label
        majority = int(np.bincount(lbl_window).argmax())

        sequences.append(np.array(seq))
        seq_labels.append(majority)
        seq_filenames.append(f"{filenames[i]}_seq{i:04d}")

        i += stride

    return sequences, seq_labels, seq_filenames


# FEATURE EXTRACTION FUNCTIONS
def extract_2d_features(model, frames, device, batch_size=16):
    model.eval()
    all_features = []

    for i in tqdm(range(0, len(frames), batch_size), desc="  2D CNN", leave=False):
        batch_frames = frames[i : i + batch_size]
        batch = np.stack([
            np.expand_dims(normalize_frame(f), 0) for f in batch_frames
        ])
        tensor = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            features = model.extract_features(tensor)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def extract_ae_features(model, frames, device, batch_size=16):
    model.eval()
    all_features = []

    for i in tqdm(range(0, len(frames), batch_size), desc="  Autoencoder", leave=False):
        batch_frames = frames[i : i + batch_size]
        batch = np.stack([
            np.expand_dims(normalize_frame(f), 0) for f in batch_frames
        ])
        tensor = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            features = model.encode(tensor)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def extract_3d_features(model, sequences, device, batch_size=8):
    model.eval()
    all_features = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="  3D CNN", leave=False):
        batch_seqs = sequences[i : i + batch_size]

        # Normalize each frame in each sequence
        batch = np.stack([
            np.stack([normalize_frame(f) for f in seq])
            for seq in batch_seqs
        ])  # (batch, seq_len, H, W)

        # Add channel dim: (batch, 1, seq_len, H, W)
        batch = np.expand_dims(batch, axis=1)

        tensor = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            features = model.extract_features(tensor)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)

def main():
    print("=" * 80)
    print("3D CNN FEATURE EXTRACTION")

    # Paths
    project_dir = Path('.')
    processed_dir = project_dir / 'data' / 'processed'
    models_dir = project_dir / 'results' / 'models'
    features_dir = project_dir / 'results' / 'features_3d'
    features_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    condition_to_label = {'anesthetic': 0, 'stimulant': 1, 'control': 2}
    SEQ_LEN = 16
    STRIDE = 8

    # Load Models
    print("Loading trained models...")

    # 2D CNN
    cnn2d_path = models_dir / 'cnn2d_best.pth'
    if not cnn2d_path.exists():
        print(f"  WARNING: {cnn2d_path} not found - skipping 2D CNN features")
        cnn2d_model = None
    else:
        ckpt = torch.load(cnn2d_path, map_location=device)
        cfg = ckpt.get('config', {})
        cnn2d_model = ZebrafishCNN2D(
            num_classes=cfg.get('num_classes', 3),
            feature_dim=cfg.get('feature_dim', 512)
        ).to(device)
        cnn2d_model.load_state_dict(ckpt['model_state_dict'])
        cnn2d_model.eval()
        print(f"  2D CNN loaded  ({cnn2d_model.feature_dim}D features)")

    # Autoencoder
    ae_path = models_dir / 'autoencoder_best.pth'
    if not ae_path.exists():
        print(f"  WARNING: {ae_path} not found - skipping autoencoder features")
        ae_model = None
    else:
        ckpt = torch.load(ae_path, map_location=device)
        cfg = ckpt.get('config', {})
        ae_model = ConvAutoencoder(
            latent_dim=cfg.get('latent_dim', 128)
        ).to(device)
        ae_model.load_state_dict(ckpt['model_state_dict'])
        ae_model.eval()
        print(f"  Autoencoder loaded ({ae_model.latent_dim}D features)")

    # 3D CNN
    cnn3d_path = models_dir / 'cnn3d_best.pth'
    if not cnn3d_path.exists():
        print(f"  WARNING: {cnn3d_path} not found - skipping 3D CNN features")
        cnn3d_model = None
    else:
        ckpt = torch.load(cnn3d_path, map_location=device)
        cfg = ckpt.get('config', {})
        cnn3d_model = ZebrafishCNN3D(
            num_classes=cfg.get('num_classes', 3),
            feature_dim=cfg.get('feature_dim', 512)
        ).to(device)
        cnn3d_model.load_state_dict(ckpt['model_state_dict'])
        cnn3d_model.eval()
        print(f"  3D CNN loaded  ({cnn3d_model.feature_dim}D features)")

    print()

    # Process each split
    metadata = {}

    for split in ['train', 'val', 'test']:
        print("=" * 80)
        print(f"PROCESSING SPLIT: {split.upper()}")
        print("=" * 80 + "\n")

        split_dir = processed_dir / split

        # --- Load all frames and labels ---
        all_frames, all_labels, all_filenames = [], [], []

        for condition, label in condition_to_label.items():
            cond_dir = split_dir / condition
            if not cond_dir.exists():
                continue
            frames, filenames = load_frames_from_condition(cond_dir)
            labels = [label] * len(frames)
            all_frames.extend(frames)
            all_labels.extend(labels)
            all_filenames.extend(filenames)
            print(f"  {condition}: {len(frames)} frames")

        all_labels = np.array(all_labels)
        print(f"  Total frames: {len(all_frames)}\n")

        # --- 2D CNN features (frame-level) ---
        if cnn2d_model is not None:
            print("Extracting 2D CNN features...")
            cnn2d_features = extract_2d_features(cnn2d_model, all_frames, device)
            np.save(features_dir / f'{split}_cnn2d_features.npy', cnn2d_features)
            print(f"  Saved: {split}_cnn2d_features.npy  shape={cnn2d_features.shape}")

        # --- Autoencoder features (frame-level) ---
        if ae_model is not None:
            print("Extracting Autoencoder features...")
            ae_features = extract_ae_features(ae_model, all_frames, device)
            np.save(features_dir / f'{split}_ae_features.npy', ae_features)
            print(f"  Saved: {split}_ae_features.npy  shape={ae_features.shape}")

        # --- Combined 2D frame-level features ---
        if cnn2d_model is not None and ae_model is not None:
            combined_2d = np.concatenate([cnn2d_features, ae_features], axis=1)
            np.save(features_dir / f'{split}_combined_2d_features.npy', combined_2d)
            print(f"  Saved: {split}_combined_2d_features.npy  shape={combined_2d.shape}")

        # --- Save frame-level labels ---
        np.save(features_dir / f'{split}_frame_labels.npy', all_labels)
        np.savetxt(features_dir / f'{split}_frame_filenames.txt',
                   all_filenames, fmt='%s')

        # 3D CNN: Build sequences THEN extract features
        if cnn3d_model is not None:
            print(f"\nBuilding sequences (len={SEQ_LEN}, stride={STRIDE})...")

            # Build sequences per-condition to avoid mixing stacks
            seq_frames_all, seq_labels_all, seq_filenames_all = [], [], []

            for condition, label in condition_to_label.items():
                cond_dir = split_dir / condition
                if not cond_dir.exists():
                    continue

                cond_frames, cond_filenames = load_frames_from_condition(cond_dir)
                cond_labels = [label] * len(cond_frames)

                seqs, slbls, sfnames = frames_to_sequences(
                    cond_frames, cond_labels, cond_filenames,
                    seq_len=SEQ_LEN, stride=STRIDE
                )
                seq_frames_all.extend(seqs)
                seq_labels_all.extend(slbls)
                seq_filenames_all.extend(sfnames)
                print(f"  {condition}: {len(seqs)} sequences")

            seq_labels_np = np.array(seq_labels_all)
            print(f"  Total sequences: {len(seq_frames_all)}\n")

            print("Extracting 3D CNN features...")
            cnn3d_features = extract_3d_features(cnn3d_model, seq_frames_all, device)
            np.save(features_dir / f'{split}_cnn3d_features.npy', cnn3d_features)
            np.save(features_dir / f'{split}_seq_labels.npy', seq_labels_np)
            np.savetxt(features_dir / f'{split}_seq_filenames.txt',
                       seq_filenames_all, fmt='%s')
            print(f"  Saved: {split}_cnn3d_features.npy  shape={cnn3d_features.shape}")

        # --- Metadata ---
        metadata[split] = {
            'n_frames': len(all_frames),
            'n_sequences': len(seq_frames_all) if cnn3d_model else 0,
            'seq_len': SEQ_LEN,
            'stride': STRIDE,
            'cnn2d_dim': int(cnn2d_features.shape[1]) if cnn2d_model else None,
            'ae_dim': int(ae_features.shape[1]) if ae_model else None,
            'cnn3d_dim': int(cnn3d_features.shape[1]) if cnn3d_model else None,
        }
        print()

    # Save metadata
    with open(features_dir / 'feature_metadata_3d.json', 'w') as f:
        json.dump({
            'splits': metadata,
            'feature_dimensions': {
                'cnn2d': 512,
                'autoencoder': 128,
                'cnn3d': 256,
                'combined_2d': 640,
                'combined_all': 896
            },
            'granularity': {
                'cnn2d': 'frame-level',
                'autoencoder': 'frame-level',
                'cnn3d': 'sequence-level  (16 frames)',
                'combined_2d': 'frame-level',
                'combined_all': 'sequence-level'
            }
        }, f, indent=2)

    # Summary
    print("EXTRACTION COMPLETE")
    print(f"\nAll features saved in: {features_dir}/\n")
    print("Feature files per split:")
    print("  Frame-level (can use directly with classifiers):")
    print("    {split}_cnn2d_features.npy      512D")
    print("    {split}_ae_features.npy          128D")
    print("    {split}_combined_2d_features.npy 640D")
    print("    {split}_frame_labels.npy")
    print()
    print("  Sequence-level (richer, captures motion):")
    print("    {split}_cnn3d_features.npy       256D")
    print("    {split}_seq_labels.npy")
    print()
    print("Key difference:")
    print("  Frame-level: 1 feature vector per frame")
    print("  Sequence-level: 1 feature vector per 16-frame window")
    print("  -> Sequence features encode HOW the fish moves, not just what it looks like")
    print()
    print("Next step: run 08b_analyze_features_3d.py to compare all feature types")


if __name__ == "__main__":
    main()