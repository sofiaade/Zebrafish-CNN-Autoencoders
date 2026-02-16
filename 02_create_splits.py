import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import json

def create_splits(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print(f"\nSplit ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"Random seed: {random_state}\n")
    
    # Collect all files by condition
    conditions = ['anesthetic', 'stimulant', 'control']
    condition_mapping = {'anesthetic': 0, 'stimulant': 1, 'control': 2}
    
    all_files = []
    all_labels = []
    all_conditions = []
    
    for condition in conditions:
        condition_path = data_dir / condition
        if not condition_path.exists():
            print(f"Skipping {condition}: folder not found")
            continue
        
        tif_files = sorted(condition_path.glob('*.tif'))
        if len(tif_files) == 0:
            print(f"Skipping {condition}: no .tif files found")
            continue
        
        print(f" {condition}: {len(tif_files)} files")
        
        all_files.extend(tif_files)
        all_labels.extend([condition_mapping[condition]] * len(tif_files))
        all_conditions.extend([condition] * len(tif_files))
    
    if len(all_files) == 0:
        print("\n ERROR: No files found! Check your data/raw/ folder structure.")
        return
    
    print(f"\nTotal files: {len(all_files)}")
    
    # Convert to arrays
    files_array = np.array(all_files)
    labels_array = np.array(all_labels)
    conditions_array = np.array(all_conditions)
    
    # Stratified split to maintain class balance
    # First split: train vs (val+test)
    train_files, temp_files, train_labels, temp_labels, train_conditions, temp_conditions = train_test_split(
        files_array, labels_array, conditions_array,
        test_size=(val_ratio + test_ratio),
        stratify=labels_array,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels, val_conditions, test_conditions = train_test_split(
        temp_files, temp_labels, temp_conditions,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Print split summary
    print("SPLIT SUMMARY")
    
    splits = {
        'train': (train_files, train_labels, train_conditions),
        'val': (val_files, val_labels, val_conditions),
        'test': (test_files, test_labels, test_conditions)
    }
    
    for split_name, (files, labels, conds) in splits.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total files: {len(files)}")
        for condition in conditions:
            count = np.sum(conds == condition)
            if count > 0:
                print(f"    {condition}: {count} files")
    
    # Create directories and copy files
    print("COPYING FILES") 
    split_info = {}
    
    for split_name, (files, labels, conds) in splits.items():
        split_dir = output_dir / split_name
        
        # Create directories for each condition
        for condition in conditions:
            condition_dir = split_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for file_path, label, condition in zip(files, labels, conds):
            dest_dir = split_dir / condition
            dest_path = dest_dir / file_path.name
            
            # Copy file
            shutil.copy2(file_path, dest_path)
        
        print(f" {split_name}: Copied {len(files)} files to {split_dir}")
        
        # Save split info
        split_info[split_name] = {
            'files': [str(f) for f in files],
            'labels': labels.tolist(),
            'conditions': conds.tolist()
        }
    
    # Save split metadata
    metadata_dir = output_dir.parent / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    split_json = metadata_dir / 'data_splits.json'
    with open(split_json, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\n Split metadata saved: {split_json}")
    
    # Save as CSV for easy viewing
    split_records = []
    for split_name, (files, labels, conds) in splits.items():
        for f, l, c in zip(files, labels, conds):
            split_records.append({
                'split': split_name,
                'filename': f.name,
                'condition': c,
                'label': l,
                'original_path': str(f)
            })
    
    split_df = pd.DataFrame(split_records)
    split_csv = metadata_dir / 'data_splits.csv'
    split_df.to_csv(split_csv, index=False)
    print(f"Split CSV saved: {split_csv}")
    
    # Print final summary
    print("SPLITS CREATED SUCCESSFULLY")

def main():
    # Paths
    project_dir = Path('.')
    raw_dir = project_dir / 'data' / 'raw'
    processed_dir = project_dir / 'data' / 'processed'
    
    # Check if raw data exists
    if not raw_dir.exists():
        print(f" ERROR: {raw_dir} not found!")
        print("Please run 01_inspect_complete_dataset.py first")
        return
    
    # Create splits
    create_splits(
        data_dir=raw_dir,
        output_dir=processed_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

if __name__ == "__main__":
    main()
