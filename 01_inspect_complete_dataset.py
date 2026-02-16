import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_condition(condition_dir):
    condition_dir = Path(condition_dir)
    tif_files = sorted(condition_dir.glob('*.tif'))
    
    if len(tif_files) == 0:
        print(f"⚠️  No .tif files found in {condition_dir}")
        return None
    
    results = []
    for filepath in tif_files:
        try:
            with tifffile.TiffFile(filepath) as tif:
                stack = tif.asarray()
                first_page = tif.pages[0]
                
                info = {
                    'filename': filepath.name,
                    'condition': condition_dir.name,
                    'num_frames': len(tif.pages),
                    'height': stack.shape[-2],
                    'width': stack.shape[-1],
                    'dtype': str(stack.dtype),
                    'min_value': int(stack.min()),
                    'max_value': int(stack.max()),
                    'mean_value': float(stack.mean()),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024)
                }
                
                # Calculate motion
                if len(stack.shape) >= 3 and stack.shape[0] > 1:
                    diffs = []
                    for i in range(min(10, stack.shape[0] - 1)):
                        diff = np.abs(stack[i+1].astype(float) - stack[i].astype(float)).mean()
                        diffs.append(diff)
                    info['avg_frame_diff'] = float(np.mean(diffs))
                else:
                    info['avg_frame_diff'] = 0.0
                
                results.append(info)
        except Exception as e:
            print(f"Error reading {filepath.name}: {e}")
    
    return pd.DataFrame(results) if results else None

def main():
    print("ZEBRAFISH DATASET COMPLETE INSPECTION")
    
    # Project paths
    project_dir = Path('.')
    data_dir = project_dir / 'data' / 'raw'
    metadata_dir = project_dir / 'data' / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Conditions to check
    conditions = ['anesthetic', 'stimulant', 'control']
    
    all_data = []
    summary = {}
    
    # Inspect each condition
    for condition in conditions:
        condition_path = data_dir / condition
        print(f"INSPECTING: {condition.upper()}")
        
        if not condition_path.exists():
            print(f" Folder not found: {condition_path}")
            print(f"   Create it with: mkdir -p {condition_path}")
            summary[condition] = {'status': 'missing', 'files': 0, 'frames': 0}
            continue
        
        df = inspect_condition(condition_path)
        
        if df is None or len(df) == 0:
            print(f" No .tif files in {condition}")
            summary[condition] = {'status': 'empty', 'files': 0, 'frames': 0}
            continue
        
        # Save individual condition metadata
        csv_path = metadata_dir / f'{condition}_inspection.csv'
        df.to_csv(csv_path, index=False)
        
        # Print summary
        print(f"\nFiles found: {len(df)}")
        print(f"   Total frames: {df['num_frames'].sum()}")
        print(f"   Frames per file: {df['num_frames'].min()}-{df['num_frames'].max()} (avg: {df['num_frames'].mean():.1f})")
        print(f"   Resolution: {df['height'].iloc[0]} × {df['width'].iloc[0]}")
        print(f"   Data type: {df['dtype'].iloc[0]}")
        print(f"   Value range: [{df['min_value'].min()}, {df['max_value'].max()}]")
        print(f"   Avg motion: {df['avg_frame_diff'].mean():.2f}")
        print(f"   Total size: {df['file_size_mb'].sum():.1f} MB")
        print(f"   Metadata saved: {csv_path}")
        
        summary[condition] = {
            'status': 'complete',
            'files': len(df),
            'frames': int(df['num_frames'].sum()),
            'size_mb': float(df['file_size_mb'].sum())
        }
        
        all_data.append(df)
    
    # Combined dataset summary

    print("COMPLETE DATASET SUMMARY")
    
    total_files = sum(s['files'] for s in summary.values())
    total_frames = sum(s['frames'] for s in summary.values())
    total_size = sum(s.get('size_mb', 0) for s in summary.values())
    
    print(f"{'Condition':<15} {'Status':<12} {'Files':<8} {'Frames':<10} {'Size (MB)':<12}")
    for condition in conditions:
        s = summary[condition]
        status_icon = "done" if s['status'] == 'complete' else "not done"
        print(f"{condition:<15} {status_icon} {s['status']:<10} {s['files']:<8} {s['frames']:<10} {s.get('size_mb', 0):<12.1f}")
    print(f"{'TOTAL':<15} {'':<12} {total_files:<8} {total_frames:<10} {total_size:<12.1f}")
    
    # Save combined metadata
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv = metadata_dir / 'complete_dataset.csv'
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n Combined metadata saved: {combined_csv}")
    
    # Check if dataset is ready
    print("DATASET STATUS")    
    complete_conditions = sum(1 for s in summary.values() if s['status'] == 'complete')
    
    if complete_conditions == 3:
        print("DATASET COMPLETE! All 3 conditions ready.")
        print(f" Total: {total_files} files, {total_frames} frames")
        print("\n Ready for next steps:")
        print("   1. Create train/val/test splits")
        print("   2. Train CNN and Autoencoder")
        print("   3. Extract features")
    elif complete_conditions > 0:
        print(f"DATASET INCOMPLETE: {complete_conditions}/3 conditions ready")
        print("\n Missing conditions:")
        for condition, s in summary.items():
            if s['status'] != 'complete':
                print(f"   • {condition}: Add .tif files to data/raw/{condition}/")
    else:
        print(" NO DATA FOUND")
        print("\n Instructions:")
        print("   1. Create folders: data/raw/anesthetic, data/raw/stimulant, data/raw/control")
        print("   2. Copy your .tif stack files to respective folders")
        print("   3. Run this script again")
    
    # Visualize distribution if data exists
    if all_data and len(all_data) > 0:
        visualize_dataset(combined_df)

def visualize_dataset(df):
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Frames per condition
    ax = axes[0, 0]
    condition_frames = df.groupby('condition')['num_frames'].sum()
    condition_frames.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title('Total Frames per Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Frames')
    ax.set_xlabel('Condition')
    ax.grid(axis='y', alpha=0.3)
    
    # Files per condition
    ax = axes[0, 1]
    condition_files = df.groupby('condition').size()
    condition_files.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title('Files per Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Files')
    ax.set_xlabel('Condition')
    ax.grid(axis='y', alpha=0.3)
    
    # Frame distribution
    ax = axes[1, 0]
    df['num_frames'].hist(bins=15, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Frames per File', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Frames')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)
    
    # Motion comparison
    ax = axes[1, 1]
    df.boxplot(column='avg_frame_diff', by='condition', ax=ax)
    ax.set_title('Motion Comparison Across Conditions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Frame Difference')
    ax.set_xlabel('Condition')
    plt.suptitle('') 
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path('results/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / 'dataset_overview.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {fig_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
