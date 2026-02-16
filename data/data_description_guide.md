# Data Description Guide

## Overview

This guide provides a complete description of the zebrafish behavioral analysis dataset, including file formats, structure, conditions, and key characteristics.

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Total files | 30 TIFF stacks |
| Conditions | 3 (Anesthetic, Stimulant, Control) |
| Files per condition | 10 |
| Total frames | Approximately 1,200 |
| Image resolution | 512 x 512 pixels |
| Bit depth | 16-bit grayscale |
| File format | Multi-page TIFF (.tif) |
| Image type | Dorsal-view fluorescence microscopy |

---

## Directory Structure

### Raw Data Location

```
data/raw/
├── anesthetic/     # 10 .tif files - anesthetic condition
├── stimulant/      # 10 .tif files - stimulant condition
└── control/        # 10 .tif files - control condition
```

### Processed Data Location

After running `02_create_splits.py`, data is organized into train/validation/test splits:

```
data/processed/
├── train/
│   ├── anesthetic/
│   ├── stimulant/
│   └── control/
├── val/
│   ├── anesthetic/
│   ├── stimulant/
│   └── control/
└── test/
    ├── anesthetic/
    ├── stimulant/
    └── control/
```

---

## Experimental Conditions

### 1. Anesthetic Condition

**Description:** Zebrafish larvae exposed to anesthetic agents that suppress motor activity and neural signaling.

**Expected Behavioral Characteristics:**
- Reduced movement
- Suppressed neural activity
- Characteristic postural changes
- Reduced fluorescent signal dynamics
- Lowest frame-to-frame motion (mean: 10.6 pixel difference)

**Typical Use:** Models quiescent/sedated state as a baseline for pharmacological studies.

---

### 2. Stimulant Condition

**Description:** Zebrafish larvae exposed to stimulant compounds that elevate activity levels above baseline.

**Expected Behavioral Characteristics:**
- Increased movement frequency
- Elevated neural activity
- Higher frame-to-frame variation
- Most dynamic fluorescent patterns
- Highest frame-to-frame motion (mean: 22.3 pixel difference)

**Typical Use:** Models hyperactive state for contrast with control and anesthetic conditions.

---

### 3. Control Condition

**Description:** Zebrafish larvae under normal, undisturbed conditions without pharmacological intervention.

**Expected Behavioral Characteristics:**
- Baseline motor activity
- Normal neural signaling patterns
- Intermediate movement levels
- Moderate frame-to-frame motion (mean: 16.1 pixel difference)

**Typical Use:** Baseline reference for comparing pharmacological effects.

---

## File Format Specifications

### TIFF Stack Structure

Each `.tif` file is a **multi-page TIFF stack** containing multiple sequential frames:

```
filename.tif
├── Frame 0   (512 x 512, 16-bit)
├── Frame 1   (512 x 512, 16-bit)
├── Frame 2   (512 x 512, 16-bit)
...
└── Frame N   (30-70 frames typical)
```

### Reading TIFF Files

**Python with tifffile:**
```python
import tifffile

# Load entire stack
stack = tifffile.imread('filename.tif')  # Shape: (n_frames, 512, 512)

# Load single frame
frame = stack[0]  # Shape: (512, 512)
```

### Image Characteristics

| Property | Value |
|----------|-------|
| Width | 512 pixels |
| Height | 512 pixels |
| Channels | 1 (grayscale) |
| Bit depth | 16-bit unsigned integer |
| Value range | 0 to 65,535 |
| Typical actual range | 200 to 15,000 (varies by frame) |

---

## Data Statistics

### Per-Condition Frame Counts

After train/validation/test split (70/15/15):

| Condition | Total Frames | Train (~70%) | Val (~15%) | Test (~15%) |
|-----------|--------------|--------------|------------|-------------|
| Anesthetic | ~400 | ~280 | ~60 | ~60 |
| Stimulant | ~400 | ~280 | ~60 | ~60 |
| Control | ~400 | ~280 | ~60 | ~60 |
| **Total** | **~1,200** | **~840** | **~180** | **~180** |

Note: Exact counts vary slightly based on the number of frames per file.

### Motion Statistics

Frame-to-frame pixel difference (mean absolute difference between consecutive frames):

| Condition | Mean Motion | Interpretation |
|-----------|-------------|----------------|
| Anesthetic | 10.6 | Low activity (sedated) |
| Control | 16.1 | Baseline activity |
| Stimulant | 22.3 | High activity (stimulated) |

These statistics validate expected pharmacological effects and confirm behavioral relevance of the dataset.

---

## Image Content

### What the Images Show

- **View:** Dorsal (top-down) view of zebrafish larvae
- **Subject:** Individual zebrafish larva per stack
- **Imaging:** Fluorescence microscopy
- **Background:** Dark (low-intensity pixels)
- **Foreground:** Zebrafish body (high-intensity fluorescent signal)
- **Key Features:** Body outline, yolk sac, head region, trunk

### Spatial Features

The images capture:
1. **Body structure:** Overall shape and posture of the fish
2. **Fluorescent expression patterns:** Spatial distribution of fluorescent markers
3. **Anatomical landmarks:** Head, yolk sac, tail
4. **Postural variation:** Different body positions across frames

### Temporal Features (3D CNN)

Sequential frames capture:
1. **Movement patterns:** How the fish moves over time
2. **Activity frequency:** Rate of position changes
3. **Motion amplitude:** Magnitude of movements
4. **Behavioral dynamics:** Temporal evolution of posture and position

---

## Data Quality

### Normalization Requirements

Raw pixel values vary across frames and files. Before model input, frames are normalized:

```python
# Per-frame min-max normalization
frame_normalized = (frame - frame.min()) / (frame.max() - frame.min())
# Result: values in [0, 1]
```

### Data Integrity Checks

Run `01_inspect_complete_dataset.py` to verify:
- All files load correctly
- Frame counts are reasonable (30-70 per file)
- Image dimensions are consistent (512 x 512)
- Pixel value ranges are valid
- Motion statistics match expected patterns

---

## Split Strategy

### Train/Validation/Test Split

**Rationale:** 70/15/15 split balances training data availability with robust validation and testing.

**Method:** Stratified sampling at the FILE level (not frame level) to prevent data leakage.

**Why file-level splitting matters:**
- Frames from the same stack are highly correlated
- Splitting at frame level would leak information from train to test
- File-level split ensures independent evaluation

**Implementation:**
```python
# In 02_create_splits.py
# Shuffle files (not frames), then split 70/15/15
# All frames from a given file go to the same split
```

---

## Label Assignment

### Frame-Level Labels

For 2D CNN and Autoencoder (frame-level models):
- Each frame receives a label based on its condition
- Anesthetic = 0
- Stimulant = 1
- Control = 2

### Sequence-Level Labels (3D CNN)

For 3D CNN (sequence-level model):
- Each 16-frame sequence receives a label
- Label = majority vote of constituent frame labels
- Example: If 12/16 frames are "stimulant", sequence label = "stimulant"

---

## Data Characteristics for Modeling

### Challenges

1. **Small dataset:** Only ~400 frames per condition after splitting
2. **Subtle differences:** Behavioral changes are visible but not dramatic
3. **High resolution:** 512 x 512 images require substantial model capacity
4. **Temporal dynamics:** Motion patterns require sequence modeling for full capture
5. **Individual variation:** Different fish exhibit different baseline behaviors

### Opportunities

1. **Balanced classes:** Equal representation across all three conditions
2. **High-quality imaging:** Clear fluorescent signal, minimal noise
3. **Validated differences:** Motion statistics confirm expected pharmacological effects
4. **Multi-modal learning:** Can combine spatial (2D) and temporal (3D) features
5. **Interpretable features:** Saliency maps show model attends to fish body, not artifacts

---

## Data Augmentation Considerations

To address the small dataset size, consider these augmentation strategies:

**Geometric transformations:**
- Horizontal flip (fish orientation can vary)
- Vertical flip
- Rotation (90, 180, 270 degrees)

**Photometric transformations:**
- Brightness adjustment (±20%)
- Contrast adjustment (±20%)
- Gaussian noise (small amount)

**Not recommended:**
- Crop/zoom (may lose important anatomical features)
- Extreme color jitter (images are grayscale)
- Elastic deformation (may distort biologically relevant features)

See `IMPROVE_ACCURACY_GUIDE.md` for implementation details.

---

## Expected Model Performance

### Baseline (Random Guess)

With 3 balanced classes: 33.3% accuracy

### Reasonable Performance Targets

| Model Type | Expected Accuracy | Notes |
|------------|-------------------|-------|
| Frame-level (2D CNN) | 55-75% | Limited by lack of temporal context |
| Sequence-level (3D CNN) | 65-85% | Benefits from motion information |
| Combined features | 60-80% | Ensemble of spatial + temporal |

### Why Perfect Accuracy is Unrealistic

1. **Overlapping distributions:** Some frames from different conditions look similar
2. **Individual variation:** Not all fish respond identically to treatments
3. **Frame-level ambiguity:** Single frames may not capture full behavioral state
4. **Subtle pharmacological effects:** Differences are real but not always visually obvious

---

## Data Citation and Usage

If using this dataset for publication:

**Dataset Description to Include:**
"The dataset consists of 30 multi-page TIFF stacks (10 per condition) of zebrafish larvae under three experimental conditions: anesthetic, stimulant, and control. Each stack contains 30-70 sequential 512x512 16-bit grayscale frames captured via dorsal-view fluorescence microscopy, yielding approximately 1,200 total frames across all conditions."

**Metadata to Report:**
- Number of files per condition
- Total frame count
- Image resolution and bit depth
- Train/validation/test split ratios
- Motion statistics per condition

---

## Troubleshooting Data Issues

### Common Problems

**Problem:** `tifffile` can't read a file
**Solution:** Check file is not corrupted, verify it's a valid TIFF format

**Problem:** Frames have unexpected dimensions
**Solution:** Some files may be single-page (2D array) instead of multi-page (3D array)
```python
if len(stack.shape) == 2:  # Single frame
    stack = stack[np.newaxis, ...]  # Add frame dimension
```

**Problem:** Class imbalance after splitting
**Solution:** Use stratified sampling in `02_create_splits.py` to maintain balance

**Problem:** Memory errors during batch processing
**Solution:** Reduce batch size or process fewer frames at once

---

## Data Versioning

**Current Version:** v1.0

**Change Log:**
- v1.0 (Initial): 30 files, 3 conditions, ~1,200 frames

If you update the dataset (add files, fix corrupted data, etc.), increment the version and document changes here.

---

## Next Steps

After understanding the data structure:

1. **Inspect the data:** Run `01_inspect_complete_dataset.py`
2. **Create splits:** Run `02_create_splits.py`
3. **Verify splits:** Check `data/metadata/data_splits.csv`
4. **Train models:** Proceed with scripts 03-06
5. **Extract features:** Run scripts 07/07b
6. **Analyze results:** Run scripts 08-12

See `COMPLETE_WORKFLOW_GUIDE.md` for step-by-step instructions.
