# Zebrafish Behavioral Analysis - Complete Project README

##  **Project Overview**

This project uses deep learning (CNN + Autoencoder) to automatically classify zebrafish behavioral responses across three experimental conditions:
- **Anesthetic** - Zebrafish under anesthetic treatment
- **Stimulant** - Zebrafish under stimulant treatment  
- **Control** - Zebrafish under normal conditions

**Goal:** Automated classification of zebrafish behavior from video frames using extracted deep learning features.

---

##  **Project Structure**

```
zebrafish-cnn-autoencoder-project/
├── data/
│   ├── raw/                      # Original .tif files
│   │   ├── anesthetic/          # Anesthetic condition files
│   │   ├── stimulant/           # Stimulant condition files
│   │   └── control/             # Control condition files
│   ├── processed/               # Train/val/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/                # Dataset statistics and split info
│
├── results/
│   ├── models/                  # Trained model weights (.pth files)
│   ├── features/                # Extracted features (.npy files)
│   ├── figures/                 # All visualizations and plots
│   ├── test_evaluation/         # Test set results
│   ├── visualization/           # Saliency maps, filters
│   └── statistics/              # Statistical analysis results
│
└── [Python scripts 01-12]       # Analysis pipeline scripts
```

---

##  **Python Scripts - What Each One Does**

### **Data Preparation Scripts**

#### **01_inspect_complete_dataset.py**
**Purpose:** Analyze and validate your raw data
**What it does:**
- Scans all .tif files in `data/raw/`
- Counts frames per condition
- Checks image dimensions and quality
- Calculates motion statistics
- Generates dataset overview plots

**Output:**
- `data/metadata/complete_dataset.csv` - Dataset statistics
- `results/figures/dataset_overview.png` - Visual summary

**Run when:** First step, before any training

---

#### **02_create_splits.py**
**Purpose:** Split data into train/validation/test sets
**What it does:**
- Creates 70% train, 15% validation, 15% test split
- Maintains class balance across splits
- Copies files to organized folders
- Saves split metadata

**Output:**
- `data/processed/train/` - Training data
- `data/processed/val/` - Validation data
- `data/processed/test/` - Test data
- `data/metadata/data_splits.csv` - Split information

**Run when:** After inspecting data, before training

---

### **Model Training Scripts**

#### **03_train_cnn.py**
**Purpose:** Train a simple baseline CNN
**What it does:**
- Trains SimpleCNN (~3M parameters)
- 4 convolutional blocks
- Frame-level classification
- Quick baseline to verify setup works

**Model Architecture:**
```
SimpleCNN:
- Conv Block 1: 32 filters
- Conv Block 2: 64 filters  
- Conv Block 3: 128 filters
- Conv Block 4: 256 filters
- Fully Connected: 256 → 3 classes
```

**Output:**
- `results/models/cnn_best.pth` - Best model weights
- `results/figures/cnn_training_curves.png` - Training plot

**Run when:** After creating splits, for quick baseline

**Expected:** 70-80% validation accuracy, ~10 minutes training

---

#### **04_train_advanced_cnn.py**  **MAIN CNN MODEL**
**Purpose:** Train the primary CNN for feature extraction
**What it does:**
- Trains ZebrafishCNN2D (~19M parameters)
- 5 convolutional blocks with BatchNorm
- Extracts 512-dimensional features
- Early stopping and learning rate scheduling

**Model Architecture:**
```
ZebrafishCNN2D:
Block 1: Input(1, 512, 512) → Conv(32) → Conv(32) → MaxPool → (32, 256, 256)
Block 2: Conv(64) → Conv(64) → MaxPool → (64, 128, 128)
Block 3: Conv(128) → Conv(128) → MaxPool → (128, 64, 64)
Block 4: Conv(256) → Conv(256) → MaxPool → (256, 32, 32)
Block 5: Conv(512) → AdaptiveAvgPool → (512, 8, 8)

Feature Extraction:
- Flatten: 512 × 8 × 8 = 32,768 dimensions
- FC Layer: 32,768 → 512 dimensions (feature vector)
- Dropout: 0.5
- Classifier: 512 → 3 classes

Total Parameters: ~19 million
Feature Dimension: 512D
```

**Output:**
- `results/models/cnn2d_best.pth` - Best model (use this for features!)
- `results/figures/cnn2d_training_curves.png`

**Run when:** Main training step

**Expected:** 75-90% validation accuracy, ~30 minutes training

---

#### **04b_train_cnn_with_augmentation.py**  **IMPROVED VERSION**
**Purpose:** Train CNN with data augmentation for better accuracy
**What it does:**
- Same ZebrafishCNN2D architecture as Script 04
- Adds data augmentation during training
- Augmentations: flips, rotations, brightness, contrast, noise
- Helps prevent overfitting on small datasets

**Augmentation Pipeline:**
```
For each training image (50% probability each):
✓ Horizontal flip
✓ Vertical flip
✓ Rotation (90°, 180°, 270°)
✓ Brightness adjustment (±20%)
✓ Contrast adjustment (±20%)
✓ Gaussian noise
```

**Output:**
- `results/models/cnn2d_augmented_best.pth` - Improved model
- `results/figures/cnn2d_augmented_training_curves.png`

**Run when:** To improve accuracy beyond baseline

**Expected:** 10-20% improvement over Script 04, ~30 minutes training

---

#### **05_train_3d_cnn.py** (Optional)
**Purpose:** Train 3D CNN for temporal/motion analysis
**What it does:**
- Processes 16-frame sequences (not individual frames)
- Captures motion patterns over time
- Good if behavioral differences are in movement

**Model Architecture:**
```
ZebrafishCNN3D:
- 3D Conv layers (captures time dimension)
- Input: (batch, 16 frames, 1, 512, 512)
- 4 3D convolutional blocks
- ~30M parameters
```

**Output:**
- `results/models/cnn3d_best.pth`

**Run when:** Only if temporal patterns matter more than spatial

**Expected:** 70-85% accuracy, ~60 minutes training (slow!)

---

#### **06_train_autoencoder.py**
**Purpose:** Train autoencoder for unsupervised feature learning
**What it does:**
- Learns to compress images into 128D latent space
- Unsupervised learning (no labels needed)
- Extracts different features than CNN
- Complements CNN features

**Model Architecture:**
```
ConvAutoencoder:

ENCODER:
Input: (1, 512, 512)
↓ Conv(32) + BN + ReLU → (32, 256, 256)
↓ Conv(64) + BN + ReLU → (64, 128, 128)
↓ Conv(128) + BN + ReLU → (128, 64, 64)
↓ Conv(256) + BN + ReLU → (256, 32, 32)
↓ Conv(512) + BN + ReLU → (512, 16, 16)
↓ Flatten → 131,072 dimensions
↓ FC → 128 dimensions (LATENT SPACE)

DECODER:
↓ FC → 131,072 dimensions
↓ Reshape → (512, 16, 16)
↓ ConvTranspose(256) → (256, 32, 32)
↓ ConvTranspose(128) → (128, 64, 64)
↓ ConvTranspose(64) → (64, 128, 128)
↓ ConvTranspose(32) → (32, 256, 256)
↓ ConvTranspose(1) → (1, 512, 512) - Reconstructed image

Total Parameters: ~37 million
Latent Dimension: 128D
Loss: MSE (reconstruction error)
```

**Output:**
- `results/models/autoencoder_best.pth` - Best autoencoder
- `results/figures/autoencoder_reconstructions.png` - Quality check

**Run when:** After CNN training, before feature extraction

**Expected:** Low reconstruction loss, ~45 minutes training

---

### **Feature Extraction & Analysis Scripts**

#### **07_extract_features.py**
**Purpose:** Extract features from trained models
**What it does:**
- Loads trained CNN and Autoencoder
- Processes all frames in train/val/test sets
- Extracts feature vectors for each frame
- Saves as .npy arrays for analysis

**Feature Extraction Process:**
```
For each frame:
1. CNN → 512D feature vector
2. Autoencoder → 128D latent vector
3. Combined → 640D feature vector (concatenate)

Saved files:
- train_cnn_features.npy: (N_train, 512)
- train_ae_features.npy: (N_train, 128)
- train_combined_features.npy: (N_train, 640)
- train_labels.npy: (N_train,)
(Same for val and test)
```

**Output:**
- `results/features/test_cnn_features.npy` - CNN features
- `results/features/test_ae_features.npy` - Autoencoder features
- `results/features/test_combined_features.npy` - Combined features
- `results/features/test_labels.npy` - Ground truth labels

**Run when:** After training both CNN and Autoencoder

**Expected:** ~10 minutes

---

#### **08_analyze_features.py**
**Purpose:** Comprehensive feature analysis
**What it does:**
- PCA and t-SNE visualization
- Clustering analysis (K-means, hierarchical)
- Classification with SVM, Random Forest, Logistic Regression
- Statistical tests (ANOVA)
- Feature importance analysis

**Analyses Performed:**
```
1. Dimensionality Reduction:
   - PCA (shows variance explained)
   - t-SNE (shows cluster separation)

2. Clustering:
   - K-means clustering
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)

3. Classification (on features):
   - SVM with RBF kernel
   - Random Forest (100 trees)
   - Logistic Regression
   - Reports: accuracy, F1, confusion matrix

4. Statistics:
   - ANOVA for each feature
   - Identifies significant features
```

**Output:**
- `results/figures/CNN_dimensionality_reduction.png`
- `results/figures/CNN_confusion_matrix.png`
- `results/figures/feature_type_comparison.png`

**Run when:** After feature extraction

**Expected:** ~5 minutes

---

### **Advanced Analysis Scripts**

#### **09_test_evaluation.py**
**Purpose:** Final test set evaluation (unbiased results)
**What it does:**
- Evaluates all feature types on held-out test set
- Generates ROC curves and AUC scores
- Creates Precision-Recall curves
- Produces publication-ready confusion matrices
- Compares all classifiers

**Metrics Reported:**
```
For each feature type (CNN, Autoencoder, Combined):
- Test Accuracy
- Precision, Recall, F1 Score
- ROC-AUC per class
- Average Precision per class
- Confusion matrices (raw counts + normalized)
```

**Output:**
- `results/test_evaluation/CNN_confusion_matrix_test.png`
- `results/test_evaluation/Combined_roc_curves_test.png`
- `results/test_evaluation/test_evaluation_summary.json`

**Run when:** After Script 08, for final results

**Expected:** ~5 minutes

---

#### **10_feature_visualization.py**
**Purpose:** Visualize what the CNN learned
**What it does:**
- Visualizes learned convolutional filters
- Generates activation maps from different layers
- Creates saliency maps (what the model focuses on)
- Validates biological relevance of learned features

**Visualizations Created:**
```
1. Learned Filters:
   - Shows 32 filters from first conv layer
   - Reveals edge/texture detectors learned

2. Activation Maps:
   - Shows network response at different layers
   - Early layers: simple features (edges)
   - Deep layers: complex patterns (fish body)

3. Saliency Maps:
   - Highlights important image regions
   - Shows if model focuses on fish vs background
   - Validates biological relevance
```

**Output:**
- `results/visualization/learned_filters.png`
- `results/visualization/saliency_map_anesthetic.png`
- `results/visualization/anesthetic_Conv_2_activations.png`

**Run when:** After training CNN, for interpretation

**Expected:** ~5 minutes

---

#### **11_statistical_analysis.py**
**Purpose:** Rigorous statistical validation
**What it does:**
- One-way ANOVA for each feature
- Post-hoc pairwise comparisons
- Effect size calculations (Cohen's d)
- Bonferroni correction for multiple testing
- Feature correlation analysis

**Statistical Tests:**
```
1. ANOVA:
   - Tests if conditions differ on each feature
   - Reports F-statistic and p-value
   - Applies Bonferroni correction

2. Post-hoc Tests:
   - Pairwise t-tests (anesthetic vs stimulant, etc.)
   - Mann-Whitney U (non-parametric)
   - Cohen's d (effect size)

3. Visualizations:
   - Feature distributions by condition
   - Violin plots for top features
   - PCA biplot with feature loadings
   - Correlation heatmap
```

**Output:**
- `results/statistics/feature_distributions.png`
- `results/statistics/pairwise_comparison_violin.png`
- `results/statistics/statistical_analysis_results.json`

**Run when:** After feature extraction, for statistical validation

**Expected:** ~3 minutes

---

#### **12_generate_report.py**
**Purpose:** Automatically generate comprehensive PDF report
**What it does:**
- Combines ALL results and figures into one PDF
- Creates professional layout with sections
- Includes title page and executive summary
- Ready for thesis/publication/presentation

**Report Sections:**
```
1. Title Page
2. Executive Summary
3. Dataset Overview
4. Model Training Results
5. Feature Analysis (PCA, t-SNE)
6. Classification Results
7. Feature Visualization
8. Statistical Analysis
9. Performance Comparison
```

**Output:**
- `results/comprehensive_report_YYYYMMDD_HHMMSS.pdf`

**Run when:** After all other scripts, for final documentation

**Expected:** ~2 minutes

---

##  **Model Architectures Explained**

### **Why are there architecture.py files in the folders?**

**Answer:** Those would be **standalone architecture definition files** if you want to organize code better. They're optional but help with code organization.

**Typical structure:**
```
models/
├── cnn/
│   ├── architecture.py      # Defines ZebrafishCNN2D class
│   └── train.py             # Training script
└── autoencoder/
    ├── architecture.py      # Defines ConvAutoencoder class
    └── train.py             # Training script
```

**What would be in `architecture.py`:**
```python
# cnn/architecture.py
import torch.nn as nn

class ZebrafishCNN2D(nn.Module):
    """
    Advanced 2D CNN for zebrafish classification
    
    Architecture:
    - 5 convolutional blocks with BatchNorm
    - Extracts 512D feature vectors
    - Final classifier for 3 classes
    
    Parameters: ~19M
    """
    def __init__(self, input_channels=1, num_classes=3, feature_dim=512):
        # ... model definition ...
    
    def forward(self, x):
        # ... forward pass ...
    
    def extract_features(self, x):
        # ... feature extraction ...
```

**Benefits of separate architecture.py files:**
1. **Reusability** - Import model in multiple scripts
2. **Organization** - Clean separation of model vs training code
3. **Testing** - Easier to test model architecture independently
4. **Sharing** - Easy to share just the model architecture

**Current setup:**
- Models are defined **inside** training scripts (04, 06, etc.)
- Works fine for single-use scripts
- architecture.py would be for better organization

---

##  **Complete Workflow**

### **Standard Pipeline (Scripts 01-09):**
```bash
# 1. Data Preparation
python 01_inspect_complete_dataset.py    # 2 min
python 02_create_splits.py               # 1 min

# 2. Model Training
python 04_train_advanced_cnn.py          # 30 min 
python 06_train_autoencoder.py           # 45 min

# 3. Feature Extraction & Analysis
python 07_extract_features.py            # 10 min
python 08_analyze_features.py            # 5 min
python 09_test_evaluation.py             # 5 min

# TOTAL: ~100 minutes
```

### **With Visualization & Stats (Add Scripts 10-12):**
```bash
# Add these for complete analysis:
python 10_feature_visualization.py       # 5 min
python 11_statistical_analysis.py        # 3 min
python 12_generate_report.py             # 2 min

# TOTAL: ~110 minutes
```

### **For Best Accuracy (Use augmentation):**
```bash
# Replace Script 04 with:
python 04b_train_cnn_with_augmentation.py  # 30 min

# Expected improvement: +10-20% accuracy
```

---

##  **Expected Results**

| Script | Output | Expected Value |
|--------|--------|----------------|
| 03 | SimpleCNN accuracy | 70-80% |
| 04 | ZebrafishCNN2D accuracy | 75-90% |
| 04b | With augmentation | 80-95% |
| 06 | Autoencoder reconstruction | Low MSE |
| 08 | Best classifier accuracy | 50-90% |
| 09 | Test set accuracy | Final unbiased metric |

---

##  **Key Files to Keep**

**After training:**
- `results/models/cnn2d_best.pth` - Best CNN (512D features)
- `results/models/autoencoder_best.pth` - Best autoencoder (128D)
- `results/features/*.npy` - All extracted features
- `results/comprehensive_report_*.pdf` - Final report

**For your paper:**
- All figures in `results/figures/`
- Test results in `results/test_evaluation/`
- Statistical analysis in `results/statistics/`

---

##  **Tips**

1. **Run scripts in order** - Each depends on previous outputs
2. **Check training curves** - Make sure models converge
3. **Use GPU if available** - Much faster (5-10x speedup)
4. **Save everything** - All results automatically saved
5. **Read the guides** - Check COMPLETE_WORKFLOW_GUIDE.md

---

##  **Troubleshooting**

**"No module named X"**
```bash
pip install torch torchvision tifffile scikit-learn matplotlib seaborn pandas numpy
```

**"No data found"**
- Check files are in `data/raw/anesthetic/`, `stimulant/`, `control/`
- Run Script 01 to verify

**"CUDA out of memory"**
- Reduce batch_size in config (e.g., from 16 to 8)
- Or train on CPU (slower but works)

**"Model not found"**
- Run training scripts (04, 06) before extraction (07)

---

## **Additional Documentation**

- `QUICK_START.md` - Fast track guide
- `COMPLETE_SETUP_GUIDE.md` - Detailed setup
- `COMPLETE_WORKFLOW_GUIDE.md` - All 8 scripts explained
- `ADVANCED_ANALYSIS_GUIDE.md` - Scripts 09-12 explained
- `IMPROVE_ACCURACY_GUIDE.md` - How to boost performance
- `MODEL_COMPARISON_GUIDE.md` - Which model to use

---
##  **Quick Checklist**

Before starting:
- [ ] All .tif files in `data/raw/`
- [ ] Python packages installed
- [ ] 2GB+ disk space available

After completing:
- [ ] Models trained and saved
- [ ] Features extracted
- [ ] Analysis complete
- [ ] PDF report generated
- [ ] Ready to write paper!

---

**Questions? Check the guides in `/mnt/user-data/outputs/` or ask!** 
