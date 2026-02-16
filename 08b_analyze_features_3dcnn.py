"""
Feature Analysis - All Feature Types
Compares 2D CNN, Autoencoder, Combined 2D, and 3D CNN features

Frame-level features (cnn2d, ae, combined_2d):
  -> Trained and evaluated per frame

Sequence-level features (cnn3d):
  -> Trained and evaluated per 16-frame sequence
  -> These are inherently richer (motion encoded)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
CONDITION_NAMES = ['Anesthetic', 'Stimulant', 'Control']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


def load(features_dir, split, name):
    p = Path(features_dir) / f'{split}_{name}.npy'
    if p.exists():
        return np.load(p)
    print(f"  WARNING: {p.name} not found")
    return None


def run_classifiers(train_X, train_y, test_X, test_y, label):
    """Train and evaluate SVM, RF, LR on a feature set"""
    classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    print(f"\n  [{label}]")

    for name, clf in classifiers.items():
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)
        acc = accuracy_score(test_y, pred)
        f1  = f1_score(test_y, pred, average='weighted', zero_division=0)
        results[name] = {
            'accuracy': acc, 'f1': f1,
            'predictions': pred,
            'proba': clf.predict_proba(test_X) if hasattr(clf, 'predict_proba') else None
        }
        print(f"    {name:20s}  Acc={acc*100:.2f}%  F1={f1:.3f}")

    return results


def plot_tsne(features, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cond, color) in enumerate(zip(CONDITION_NAMES, COLORS)):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=cond,
                   alpha=0.7, s=40, edgecolors='white', linewidths=0.3)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, fmt, subtitle in zip(
        axes, [cm, cm_norm], ['d', '.2f'], ['Counts', 'Normalized']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CONDITION_NAMES,
                    yticklabels=CONDITION_NAMES, ax=ax)
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(y_true, y_proba, title, save_path):
    n_classes = len(CONDITION_NAMES)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cond, color) in enumerate(zip(CONDITION_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{cond} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--', lw=1.5, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(all_results, save_path):
    """Bar chart comparing all feature types and classifiers"""
    feature_types = list(all_results.keys())
    clf_names = ['SVM', 'Random Forest', 'Logistic Regression']
    x = np.arange(len(clf_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (feat, color) in enumerate(zip(feature_types, bar_colors)):
        accs = [all_results[feat][clf]['accuracy'] * 100 for clf in clf_names]
        bars = ax.bar(x + i * width, accs, width, label=feat, color=color, alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=33.3, color='gray', linestyle='--', linewidth=1.5, label='Random baseline (33.3%)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clf_names, fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('All Feature Types — Classification Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("ANALYSIS — ALL FEATURE TYPES (2D CNN + AE + 3D CNN)")
    print("=" * 80 + "\n")

    features_dir = Path('results/features_3d')
    figures_dir  = Path('results/figures_3d')
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not features_dir.exists():
        print("ERROR: Run 07b_extract_features_3dcnn.py first!")
        return

    # --------------------------------------------------------
    # Load frame-level features
    # --------------------------------------------------------
    print("Loading frame-level features (2D CNN, AE, Combined)...")
    train_cnn2d  = load(features_dir, 'train', 'cnn2d_features')
    train_ae     = load(features_dir, 'train', 'ae_features')
    train_comb2d = load(features_dir, 'train', 'combined_2d_features')
    train_flabels = load(features_dir, 'train', 'frame_labels')

    test_cnn2d   = load(features_dir, 'test', 'cnn2d_features')
    test_ae      = load(features_dir, 'test', 'ae_features')
    test_comb2d  = load(features_dir, 'test', 'combined_2d_features')
    test_flabels = load(features_dir, 'test', 'frame_labels')

    # --------------------------------------------------------
    # Load sequence-level features (3D CNN)
    # --------------------------------------------------------
    print("Loading sequence-level features (3D CNN)...")
    train_cnn3d   = load(features_dir, 'train', 'cnn3d_features')
    train_slabels = load(features_dir, 'train', 'seq_labels')
    test_cnn3d    = load(features_dir, 'test',  'cnn3d_features')
    test_slabels  = load(features_dir, 'test',  'seq_labels')

    print()

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------
    print("=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)

    all_results = {}

    if train_cnn2d is not None:
        all_results['2D CNN (512D)'] = run_classifiers(
            train_cnn2d, train_flabels, test_cnn2d, test_flabels, '2D CNN')

    if train_ae is not None:
        all_results['Autoencoder (128D)'] = run_classifiers(
            train_ae, train_flabels, test_ae, test_flabels, 'Autoencoder')

    if train_comb2d is not None:
        all_results['Combined 2D (640D)'] = run_classifiers(
            train_comb2d, train_flabels, test_comb2d, test_flabels, 'Combined 2D')

    if train_cnn3d is not None:
        all_results['3D CNN (256D)'] = run_classifiers(
            train_cnn3d, train_slabels, test_cnn3d, test_slabels, '3D CNN (sequences)')

    # --------------------------------------------------------
    # Find best overall
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEST RESULTS SUMMARY")
    print("=" * 80)

    best_acc  = 0
    best_combo = ('', '')
    rows = []

    for feat, clfs in all_results.items():
        for clf_name, r in clfs.items():
            rows.append((feat, clf_name, r['accuracy'], r['f1']))
            if r['accuracy'] > best_acc:
                best_acc   = r['accuracy']
                best_combo = (feat, clf_name)

    rows.sort(key=lambda x: -x[2])
    print(f"\n{'Feature':<25} {'Classifier':<22} {'Accuracy':>10} {'F1':>8}")
    print("-" * 70)
    for feat, clf, acc, f1 in rows:
        marker = " <-- BEST" if (feat == best_combo[0] and clf == best_combo[1]) else ""
        print(f"{feat:<25} {clf:<22} {acc*100:>9.2f}%  {f1:>7.3f}{marker}")

    print(f"\nBest: {best_combo[0]} + {best_combo[1]}  ({best_acc*100:.2f}%)")
    print(f"Random baseline: 33.33%")
    print(f"Improvement over random: {(best_acc - 0.333) / 0.333 * 100:.1f}%")

    # --------------------------------------------------------
    # Visualisations
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING VISUALISATIONS")
    print("=" * 80 + "\n")

    # t-SNE for each feature type
    for feat_name, train_X, test_X, test_y in [
        ('2D_CNN',      train_cnn2d,  test_cnn2d,  test_flabels),
        ('Autoencoder', train_ae,     test_ae,     test_flabels),
        ('Combined_2D', train_comb2d, test_comb2d, test_flabels),
        ('3D_CNN',      train_cnn3d,  test_cnn3d,  test_slabels),
    ]:
        if test_X is None or test_y is None:
            continue
        path = figures_dir / f'tsne_{feat_name}.png'
        plot_tsne(test_X, test_y, f't-SNE: {feat_name.replace("_"," ")} Features', path)
        print(f"  Saved: {path.name}")

    # Confusion matrices + ROC for best classifier per feature type
    for feat_name, feat_key, test_y in [
        ('2D CNN',       '2D CNN (512D)',       test_flabels),
        ('Autoencoder',  'Autoencoder (128D)',   test_flabels),
        ('Combined 2D',  'Combined 2D (640D)',   test_flabels),
        ('3D CNN',       '3D CNN (256D)',        test_slabels),
    ]:
        if feat_key not in all_results:
            continue

        best_clf = max(all_results[feat_key].items(), key=lambda x: x[1]['accuracy'])
        clf_name, r = best_clf

        cm_path  = figures_dir / f'cm_{feat_name.replace(" ","_")}.png'
        roc_path = figures_dir / f'roc_{feat_name.replace(" ","_")}.png'

        plot_confusion(test_y, r['predictions'],
                       f'{feat_name} — {clf_name}', cm_path)
        print(f"  Saved: {cm_path.name}")

        if r['proba'] is not None:
            plot_roc(test_y, r['proba'],
                     f'ROC Curves — {feat_name} ({clf_name})', roc_path)
            print(f"  Saved: {roc_path.name}")

    # Overall comparison bar chart
    if all_results:
        bar_path = figures_dir / 'all_features_comparison.png'
        plot_comparison_bar(all_results, bar_path)
        print(f"  Saved: {bar_path.name}")

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------
    summary = {
        'best_model': {
            'feature_type': best_combo[0],
            'classifier': best_combo[1],
            'accuracy': float(best_acc)
        },
        'all_results': {
            feat: {
                clf: {'accuracy': float(r['accuracy']), 'f1': float(r['f1'])}
                for clf, r in clfs.items()
            }
            for feat, clfs in all_results.items()
        }
    }
    with open(figures_dir / 'analysis_summary_3d.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved: {figures_dir}/analysis_summary_3d.json")
    print("\n🎉 Analysis complete!")
    print("\nKey insight:")
    print("  2D CNN / AE = spatial features  (what the fish looks like)")
    print("  3D CNN      = temporal features (how the fish moves)")
    print("  Compare the two to see which signal matters more for your data!")


if __name__ == "__main__":
    main()
