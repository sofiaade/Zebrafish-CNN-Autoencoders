import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score)
from scipy import stats

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

def load_features(features_dir, split='test'):
    features_dir = Path(features_dir)
    
    cnn_features = np.load(features_dir / f'{split}_cnn_features.npy')
    ae_features = np.load(features_dir / f'{split}_ae_features.npy')
    combined_features = np.load(features_dir / f'{split}_combined_features.npy')
    labels = np.load(features_dir / f'{split}_labels.npy')
    
    return cnn_features, ae_features, combined_features, labels


def dimensionality_reduction_analysis(features, labels, save_dir, feature_name='combined'):
    print(f"\nDimensionality Reduction Analysis ({feature_name})...")
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # PCA
    print("  Computing PCA...")
    pca = PCA(n_components=50)
    features_pca_50 = pca.fit_transform(features)
    
    # PCA 2D for visualization
    pca_2d = PCA(n_components=2)
    features_pca_2d = pca_2d.fit_transform(features)
    
    # t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA plot
    ax = axes[0]
    for label_id, (condition, color) in enumerate(zip(condition_names, colors)):
        mask = labels == label_id
        ax.scatter(features_pca_2d[mask, 0], features_pca_2d[mask, 1],
                  c=color, label=condition, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'PCA - {feature_name} Features', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # t-SNE plot
    ax = axes[1]
    for label_id, (condition, color) in enumerate(zip(condition_names, colors)):
        mask = labels == label_id
        ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                  c=color, label=condition, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title(f't-SNE - {feature_name} Features', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{feature_name}_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # PCA variance plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=11)
    ax.set_title('PCA - Explained Variance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'{feature_name}_pca_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {feature_name}_dimensionality_reduction.png")
    print(f" Saved: {feature_name}_pca_variance.png")
    
    return features_pca_50


def clustering_analysis(features, labels, save_dir, feature_name='combined'):
    print(f"\nClustering Analysis ({feature_name})...")
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Evaluate clustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    
    print(f"  K-means Clustering:")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Normalized Mutual Info: {nmi:.3f}")
    
    # Hierarchical clustering dendrogram
    if len(features) > 100:
        sample_idx = np.random.choice(len(features), 100, replace=False)
        features_sample = features[sample_idx]
        labels_sample = labels[sample_idx]
    else:
        features_sample = features
        labels_sample = labels
    
    linkage_matrix = linkage(features_sample, method='ward')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, ax=ax, color_threshold=0)
    ax.set_title(f'Hierarchical Clustering Dendrogram - {feature_name}', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Distance', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / f'{feature_name}_dendrogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {feature_name}_dendrogram.png")
    
    return {'ari': ari, 'nmi': nmi}


def classification_analysis(train_features, train_labels, test_features, test_labels, 
                           save_dir, feature_name='combined'):
    print(f"\nClassification Analysis ({feature_name})...")
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    
    classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\n  {clf_name}:")
        
        # Train
        clf.fit(train_features, train_labels)
        
        # Predict
        train_pred = clf.predict(train_features)
        test_pred = clf.predict(test_features)
        
        # Metrics
        train_acc = accuracy_score(train_labels, train_pred)
        test_acc = accuracy_score(test_labels, test_pred)
        test_f1 = f1_score(test_labels, test_pred, average='weighted')
        
        print(f"    Train Accuracy: {train_acc*100:.2f}%")
        print(f"    Test Accuracy: {test_acc*100:.2f}%")
        print(f"    Test F1 Score: {test_f1:.3f}")
        
        results[clf_name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'predictions': test_pred
        }
        
        # Classification report
        print(f"\n{classification_report(test_labels, test_pred, target_names=condition_names)}")
    
    # Confusion matrix for best classifier
    best_clf = max(results.items(), key=lambda x: x[1]['test_acc'])
    best_clf_name = best_clf[0]
    best_pred = best_clf[1]['predictions']
    
    cm = confusion_matrix(test_labels, best_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=condition_names, yticklabels=condition_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix - {best_clf_name}\n{feature_name} Features', 
                fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / f'{feature_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n Saved: {feature_name}_confusion_matrix.png")
    
    return results


def statistical_analysis(features, labels, save_dir, feature_name='combined'):
    print(f"\nStatistical Analysis ({feature_name})...")
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    
    # Separate features by condition
    features_by_condition = [features[labels == i] for i in range(3)]
    
    # Compute mean features per condition
    mean_features = [np.mean(f, axis=0) for f in features_by_condition]
    std_features = [np.std(f, axis=0) for f in features_by_condition]
    
    # ANOVA for each feature dimension
    f_stats = []
    p_values = []
    
    for dim in range(features.shape[1]):
        f_stat, p_val = stats.f_oneway(
            features_by_condition[0][:, dim],
            features_by_condition[1][:, dim],
            features_by_condition[2][:, dim]
        )
        f_stats.append(f_stat)
        p_values.append(p_val)
    
    f_stats = np.array(f_stats)
    p_values = np.array(p_values)
    
    # Find most discriminative features
    significant_features = np.where(p_values < 0.05)[0]
    print(f" Significant features (p < 0.05): {len(significant_features)}/{len(p_values)}")
    
    # Top 10 most discriminative features
    top_indices = np.argsort(p_values)[:10]
    
    print(f"\n  Top 10 discriminative features:")
    for i, idx in enumerate(top_indices, 1):
        print(f" {i}. Feature {idx}: p-value = {p_values[idx]:.2e}")
    
    # Plot feature importance (p-values)
    fig, ax = plt.subplots(figsize=(12, 5))
    sorted_indices = np.argsort(p_values)[:50]  # Top 50
    ax.bar(range(len(sorted_indices)), -np.log10(p_values[sorted_indices]))
    ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
    ax.set_xlabel('Feature Rank', fontsize=11)
    ax.set_ylabel('-log10(p-value)', fontsize=11)
    ax.set_title(f'Feature Importance (ANOVA) - {feature_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / f'{feature_name}_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {feature_name}_feature_importance.png")
    
    return {
        'significant_features': len(significant_features),
        'total_features': len(p_values),
        'top_10_indices': top_indices.tolist(),
        'top_10_pvalues': p_values[top_indices].tolist()
    }


def compare_feature_types(cnn_results, ae_results, combined_results, save_dir):
    print("\nComparing Feature Types...")
    
    # Extract test accuracies
    feature_types = ['CNN', 'Autoencoder', 'Combined']
    results_list = [cnn_results, ae_results, combined_results]
    
    classifiers = list(cnn_results.keys())
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(classifiers))
    width = 0.25
    
    for i, (feature_type, results) in enumerate(zip(feature_types, results_list)):
        accuracies = [results[clf]['test_acc'] * 100 for clf in classifiers]
        ax.bar(x + i*width, accuracies, width, label=feature_type, alpha=0.8)
    
    ax.set_xlabel('Classifier', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Classification Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classifiers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_type_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: feature_type_comparison.png")


def main():
    print("\nFEATURE ANALYSIS")    
    # Paths
    project_dir = Path('.')
    features_dir = project_dir / 'results' / 'features'
    figures_dir = project_dir / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if features exist
    if not features_dir.exists():
        print("ERROR: Features directory not found!")
        print("Run 07_extract_features.py first")
        return
    
    # Load features
    print("Loading features...")
    train_cnn, train_ae, train_combined, train_labels = load_features(features_dir, 'train')
    test_cnn, test_ae, test_combined, test_labels = load_features(features_dir, 'test')
    
    print(f"  Train: {len(train_labels)} samples")
    print(f"  Test: {len(test_labels)} samples")
    print(f"  CNN features: {train_cnn.shape[1]} dimensions")
    print(f"  Autoencoder features: {train_ae.shape[1]} dimensions")
    print(f"  Combined features: {train_combined.shape[1]} dimensions")
    
    # Results storage
    all_results = {}
    
    # Analyze each feature type
    for feature_name, train_feat, test_feat in [
        ('CNN', train_cnn, test_cnn),
        ('Autoencoder', train_ae, test_ae),
        ('Combined', train_combined, test_combined)
    ]:
        print(f"ANALYZING {feature_name.upper()} FEATURES")
        
        # Dimensionality reduction
        _ = dimensionality_reduction_analysis(test_feat, test_labels, figures_dir, feature_name)
        
        # Clustering
        clustering_results = clustering_analysis(test_feat, test_labels, figures_dir, feature_name)
        
        # Classification
        classification_results = classification_analysis(
            train_feat, train_labels, test_feat, test_labels, figures_dir, feature_name
        )
        
        # Statistical analysis
        stat_results = statistical_analysis(test_feat, test_labels, figures_dir, feature_name)
        
        # Store results
        all_results[feature_name] = {
            'clustering': clustering_results,
            'classification': classification_results,
            'statistics': stat_results
        }
    
    # Compare feature types
    print("\nCOMPARING FEATURE TYPES")
    compare_feature_types(
        all_results['CNN']['classification'],
        all_results['Autoencoder']['classification'],
        all_results['Combined']['classification'],
        figures_dir
    )
    
    # Save results summary
    results_summary = {}
    for feature_type in ['CNN', 'Autoencoder', 'Combined']:
        clf_results = all_results[feature_type]['classification']
        best_clf = max(clf_results.items(), key=lambda x: x[1]['test_acc'])
        
        results_summary[feature_type] = {
            'best_classifier': best_clf[0],
            'best_test_accuracy': float(best_clf[1]['test_acc']),
            'best_test_f1': float(best_clf[1]['test_f1']),
            'clustering_ari': float(all_results[feature_type]['clustering']['ari']),
            'clustering_nmi': float(all_results[feature_type]['clustering']['nmi']),
            'significant_features': all_results[feature_type]['statistics']['significant_features']
        }
    
    # Save to JSON
    with open(figures_dir / 'analysis_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    for feature_type in ['CNN', 'Autoencoder', 'Combined']:
        summary = results_summary[feature_type]
        print(f"\n{feature_type} Features:")
        print(f"  Best Classifier: {summary['best_classifier']}")
        print(f"  Test Accuracy: {summary['best_test_accuracy']*100:.2f}%")
        print(f"  Test F1 Score: {summary['best_test_f1']:.3f}")
        print(f"  Clustering ARI: {summary['clustering_ari']:.3f}")
        print(f"  Significant Features: {summary['significant_features']}")
    
    print(f"\n All results saved in: {figures_dir}/")   
    print("\n Analysis complete! Check results/figures/ for all visualizations.")

if __name__ == "__main__":
    main()
