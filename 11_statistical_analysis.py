"""
Advanced Statistical Analysis Script
Detailed statistical comparisons between conditions
- Post-hoc tests
- Effect sizes
- Feature distribution analysis
- Pairwise comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd

from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

def load_features(features_dir, split='test'):
    """Load extracted features"""
    features_dir = Path(features_dir)
    
    combined_features = np.load(features_dir / f'{split}_combined_features.npy')
    labels = np.load(features_dir / f'{split}_labels.npy')
    
    return combined_features, labels


def cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def perform_anova(features, labels):
    """Perform one-way ANOVA for each feature"""
    
    n_features = features.shape[1]
    n_classes = len(np.unique(labels))
    
    # Separate features by class
    features_by_class = [features[labels == i] for i in range(n_classes)]
    
    # Perform ANOVA
    f_stats = []
    p_values = []
    
    for dim in range(n_features):
        groups = [features_by_class[i][:, dim] for i in range(n_classes)]
        f_stat, p_val = f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_val)
    
    return np.array(f_stats), np.array(p_values)


def perform_post_hoc(features, labels, feature_idx):
    """Perform post-hoc pairwise comparisons for a specific feature"""
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    n_classes = len(condition_names)
    
    results = []
    
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            group1 = features[labels == i, feature_idx]
            group2 = features[labels == j, feature_idx]
            
            # T-test
            t_stat, t_pval = ttest_ind(group1, group2)
            
            # Mann-Whitney U (non-parametric)
            u_stat, u_pval = mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Effect size
            effect_size = cohen_d(group1, group2)
            
            results.append({
                'comparison': f'{condition_names[i]} vs {condition_names[j]}',
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'u_statistic': u_stat,
                'u_pvalue': u_pval,
                'cohens_d': effect_size,
                'mean_diff': np.mean(group1) - np.mean(group2)
            })
    
    return results


def plot_feature_distributions(features, labels, top_features_idx, save_dir):
    """Plot distributions of top discriminative features"""
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Select top 6 features to visualize
    top_6 = top_features_idx[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature_idx in enumerate(top_6):
        ax = axes[idx]
        
        for label, (condition, color) in enumerate(zip(condition_names, colors)):
            data = features[labels == label, feature_idx]
            ax.hist(data, bins=30, alpha=0.5, label=condition, color=color, edgecolor='black')
        
        ax.set_xlabel(f'Feature {feature_idx} Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Feature {feature_idx} Distribution', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'feature_distributions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Feature distributions saved: {save_path}")


def plot_pairwise_comparisons(features, labels, top_feature_idx, save_dir):
    """Plot pairwise comparisons for top feature"""
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    
    # Prepare data
    data_list = []
    for label in range(3):
        values = features[labels == label, top_feature_idx]
        for val in values:
            data_list.append({'Condition': condition_names[label], 'Value': val})
    
    df = pd.DataFrame(data_list)
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(data=df, x='Condition', y='Value', ax=ax, palette='Set2')
    sns.swarmplot(data=df, x='Condition', y='Value', ax=ax, color='black', alpha=0.5, size=3)
    
    ax.set_title(f'Most Discriminative Feature (Feature {top_feature_idx})', 
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Value', fontsize=11)
    ax.set_xlabel('Condition', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / 'pairwise_comparison_violin.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Pairwise comparison saved: {save_path}")


def create_pca_biplot(features, labels, save_dir):
    """Create PCA biplot showing feature loadings"""
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Perform PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Get loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot of samples
    for label, (condition, color) in enumerate(zip(condition_names, colors)):
        mask = labels == label
        ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                  c=color, label=condition, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Plot top loading vectors
    top_n = 10  # Show top 10 features
    feature_importance = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_indices = np.argsort(feature_importance)[-top_n:]
    
    scale_factor = 3
    for idx in top_indices:
        ax.arrow(0, 0, loadings[idx, 0]*scale_factor, loadings[idx, 1]*scale_factor,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5, linewidth=2)
        ax.text(loadings[idx, 0]*scale_factor*1.1, loadings[idx, 1]*scale_factor*1.1,
               f'F{idx}', fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title('PCA Biplot with Feature Loadings', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'pca_biplot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" PCA biplot saved: {save_path}")


def create_correlation_heatmap(features, labels, top_features_idx, save_dir):
    """Create correlation heatmap of top features"""
    
    # Select top features
    top_features = features[:, top_features_idx[:20]]  # Top 20
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(top_features.T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=[f'F{i}' for i in top_features_idx[:20]],
                yticklabels=[f'F{i}' for i in top_features_idx[:20]],
                ax=ax)
    
    ax.set_title('Feature Correlation Heatmap (Top 20)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / 'feature_correlation_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Correlation heatmap saved: {save_path}")


def main():
    print("="*80)
    print("ADVANCED STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    # Paths
    project_dir = Path('.')
    features_dir = project_dir / 'results' / 'features'
    stats_dir = project_dir / 'results' / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print("Loading features...")
    features, labels = load_features(features_dir, 'test')
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}\n")
    
    # 1. ANOVA
    print("ONE-WAY ANOVA")
    f_stats, p_values = perform_anova(features, labels)
    
    # Apply Bonferroni correction
    alpha = 0.05
    bonferroni_threshold = alpha / len(p_values)
    significant_features = np.sum(p_values < bonferroni_threshold)
    
    print(f"Total features tested: {len(p_values)}")
    print(f"Significant features (p < {bonferroni_threshold:.2e}, Bonferroni): {significant_features}")
    print(f"Significant features (p < 0.05, uncorrected): {np.sum(p_values < 0.05)}")
    
    # Top 10 most discriminative features
    top_indices = np.argsort(p_values)[:10]
    print(f"\nTop 10 most discriminative features:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Feature {idx}: F={f_stats[idx]:.2f}, p={p_values[idx]:.2e}")
    print()
    
    # Post-hoc tests for top feature
    print("POST-HOC PAIRWISE COMPARISONS")

    
    top_feature = top_indices[0]
    print(f"Analyzing Feature {top_feature} (most discriminative):\n")
    
    post_hoc_results = perform_post_hoc(features, labels, top_feature)
    
    for result in post_hoc_results:
        print(f"{result['comparison']}:")
        print(f"  t-test: t={result['t_statistic']:.3f}, p={result['t_pvalue']:.4f}")
        print(f"  Mann-Whitney U: U={result['u_statistic']:.1f}, p={result['u_pvalue']:.4f}")
        print(f"  Cohen's d: {result['cohens_d']:.3f}")
        print(f"  Mean difference: {result['mean_diff']:.4f}\n")
    
    # Visualizations
    print("3. GENERATING VISUALIZATIONS")
    
    plot_feature_distributions(features, labels, top_indices, stats_dir)
    plot_pairwise_comparisons(features, labels, top_feature, stats_dir)
    create_pca_biplot(features, labels, stats_dir)
    create_correlation_heatmap(features, labels, top_indices, stats_dir)
    
    print()
    
    #Save results
    print("\nSAVING RESULTS")
  
    
    results_dict = {
        'anova': {
            'total_features': int(len(p_values)),
            'significant_bonferroni': int(significant_features),
            'significant_uncorrected': int(np.sum(p_values < 0.05)),
            'bonferroni_threshold': float(bonferroni_threshold)
        },
        'top_10_features': {
            int(idx): {
                'f_statistic': float(f_stats[idx]),
                'p_value': float(p_values[idx])
            }
            for idx in top_indices
        },
        'post_hoc': {
            result['comparison']: {
                't_pvalue': float(result['t_pvalue']),
                'u_pvalue': float(result['u_pvalue']),
                'cohens_d': float(result['cohens_d']),
                'mean_diff': float(result['mean_diff'])
            }
            for result in post_hoc_results
        }
    }
    
    results_path = stats_dir / 'statistical_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f" Results saved: {results_path}\n")
    
    # Summary
    
    print("\n STATISTICAL ANALYSIS COMPLETE")
    print(f"\n All results saved in: {stats_dir}/")
    print("\n Generated:")
    print("  • ANOVA results with Bonferroni correction")
    print("  • Post-hoc pairwise comparisons")
    print("  • Feature distribution plots")
    print("  • Violin plots for pairwise comparisons")
    print("  • PCA biplot with feature loadings")
    print("  • Feature correlation heatmap")
    print("\n Statistical analysis complete!")

if __name__ == "__main__":
    main()
