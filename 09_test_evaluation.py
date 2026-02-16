import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

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


def train_and_evaluate_classifier(train_features, train_labels, test_features, test_labels, 
                                  classifier_name, classifier):
    
    # Train
    classifier.fit(train_features, train_labels)
    
    # Predict
    test_pred = classifier.predict(test_features)
    
    # Get probabilities if available
    if hasattr(classifier, 'predict_proba'):
        test_proba = classifier.predict_proba(test_features)
    else:
        test_proba = None
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred, average='weighted', zero_division=0)
    recall = recall_score(test_labels, test_pred, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, test_pred, average='weighted', zero_division=0)
    
    results = {
        'classifier': classifier_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'predictions': test_pred,
        'probabilities': test_proba
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, condition_names, save_path, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=condition_names, yticklabels=condition_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    # Normalized
    ax = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=condition_names, yticklabels=condition_names,
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_proba, condition_names, save_path, title='ROC Curves'):
    
    n_classes = len(condition_names)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (condition, color) in enumerate(zip(condition_names, colors)):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{condition} (AUC = {roc_auc[i]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_precision_recall_curves(y_true, y_proba, condition_names, save_path, 
                                 title='Precision-Recall Curves'):
    
    n_classes = len(condition_names)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute PR curve and AP for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (condition, color) in enumerate(zip(condition_names, colors)):
        ax.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{condition} (AP = {avg_precision[i]:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return avg_precision


def create_performance_table(results_dict, save_path):
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    headers = ['Feature Type', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    table_data = []
    
    for feature_type in ['CNN', 'Autoencoder', 'Combined']:
        for clf_name in ['SVM', 'Random Forest', 'Logistic Regression']:
            result = results_dict[feature_type][clf_name]
            row = [
                feature_type,
                clf_name,
                f"{result['accuracy']*100:.2f}%",
                f"{result['precision']*100:.2f}%",
                f"{result['recall']*100:.2f}%",
                f"{result['f1_score']:.3f}"
            ]
            table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.2, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Test Set Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("\n TEST SET EVALUATION")
    # Paths
    project_dir = Path('.')
    features_dir = project_dir / 'results' / 'features'
    figures_dir = project_dir / 'results' / 'figures'
    test_results_dir = project_dir / 'results' / 'test_evaluation'
    test_results_dir.mkdir(parents=True, exist_ok=True)
    
    condition_names = ['Anesthetic', 'Stimulant', 'Control']
    
    # Check if features exist
    if not features_dir.exists():
        print(" ERROR: Features directory not found!")
        print("   Run 07_extract_features.py first")
        return
    
    # Load features
    print("Loading features...")
    train_cnn, train_ae, train_combined, train_labels = load_features(features_dir, 'train')
    test_cnn, test_ae, test_combined, test_labels = load_features(features_dir, 'test')
    
    print(f"  Train: {len(train_labels)} samples")
    print(f"  Test: {len(test_labels)} samples\n")
    
    # Define classifiers
    classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Results storage
    all_results = {}
    
    # Evaluate each feature type
    for feature_name, train_feat, test_feat in [
        ('CNN', train_cnn, test_cnn),
        ('Autoencoder', train_ae, test_ae),
        ('Combined', train_combined, test_combined)
    ]:
   
        print(f"\n EVALUATING {feature_name.upper()} FEATURES")
        
        all_results[feature_name] = {}
        
        for clf_name, clf in classifiers.items():
            print(f"{clf_name}:")
            
            results = train_and_evaluate_classifier(
                train_feat, train_labels, test_feat, test_labels,
                clf_name, clf
            )
            
            all_results[feature_name][clf_name] = results
            
            print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
            print(f"  Precision: {results['precision']*100:.2f}%")
            print(f"  Recall:    {results['recall']*100:.2f}%")
            print(f"  F1 Score:  {results['f1_score']:.3f}\n")
            
            # Detailed classification report
            print(classification_report(test_labels, results['predictions'], 
                                       target_names=condition_names, zero_division=0))
        
        # Find best classifier for this feature type
        best_clf = max(all_results[feature_name].items(), 
                      key=lambda x: x[1]['accuracy'])
        best_clf_name = best_clf[0]
        best_results = best_clf[1]
        
        print(f" Best classifier: {best_clf_name} ({best_results['accuracy']*100:.2f}%)\n")
        
        # Plot confusion matrix for best classifier
        cm_path = test_results_dir / f'{feature_name}_confusion_matrix_test.png'
        plot_confusion_matrix(
            test_labels, best_results['predictions'],
            condition_names, cm_path,
            title=f'Test Set - {feature_name} Features ({best_clf_name})'
        )
        print(f" Confusion matrix saved: {cm_path}")
        
        # ROC curves (if probabilities available)
        if best_results['probabilities'] is not None:
            roc_path = test_results_dir / f'{feature_name}_roc_curves_test.png'
            roc_auc = plot_roc_curves(
                test_labels, best_results['probabilities'],
                condition_names, roc_path,
                title=f'Test Set ROC - {feature_name} Features ({best_clf_name})'
            )
            print(f" ROC curves saved: {roc_path}")
            
            # Precision-Recall curves
            pr_path = test_results_dir / f'{feature_name}_pr_curves_test.png'
            avg_precision = plot_precision_recall_curves(
                test_labels, best_results['probabilities'],
                condition_names, pr_path,
                title=f'Test Set PR - {feature_name} Features ({best_clf_name})'
            )
            print(f" PR curves saved: {pr_path}")
        
        print()
    
    # Create performance comparison table
    print("\n CREATING PERFORMANCE TABLE")
    table_path = test_results_dir / 'performance_table_test.png'
    create_performance_table(all_results, table_path)
    print(f" Performance table saved: {table_path}\n")
    
    # Find overall best model
    best_overall = None
    best_acc = 0
    
    for feature_type in all_results:
        for clf_name in all_results[feature_type]:
            acc = all_results[feature_type][clf_name]['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_overall = (feature_type, clf_name)
    
    # Save summary in JSON
    summary = {
        'best_model': {
            'feature_type': best_overall[0],
            'classifier': best_overall[1],
            'accuracy': float(best_acc),
            'f1_score': float(all_results[best_overall[0]][best_overall[1]]['f1_score'])
        },
        'all_results': {}
    }
    
    for feature_type in all_results:
        summary['all_results'][feature_type] = {}
        for clf_name in all_results[feature_type]:
            summary['all_results'][feature_type][clf_name] = {
                'accuracy': float(all_results[feature_type][clf_name]['accuracy']),
                'precision': float(all_results[feature_type][clf_name]['precision']),
                'recall': float(all_results[feature_type][clf_name]['recall']),
                'f1_score': float(all_results[feature_type][clf_name]['f1_score'])
            }
    
    summary_path = test_results_dir / 'test_evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f" Summary saved: {summary_path}\n")
    
    # summary
    print("\n TEST SET EVALUATION COMPLETE")
    print(f"\n BEST MODEL:")
    print(f"   Feature Type: {best_overall[0]}")
    print(f"   Classifier: {best_overall[1]}")
    print(f"   Test Accuracy: {best_acc*100:.2f}%")
    print(f"   Test F1 Score: {all_results[best_overall[0]][best_overall[1]]['f1_score']:.3f}")
    
    print(f"\n All test results saved in: {test_results_dir}/")
    print("\n Test evaluation complete!")

if __name__ == "__main__":
    main()
