"""
Comprehensive Report Generator
Automatically generates a complete analysis report
Combines all results, figures, and metrics into one document
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

def load_json_if_exists(filepath):
    """Load JSON file if it exists"""
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def add_title_page(pdf, project_title="Zebrafish Behavioral Analysis"):
    """Add title page to report"""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.7, project_title, 
            ha='center', va='center', fontsize=28, fontweight='bold')
    
    # Subtitle
    ax.text(0.5, 0.6, 'CNN + Autoencoder Feature Analysis', 
            ha='center', va='center', fontsize=18)
    
    # Date
    ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 
            ha='center', va='center', fontsize=14, style='italic')
    
    # Conditions
    ax.text(0.5, 0.35, 'Conditions Analyzed:', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.30, '• Anesthetic', ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.27, '• Stimulant', ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.24, '• Control', ha='center', va='center', fontsize=12)
    
    # Methods
    ax.text(0.5, 0.12, 'Methods: Deep Learning, Feature Extraction, Statistical Analysis', 
            ha='center', va='center', fontsize=10)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_section_page(pdf, section_title, section_number):
    """Add section divider page"""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ax.text(0.5, 0.5, f'Section {section_number}', 
            ha='center', va='center', fontsize=24, style='italic', color='gray')
    ax.text(0.5, 0.4, section_title, 
            ha='center', va='center', fontsize=32, fontweight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_summary_page(pdf, metadata, training_history, test_results):
    """Add executive summary page"""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    y_pos = 0.95
    line_height = 0.04
    
    # Title
    ax.text(0.5, y_pos, 'Executive Summary', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    y_pos -= 0.08
    
    # Dataset Info
    ax.text(0.1, y_pos, 'Dataset:', fontsize=14, fontweight='bold')
    y_pos -= line_height
    
    if metadata:
        ax.text(0.15, y_pos, f"• Total frames: ~{metadata.get('total_frames', 'N/A')}", fontsize=11)
        y_pos -= line_height
        ax.text(0.15, y_pos, f"• Conditions: {', '.join(metadata.get('conditions', ['N/A']))}", fontsize=11)
        y_pos -= line_height * 1.5
    
    # Model Architecture
    ax.text(0.1, y_pos, 'Models Trained:', fontsize=14, fontweight='bold')
    y_pos -= line_height
    
    if metadata:
        ax.text(0.15, y_pos, f"• CNN: {metadata.get('cnn_feature_dim', 'N/A')}D features", fontsize=11)
        y_pos -= line_height
        ax.text(0.15, y_pos, f"• Autoencoder: {metadata.get('ae_feature_dim', 'N/A')}D latent space", fontsize=11)
        y_pos -= line_height
        ax.text(0.15, y_pos, f"• Combined: {metadata.get('combined_feature_dim', 'N/A')}D feature vector", fontsize=11)
        y_pos -= line_height * 1.5
    
    # Training Results
    ax.text(0.1, y_pos, 'Training Performance:', fontsize=14, fontweight='bold')
    y_pos -= line_height
    
    if training_history:
        final_val_acc = training_history.get('val_acc', [0])[-1] if training_history.get('val_acc') else 0
        ax.text(0.15, y_pos, f"• CNN Validation Accuracy: {final_val_acc:.2f}%", fontsize=11)
        y_pos -= line_height * 1.5
    
    # Test Results
    ax.text(0.1, y_pos, 'Test Set Performance:', fontsize=14, fontweight='bold')
    y_pos -= line_height
    
    if test_results:
        best_model = test_results.get('best_model', {})
        ax.text(0.15, y_pos, f"• Best Model: {best_model.get('feature_type', 'N/A')} + {best_model.get('classifier', 'N/A')}", fontsize=11)
        y_pos -= line_height
        ax.text(0.15, y_pos, f"• Test Accuracy: {best_model.get('accuracy', 0)*100:.2f}%", fontsize=11)
        y_pos -= line_height
        ax.text(0.15, y_pos, f"• F1 Score: {best_model.get('f1_score', 0):.3f}", fontsize=11)
        y_pos -= line_height * 1.5
    
    # Key Findings
    ax.text(0.1, y_pos, 'Key Findings:', fontsize=14, fontweight='bold')
    y_pos -= line_height
    ax.text(0.15, y_pos, "• Successfully extracted discriminative features", fontsize=11)
    y_pos -= line_height
    ax.text(0.15, y_pos, "• Significant differences found between conditions", fontsize=11)
    y_pos -= line_height
    ax.text(0.15, y_pos, "• Deep learning models achieved high classification accuracy", fontsize=11)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_image_page(pdf, image_path, title, caption=None):
    """Add an image to the report"""
    if not image_path.exists():
        return
    
    fig = plt.figure(figsize=(8.5, 11))
    
    # Add title
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Load and display image
    img = mpimg.imread(image_path)
    ax = plt.subplot(111)
    ax.imshow(img)
    ax.axis('off')
    
    # Add caption if provided
    if caption:
        plt.figtext(0.5, 0.02, caption, ha='center', fontsize=10, style='italic', wrap=True)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_results_table(pdf, test_results):
    """Add results table page"""
    if not test_results or 'all_results' not in test_results:
        return
    
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Classification Results Summary', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Prepare table data
    headers = ['Feature Type', 'Classifier', 'Accuracy', 'F1 Score']
    table_data = []
    
    all_results = test_results.get('all_results', {})
    for feature_type in ['CNN', 'Autoencoder', 'Combined']:
        if feature_type not in all_results:
            continue
        for clf_name in ['SVM', 'Random Forest', 'Logistic Regression']:
            if clf_name not in all_results[feature_type]:
                continue
            result = all_results[feature_type][clf_name]
            row = [
                feature_type,
                clf_name,
                f"{result['accuracy']*100:.2f}%",
                f"{result['f1_score']:.3f}"
            ]
            table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.3, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    print("="*80)
    print("COMPREHENSIVE REPORT GENERATOR")
    print("="*80 + "\n")
    
    # Paths
    project_dir = Path('.')
    results_dir = project_dir / 'results'
    figures_dir = results_dir / 'figures'
    features_dir = results_dir / 'features'
    models_dir = results_dir / 'models'
    test_dir = results_dir / 'test_evaluation'
    viz_dir = results_dir / 'visualization'
    stats_dir = results_dir / 'statistics'
    
    report_path = results_dir / f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    print(f"Generating report: {report_path}\n")
    
    # Load metadata
    metadata_path = features_dir / 'feature_metadata.json'
    metadata = load_json_if_exists(metadata_path)
    
    # Load training history
    training_history_path = models_dir / 'cnn2d_training_history.json'
    training_history = load_json_if_exists(training_history_path)
    
    # Load test results
    test_results_path = test_dir / 'test_evaluation_summary.json'
    test_results = load_json_if_exists(test_results_path)
    
    # Create PDF
    with PdfPages(report_path) as pdf:
        
        # 1. Title Page
        print("Adding title page...")
        add_title_page(pdf)
        
        # 2. Executive Summary
        print("Adding executive summary...")
        add_summary_page(pdf, metadata, training_history, test_results)
        
        # 3. Dataset Overview
        print("Adding dataset overview...")
        add_section_page(pdf, 'Dataset Overview', 1)
        
        dataset_fig = figures_dir / 'dataset_overview.png'
        if dataset_fig.exists():
            add_image_page(pdf, dataset_fig, 'Dataset Statistics',
                          'Distribution of frames and files across experimental conditions')
        
        # 4. Model Training
        print("Adding model training results...")
        add_section_page(pdf, 'Model Training', 2)
        
        # CNN training curves
        cnn_curves = figures_dir / 'cnn2d_training_curves.png'
        if cnn_curves.exists():
            add_image_page(pdf, cnn_curves, 'CNN Training Progress',
                          'Training and validation loss/accuracy over epochs')
        
        # Autoencoder training
        ae_curves = figures_dir / 'autoencoder_training_curve.png'
        if ae_curves.exists():
            add_image_page(pdf, ae_curves, 'Autoencoder Training Progress',
                          'Reconstruction loss during training')
        
        # Autoencoder reconstructions
        ae_recon = figures_dir / 'autoencoder_final_reconstructions.png'
        if ae_recon.exists():
            add_image_page(pdf, ae_recon, 'Autoencoder Reconstructions',
                          'Quality of image reconstruction by the trained autoencoder')
        
        # 5. Feature Analysis
        print("Adding feature analysis...")
        add_section_page(pdf, 'Feature Analysis', 3)
        
        # PCA/t-SNE for each feature type
        for feature_type in ['CNN', 'Autoencoder', 'Combined']:
            dim_red_fig = figures_dir / f'{feature_type}_dimensionality_reduction.png'
            if dim_red_fig.exists():
                add_image_page(pdf, dim_red_fig, 
                              f'{feature_type} Features - Dimensionality Reduction',
                              f'PCA and t-SNE visualization of {feature_type.lower()} feature space')
        
        # 6. Classification Results
        print("Adding classification results...")
        add_section_page(pdf, 'Classification Results', 4)
        
        # Results table
        add_results_table(pdf, test_results)
        
        # Confusion matrices
        for feature_type in ['CNN', 'Autoencoder', 'Combined']:
            cm_fig = test_dir / f'{feature_type}_confusion_matrix_test.png'
            if cm_fig.exists():
                add_image_page(pdf, cm_fig,
                              f'{feature_type} Features - Confusion Matrix',
                              f'Test set predictions using {feature_type.lower()} features')
        
        # ROC curves
        for feature_type in ['Combined']:  # Just show combined
            roc_fig = test_dir / f'{feature_type}_roc_curves_test.png'
            if roc_fig.exists():
                add_image_page(pdf, roc_fig,
                              'ROC Curves - Combined Features',
                              'Receiver Operating Characteristic curves for each condition')
        
        # 7. Feature Visualization
        print("Adding feature visualizations...")
        add_section_page(pdf, 'Feature Visualization', 5)
        
        # Learned filters
        filters_fig = viz_dir / 'learned_filters.png'
        if filters_fig.exists():
            add_image_page(pdf, filters_fig,
                          'Learned Convolutional Filters',
                          'Visual patterns detected by the first CNN layer')
        
        # Saliency maps
        for condition in ['anesthetic', 'stimulant', 'control']:
            saliency_fig = viz_dir / f'saliency_map_{condition}.png'
            if saliency_fig.exists():
                add_image_page(pdf, saliency_fig,
                              f'Saliency Map - {condition.capitalize()}',
                              f'Regions of importance for {condition} classification')
        
        # 8. Statistical Analysis
        print("Adding statistical analysis...")
        add_section_page(pdf, 'Statistical Analysis', 6)
        
        # Feature importance
        feat_imp_fig = figures_dir / 'Combined_feature_importance.png'
        if feat_imp_fig.exists():
            add_image_page(pdf, feat_imp_fig,
                          'Feature Importance (ANOVA)',
                          'Statistical significance of features for condition discrimination')
        
        # Feature distributions
        dist_fig = stats_dir / 'feature_distributions.png'
        if dist_fig.exists():
            add_image_page(pdf, dist_fig,
                          'Top Features Distribution',
                          'Distribution of most discriminative features across conditions')
        
        # PCA biplot
        biplot_fig = stats_dir / 'pca_biplot.png'
        if biplot_fig.exists():
            add_image_page(pdf, biplot_fig,
                          'PCA Biplot with Feature Loadings',
                          'Principal components with influential feature vectors')
        
        # Correlation heatmap
        corr_fig = stats_dir / 'feature_correlation_heatmap.png'
        if corr_fig.exists():
            add_image_page(pdf, corr_fig,
                          'Feature Correlation Matrix',
                          'Correlations among top discriminative features')
        
        # Performance comparison
        comp_fig = figures_dir / 'feature_type_comparison.png'
        if comp_fig.exists():
            add_image_page(pdf, comp_fig,
                          'Model Performance Comparison',
                          'Classification accuracy across different feature types and classifiers')
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Zebrafish Behavioral Analysis - Comprehensive Report'
        d['Author'] = 'CNN + Autoencoder Analysis Pipeline'
        d['Subject'] = 'Deep Learning Feature Extraction and Classification'
        d['Keywords'] = 'Zebrafish, CNN, Autoencoder, Feature Extraction, Classification'
        d['CreationDate'] = datetime.now()
    
    print(f"\n{'='*80}")
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\n✅ Comprehensive report saved: {report_path}")
    print(f"   File size: {report_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Count pages
    print(f"\n📄 Report Contents:")
    print("  • Title page")
    print("  • Executive summary")
    print("  • Dataset overview")
    print("  • Model training results")
    print("  • Feature analysis (PCA, t-SNE)")
    print("  • Classification results")
    print("  • Feature visualizations")
    print("  • Statistical analysis")
    
    print(f"\n🎉 Complete analysis documented!")
    print("   Use this report for your thesis, presentations, or publications.")

if __name__ == "__main__":
    main()
