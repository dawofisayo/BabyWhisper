"""Model evaluation utilities for baby cry classification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


class ModelEvaluator:
    """Comprehensive evaluation utilities for baby cry classification models."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
    
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate_classification(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None) -> Dict:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            class_names: Names of classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        results = {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'per_class_metrics': {
                'precision': dict(zip(class_names, precision_per_class)),
                'recall': dict(zip(class_names, recall_per_class)),
                'f1_score': dict(zip(class_names, f1_per_class))
            },
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            prob_metrics = self._evaluate_probabilities(y_true, y_proba, class_names)
            results['probability_metrics'] = prob_metrics
        
        return results
    
    def _evaluate_probabilities(self, 
                               y_true: np.ndarray, 
                               y_proba: np.ndarray,
                               class_names: List[str]) -> Dict:
        """Evaluate probability-based metrics."""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Calculate AUC for each class
        auc_scores = {}
        if y_true_bin.shape[1] > 1:  # Multi-class
            for i, class_name in enumerate(class_names):
                try:
                    auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                    auc_scores[class_name] = auc_score
                except:
                    auc_scores[class_name] = np.nan
        
        # Overall confidence statistics
        max_probs = np.max(y_proba, axis=1)
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs)
        }
        
        # Confidence calibration
        correct_predictions = (y_true == np.argmax(y_proba, axis=1))
        confident_predictions = max_probs > 0.8
        
        calibration_metrics = {
            'accuracy_on_confident': np.mean(correct_predictions[confident_predictions]) if np.any(confident_predictions) else 0,
            'fraction_confident': np.mean(confident_predictions),
            'fraction_correct': np.mean(correct_predictions)
        }
        
        return {
            'auc_scores': auc_scores,
            'confidence_statistics': confidence_stats,
            'calibration_metrics': calibration_metrics
        }
    
    def create_evaluation_plots(self, 
                               evaluation_results: Dict,
                               save_plots: bool = True) -> List[str]:
        """
        Create comprehensive evaluation plots.
        
        Args:
            evaluation_results: Results from evaluate_classification
            save_plots: Whether to save plots to disk
            
        Returns:
            List of plot file paths (if saved)
        """
        plot_paths = []
        
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            evaluation_results['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=evaluation_results['class_names'],
            yticklabels=evaluation_results['class_names'],
            cmap='Blues',
            ax=ax
        )
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(plot_path)
        plt.show()
        
        # 2. Per-class Performance
        metrics_df = pd.DataFrame(evaluation_results['per_class_metrics'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.legend(['Precision', 'Recall', 'F1-Score'])
        ax.set_xticklabels(evaluation_results['class_names'], rotation=45)
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'per_class_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(plot_path)
        plt.show()
        
        # 3. Overall Metrics Bar Chart
        overall_metrics = evaluation_results['overall_metrics']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        metrics = list(overall_metrics.keys())
        values = list(overall_metrics.values())
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax.set_title('Overall Model Performance')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'overall_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(plot_path)
        plt.show()
        
        # 4. Confidence Distribution (if probability metrics available)
        if 'probability_metrics' in evaluation_results:
            prob_metrics = evaluation_results['probability_metrics']
            
            if 'confidence_statistics' in prob_metrics:
                # This would require the original probability data
                # For now, we'll create a placeholder
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Show confidence statistics as text
                stats = prob_metrics['confidence_statistics']
                stats_text = []
                for key, value in stats.items():
                    stats_text.append(f"{key.replace('_', ' ').title()}: {value:.3f}")
                
                ax.text(0.1, 0.5, '\n'.join(stats_text), fontsize=12, 
                       verticalalignment='center', transform=ax.transAxes)
                ax.set_title('Confidence Statistics')
                ax.axis('off')
                
                if save_plots:
                    plot_path = os.path.join(self.output_dir, 'confidence_stats.png')
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_paths.append(plot_path)
                plt.show()
        
        return plot_paths
    
    def generate_evaluation_report(self, 
                                  evaluation_results: Dict,
                                  model_info: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_classification
            model_info: Optional information about the model
            
        Returns:
            Path to the generated report file
        """
        report_content = []
        
        # Header
        report_content.append("# Baby Cry Classification Model Evaluation Report")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Model Information
        if model_info:
            report_content.append("## Model Information")
            for key, value in model_info.items():
                report_content.append(f"- **{key}**: {value}")
            report_content.append("")
        
        # Overall Performance
        report_content.append("## Overall Performance")
        overall = evaluation_results['overall_metrics']
        report_content.append(f"- **Accuracy**: {overall['accuracy']:.4f}")
        report_content.append(f"- **Precision**: {overall['precision']:.4f}")
        report_content.append(f"- **Recall**: {overall['recall']:.4f}")
        report_content.append(f"- **F1-Score**: {overall['f1_score']:.4f}")
        report_content.append("")
        
        # Per-Class Performance
        report_content.append("## Per-Class Performance")
        per_class = evaluation_results['per_class_metrics']
        
        # Create table
        report_content.append("| Class | Precision | Recall | F1-Score |")
        report_content.append("|-------|-----------|--------|----------|")
        
        for class_name in evaluation_results['class_names']:
            precision = per_class['precision'][class_name]
            recall = per_class['recall'][class_name]
            f1 = per_class['f1_score'][class_name]
            report_content.append(f"| {class_name} | {precision:.4f} | {recall:.4f} | {f1:.4f} |")
        
        report_content.append("")
        
        # Confusion Matrix
        report_content.append("## Confusion Matrix")
        cm = evaluation_results['confusion_matrix']
        
        # Header
        header = "| True\\Pred |" + "|".join([f" {name} " for name in evaluation_results['class_names']]) + "|"
        separator = "|" + "|".join(["-" * (len(name) + 2) for name in ["True\\Pred"] + evaluation_results['class_names']]) + "|"
        
        report_content.append(header)
        report_content.append(separator)
        
        for i, true_class in enumerate(evaluation_results['class_names']):
            row = f"| {true_class} |"
            for j in range(len(evaluation_results['class_names'])):
                row += f" {cm[i, j]} |"
            report_content.append(row)
        
        report_content.append("")
        
        # Detailed Classification Report
        report_content.append("## Detailed Classification Report")
        report_content.append("```")
        report_content.append(evaluation_results['classification_report'])
        report_content.append("```")
        report_content.append("")
        
        # Probability Metrics (if available)
        if 'probability_metrics' in evaluation_results:
            prob_metrics = evaluation_results['probability_metrics']
            
            report_content.append("## Probability-based Metrics")
            
            # AUC Scores
            if 'auc_scores' in prob_metrics:
                report_content.append("### AUC Scores by Class")
                for class_name, auc_score in prob_metrics['auc_scores'].items():
                    if not np.isnan(auc_score):
                        report_content.append(f"- **{class_name}**: {auc_score:.4f}")
                report_content.append("")
            
            # Confidence Statistics
            if 'confidence_statistics' in prob_metrics:
                report_content.append("### Confidence Statistics")
                stats = prob_metrics['confidence_statistics']
                for key, value in stats.items():
                    formatted_key = key.replace('_', ' ').title()
                    report_content.append(f"- **{formatted_key}**: {value:.4f}")
                report_content.append("")
            
            # Calibration Metrics
            if 'calibration_metrics' in prob_metrics:
                report_content.append("### Model Calibration")
                cal_metrics = prob_metrics['calibration_metrics']
                for key, value in cal_metrics.items():
                    formatted_key = key.replace('_', ' ').title()
                    report_content.append(f"- **{formatted_key}**: {value:.4f}")
                report_content.append("")
        
        # Recommendations
        report_content.append("## Recommendations")
        recommendations = self._generate_recommendations(evaluation_results)
        for rec in recommendations:
            report_content.append(f"- {rec}")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'evaluation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"Evaluation report saved to: {report_path}")
        return report_path
    
    def _generate_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        overall = evaluation_results['overall_metrics']
        
        # Overall performance recommendations
        if overall['accuracy'] < 0.7:
            recommendations.append("Consider collecting more training data or improving feature engineering.")
        elif overall['accuracy'] < 0.8:
            recommendations.append("Model performance is moderate. Consider hyperparameter tuning or ensemble methods.")
        else:
            recommendations.append("Good overall performance. Focus on maintaining model quality in production.")
        
        # Class-specific recommendations
        per_class = evaluation_results['per_class_metrics']
        class_names = evaluation_results['class_names']
        
        # Find worst performing classes
        f1_scores = [per_class['f1_score'][class_name] for class_name in class_names]
        worst_class_idx = np.argmin(f1_scores)
        worst_class = class_names[worst_class_idx]
        worst_f1 = f1_scores[worst_class_idx]
        
        if worst_f1 < 0.6:
            recommendations.append(f"Class '{worst_class}' has poor performance (F1={worst_f1:.3f}). Consider collecting more samples or improving class-specific features.")
        
        # Balance recommendations
        if np.std(f1_scores) > 0.2:
            recommendations.append("Significant performance variation across classes. Consider class balancing techniques.")
        
        # Probability-based recommendations
        if 'probability_metrics' in evaluation_results:
            prob_metrics = evaluation_results['probability_metrics']
            
            if 'confidence_statistics' in prob_metrics:
                mean_conf = prob_metrics['confidence_statistics']['mean_confidence']
                if mean_conf < 0.6:
                    recommendations.append("Low average confidence. Model may benefit from more training or better feature selection.")
                elif mean_conf > 0.95:
                    recommendations.append("Very high confidence may indicate overfitting. Validate on additional test data.")
        
        return recommendations
    
    def compare_models(self, 
                      model_results: Dict[str, Dict],
                      save_comparison: bool = True) -> Dict:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: Dictionary with model_name -> evaluation_results mapping
            save_comparison: Whether to save comparison plots and report
            
        Returns:
            Comparison summary
        """
        comparison_data = {}
        
        # Extract overall metrics for all models
        for model_name, results in model_results.items():
            comparison_data[model_name] = results['overall_metrics']
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Find best model for each metric
        best_models = {}
        for metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            best_models[metric] = {'model': best_model, 'score': best_score}
        
        # Create comparison plot
        if save_comparison:
            fig, ax = plt.subplots(figsize=(12, 8))
            comparison_df.plot(kind='bar', ax=ax)
            ax.set_title('Model Performance Comparison')
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticklabels(comparison_df.index, rotation=45)
            
            plt.tight_layout()
            comparison_plot_path = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        comparison_summary = {
            'comparison_table': comparison_df,
            'best_models': best_models,
            'model_count': len(model_results)
        }
        
        if save_comparison:
            # Save comparison report
            self._save_comparison_report(comparison_summary, model_results)
        
        return comparison_summary
    
    def _save_comparison_report(self, comparison_summary: Dict, model_results: Dict):
        """Save a model comparison report."""
        report_content = []
        
        report_content.append("# Model Comparison Report")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Best models summary
        report_content.append("## Best Performing Models")
        for metric, info in comparison_summary['best_models'].items():
            report_content.append(f"- **{metric.title()}**: {info['model']} ({info['score']:.4f})")
        report_content.append("")
        
        # Comparison table
        report_content.append("## Performance Comparison")
        df = comparison_summary['comparison_table']
        
        # Convert to markdown table
        report_content.append("| Model |" + "|".join([f" {col} " for col in df.columns]) + "|")
        report_content.append("|" + "|".join(["-" * (len(col) + 2) for col in ["Model"] + list(df.columns)]) + "|")
        
        for model_name, row in df.iterrows():
            row_content = f"| {model_name} |"
            for value in row:
                row_content += f" {value:.4f} |"
            report_content.append(row_content)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'model_comparison_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"Model comparison report saved to: {report_path}") 