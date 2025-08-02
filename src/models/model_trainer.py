"""Model training utilities and data management."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ..audio_processing import AudioFeatureExtractor, AudioPreprocessor
from .classifier import BabyCryClassifier


class ModelTrainer:
    """Comprehensive model training and evaluation system."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing training data
        """
        self.data_dir = data_dir
        self.feature_extractor = AudioFeatureExtractor()
        self.preprocessor = AudioPreprocessor()
        self.classifier = None
        
        # Data storage
        self.features = None
        self.labels = None
        self.spectrograms = None
        self.feature_names = None
        
    def create_synthetic_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a synthetic dataset for demonstration purposes.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features, labels)
        """
        print("Creating synthetic baby cry dataset...")
        
        np.random.seed(42)
        classes = ['hunger', 'pain', 'discomfort', 'tiredness', 'normal']
        
        # Get actual number of features from feature extractor
        # Create a dummy audio file to determine feature count
        import tempfile
        import soundfile as sf
        
        # Create a temporary audio sample to get feature count
        temp_audio = np.random.normal(0, 0.1, 22050 * 3)  # 3 seconds at 22050 Hz
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, temp_audio, 22050)
            dummy_features = self.feature_extractor.extract_feature_vector(temp_file.name)
            n_features = len(dummy_features)
            os.unlink(temp_file.name)
        
        print(f"Using {n_features} features for synthetic dataset")
        
        # Generate features with different characteristics for each class
        features = []
        labels = []
        
        for i in range(n_samples):
            # Randomly select a class
            class_idx = np.random.randint(0, len(classes))
            class_name = classes[class_idx]
            
            # Generate features based on class characteristics
            if class_name == 'hunger':
                # Hunger cries tend to be rhythmic and persistent
                feature_vector = np.random.normal(0.3, 0.2, n_features)
                feature_vector[0:13] = np.random.normal(0.5, 0.3, 13)  # MFCC characteristics
                feature_vector[26:32] = np.random.normal(0.6, 0.2, 6)  # Spectral features
                
            elif class_name == 'pain':
                # Pain cries are often sudden and intense
                feature_vector = np.random.normal(0.7, 0.3, n_features)
                feature_vector[0:13] = np.random.normal(0.8, 0.2, 13)  # High MFCC values
                feature_vector[32:37] = np.random.normal(0.9, 0.1, 5)  # High temporal features
                
            elif class_name == 'discomfort':
                # Discomfort cries are moderate intensity
                feature_vector = np.random.normal(0.4, 0.25, n_features)
                feature_vector[0:13] = np.random.normal(0.4, 0.25, 13)
                feature_vector[37:49] = np.random.normal(0.5, 0.2, 12)  # Chroma features
                
            elif class_name == 'tiredness':
                # Tired cries are often whimpering and lower energy
                feature_vector = np.random.normal(0.2, 0.15, n_features)
                feature_vector[0:13] = np.random.normal(0.3, 0.2, 13)
                # Use last 4 features for energy stats (dynamic)
                feature_vector[-4:] = np.random.normal(0.2, 0.1, 4)  # Lower energy stats
                
            else:  # normal
                # Normal sounds (not crying)
                feature_vector = np.random.normal(0.1, 0.1, n_features)
                feature_vector[0:13] = np.random.normal(0.1, 0.15, 13)
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.05, n_features)
            feature_vector += noise
            
            features.append(feature_vector)
            labels.append(class_name)
        
        return np.array(features), np.array(labels)
    
    def load_dataset_from_audio(self, audio_dir: str, 
                               labels_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from audio files.
        
        Args:
            audio_dir: Directory containing audio files
            labels_file: Optional CSV file with labels
            
        Returns:
            Tuple of (features, labels)
        """
        audio_files = []
        labels = []
        
        # Get audio files and labels
        if labels_file and os.path.exists(labels_file):
            # Load labels from CSV
            df = pd.read_csv(labels_file)
            for _, row in df.iterrows():
                audio_path = os.path.join(audio_dir, row['filename'])
                if os.path.exists(audio_path):
                    audio_files.append(audio_path)
                    labels.append(row['label'])
        else:
            # Infer labels from subdirectories
            for class_name in os.listdir(audio_dir):
                class_dir = os.path.join(audio_dir, class_name)
                if os.path.isdir(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                            audio_files.append(os.path.join(class_dir, filename))
                            labels.append(class_name)
        
        # Extract features
        print(f"Extracting features from {len(audio_files)} audio files...")
        features = []
        valid_labels = []
        
        for i, audio_file in enumerate(tqdm(audio_files)):
            try:
                feature_vector = self.feature_extractor.extract_feature_vector(audio_file)
                features.append(feature_vector)
                valid_labels.append(labels[i])
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        return np.array(features), np.array(valid_labels)
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray,
                    test_size: float = 0.2, validation_size: float = 0.2,
                    random_state: int = 42) -> Dict:
        """
        Prepare data for training with train/validation/test splits.
        
        Args:
            features: Feature matrix
            labels: Target labels
            test_size: Proportion of test data
            validation_size: Proportion of validation data
            random_state: Random seed
            
        Returns:
            Dictionary containing data splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Store data
        self.features = features
        self.labels = labels
        self.feature_names = self.feature_extractor.get_feature_names()
        
        data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # Print data information
        print(f"Data prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {np.unique(labels)}")
        
        return data_splits
    
    def train_model(self, data_splits: Dict, model_type: str = 'ensemble',
                   hyperparameter_tuning: bool = False) -> BabyCryClassifier:
        """
        Train the baby cry classifier.
        
        Args:
            data_splits: Data splits dictionary
            model_type: Type of model to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained classifier
        """
        print(f"Training {model_type} model...")
        
        # Initialize classifier
        self.classifier = BabyCryClassifier(model_type=model_type)
        
        # Hyperparameter tuning (optional)
        if hyperparameter_tuning and model_type in ['rf', 'svm']:
            print("Performing hyperparameter tuning...")
            self.classifier = self._tune_hyperparameters(
                data_splits['X_train'], 
                data_splits['y_train'],
                model_type
            )
        
        # Train the model
        X_train_combined = np.vstack([data_splits['X_train'], data_splits['X_val']])
        y_train_combined = np.hstack([data_splits['y_train'], data_splits['y_val']])
        
        performances = self.classifier.fit(X_train_combined, y_train_combined)
        
        print("\nTraining completed!")
        for model_name, score in performances.items():
            print(f"{model_name.upper()} Performance: {score:.3f}")
        
        return self.classifier
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                            model_type: str) -> BabyCryClassifier:
        """Perform hyperparameter tuning using GridSearchCV."""
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(random_state=42)
            
        elif model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
            from sklearn.svm import SVC
            base_model = SVC(probability=True, random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=StratifiedKFold(n_splits=3),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit on scaled data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        grid_search.fit(X_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Create classifier with best parameters
        classifier = BabyCryClassifier(model_type=model_type)
        if model_type == 'rf':
            classifier.models['rf'] = RandomForestClassifier(**grid_search.best_params_, random_state=42)
        elif model_type == 'svm':
            classifier.models['svm'] = SVC(**grid_search.best_params_, probability=True, random_state=42)
        
        return classifier
    
    def evaluate_model(self, data_splits: Dict, 
                      save_plots: bool = True) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            data_splits: Data splits dictionary
            save_plots: Whether to save evaluation plots
            
        Returns:
            Evaluation results
        """
        if self.classifier is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model...")
        
        # Evaluate on test set
        results = self.classifier.evaluate(
            data_splits['X_test'], 
            data_splits['y_test']
        )
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1-Score: {results['f1_score']:.3f}")
        
        print(f"\nDetailed Classification Report:")
        print(results['classification_report'])
        
        # Create visualizations
        if save_plots:
            self._create_evaluation_plots(results, data_splits)
        
        return results
    
    def _create_evaluation_plots(self, results: Dict, data_splits: Dict):
        """Create evaluation plots and save them."""
        os.makedirs('plots', exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=self.classifier.classes,
            yticklabels=self.classifier.classes,
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance (if available)
        importance = self.classifier.get_feature_importance()
        if 'rf' in importance:
            plt.figure(figsize=(12, 8))
            top_features = np.argsort(importance['rf'])[-20:]  # Top 20 features
            
            plt.barh(range(len(top_features)), importance['rf'][top_features])
            plt.yticks(range(len(top_features)), 
                      [self.feature_names[i] for i in top_features])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Most Important Features (Random Forest)')
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Class Distribution
        plt.figure(figsize=(10, 6))
        unique, counts = np.unique(data_splits['y_test'], return_counts=True)
        plt.bar(unique, counts)
        plt.title('Test Set Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Prediction Confidence Distribution
        probabilities = results['probabilities']
        max_probs = np.max(probabilities, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Maximum Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.axvline(x=0.6, color='red', linestyle='--', label='Confidence Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_results(self, results: Dict, 
                            model_path: str = "models/baby_cry_classifier"):
        """
        Save training results and model.
        
        Args:
            results: Evaluation results
            model_path: Path to save the model
        """
        os.makedirs('models', exist_ok=True)
        
        # Save the trained model
        self.classifier.save_model(model_path)
        
        # Save training metadata
        metadata = {
            'model_type': self.classifier.model_type,
            'feature_count': len(self.feature_names),
            'classes': self.classifier.classes,
            'test_accuracy': results['accuracy'],
            'test_precision': results['precision'],
            'test_recall': results['recall'],
            'test_f1': results['f1_score']
        }
        
        pd.DataFrame([metadata]).to_csv('models/training_metadata.csv', index=False)
        
        print(f"Model saved to {model_path}")
        print("Training metadata saved to models/training_metadata.csv")
    
    def quick_demo_training(self) -> BabyCryClassifier:
        """
        Quick demonstration training with synthetic data.
        
        Returns:
            Trained classifier ready for demonstration
        """
        print("Starting quick demo training...")
        
        # Create synthetic dataset
        features, labels = self.create_synthetic_dataset(n_samples=1000)
        
        # Prepare data
        data_splits = self.prepare_data(features, labels)
        
        # Train model
        classifier = self.train_model(data_splits, model_type='ensemble')
        
        # Evaluate
        results = self.evaluate_model(data_splits, save_plots=True)
        
        # Save results
        self.save_training_results(results)
        
        print("\nDemo training completed!")
        return classifier 