"""Model training utilities for baby cry classification."""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
from tqdm import tqdm

# Use try-except for imports to handle different execution contexts
try:
    from audio_processing import AudioFeatureExtractor, AudioPreprocessor
    from models.classifier import BabyCryClassifier
except ImportError:
    try:
        from audio_processing.feature_extractor import AudioFeatureExtractor
        from audio_processing.preprocessor import AudioPreprocessor
        from models.classifier import BabyCryClassifier
    except ImportError:
        print("⚠️  Import warning: Some modules may not be available")
        # Define placeholder classes to avoid NameError
        class AudioFeatureExtractor: pass
        class AudioPreprocessor: pass
        class BabyCryClassifier: pass

import glob
from collections import Counter


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
    
    def load_donateacry_dataset(self, dataset_path: str = "data/donateacry_corpus_cleaned_and_updated_data") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the real Donate-a-Cry dataset.
        
        Args:
            dataset_path: Path to the donateacry dataset
            
        Returns:
            Tuple of (features, labels)
        """
        print(f"Loading real baby cry data from {dataset_path}...")
        
        # Map dataset folder names to our standard classes
        class_mapping = {
            'hungry': 'hunger',
            'tired': 'tiredness', 
            'discomfort': 'discomfort',
            'belly_pain': 'pain',
            'burping': 'discomfort'  # Map burping to discomfort for now
        }
        
        features_list = []
        labels_list = []
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at {dataset_path}")
            print("🔄 Falling back to synthetic data...")
            return self.create_synthetic_dataset()
        
        # Load files from each category
        for folder_name, class_name in class_mapping.items():
            folder_path = os.path.join(dataset_path, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️  Folder {folder_name} not found, skipping...")
                continue
            
            print(f"📁 Processing {folder_name} -> {class_name}")
            
            # Find all audio files in the folder
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            print(f"   Found {len(audio_files)} audio files")
            
            # Process each audio file
            for audio_file in tqdm(audio_files, desc=f"Extracting {class_name} features"):
                try:
                    # Extract features using our feature extractor
                    feature_vector = self.feature_extractor.extract_feature_vector(audio_file)
                    
                    if feature_vector is not None and len(feature_vector) > 0:
                        features_list.append(feature_vector)
                        labels_list.append(class_name)
                    else:
                        print(f"⚠️  Failed to extract features from {audio_file}")
                        
                except Exception as e:
                    print(f"❌ Error processing {audio_file}: {str(e)}")
                    continue
        
        if len(features_list) == 0:
            print("❌ No features extracted from real data, falling back to synthetic")
            return self.create_synthetic_dataset()
        
        # Convert to numpy arrays
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        print(f"✅ Loaded {len(features)} real baby cry samples")
        print(f"📊 Feature shape: {features.shape}")
        print(f"🏷️  Classes: {np.unique(labels)}")
        
        # Print class distribution
        class_counts = Counter(labels)
        print("📈 Class distribution:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} samples")
        
        return features, labels
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train ensemble of models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Training results dictionary
        """
        print("🤖 Training ensemble models...")
        
        # Prepare data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        # Train individual models
        self.trained_models = {}
        results = {}
        
        for model_name, model in models.items():
            print(f"   Training {model_name}...")
            
            try:
                model.fit(X_train_scaled, y_train_encoded)
                
                # Validate
                val_pred = model.predict(X_val_scaled)
                val_accuracy = accuracy_score(y_val_encoded, val_pred)
                
                self.trained_models[model_name] = model
                results[model_name] = {
                    'validation_accuracy': val_accuracy,
                    'model': model
                }
                
                print(f"     ✅ {model_name}: {val_accuracy:.3f} accuracy")
                
            except Exception as e:
                print(f"     ❌ {model_name} failed: {e}")
                continue
        
        # Create ensemble
        if len(self.trained_models) >= 2:
            print("   Creating ensemble...")
            
            ensemble_models = [(name, model) for name, model in self.trained_models.items()]
            self.ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'
            )
            
            self.ensemble.fit(X_train_scaled, y_train_encoded)
            
            # Validate ensemble
            ensemble_pred = self.ensemble.predict(X_val_scaled)
            ensemble_accuracy = accuracy_score(y_val_encoded, ensemble_pred)
            
            results['ensemble'] = {
                'validation_accuracy': ensemble_accuracy,
                'model': self.ensemble
            }
            
            print(f"     ✅ ensemble: {ensemble_accuracy:.3f} accuracy")
        
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate trained models on test data.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Evaluation results dictionary
        """
        print("📊 Evaluating models on test data...")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        results = {}
        
        # Evaluate individual models
        for model_name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test_encoded, y_pred)
                precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                print(f"   {model_name}: Acc={accuracy:.3f}, F1={f1:.3f}")
                
            except Exception as e:
                print(f"   ❌ {model_name} evaluation failed: {e}")
                continue
        
        # Evaluate ensemble
        if hasattr(self, 'ensemble'):
            try:
                y_pred = self.ensemble.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test_encoded, y_pred)
                precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                
                results['ensemble'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                print(f"   🏆 ensemble: Acc={accuracy:.3f}, F1={f1:.3f}")
                
            except Exception as e:
                print(f"   ❌ ensemble evaluation failed: {e}")
        
        return results
    
    def save_models(self, model_dir: str = "models") -> bool:
        """
        Save trained models to disk.
        
        Args:
            model_dir: Directory to save models
            
        Returns:
            True if successful
        """
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save individual models
            for model_name, model in self.trained_models.items():
                model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)
                print(f"💾 Saved {model_name} to {model_path}")
            
            # Save ensemble
            if hasattr(self, 'ensemble'):
                ensemble_path = os.path.join(model_dir, "ensemble_model.pkl")
                joblib.dump(self.ensemble, ensemble_path)
                print(f"💾 Saved ensemble to {ensemble_path}")
            
            # Save preprocessing objects
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
            
            print(f"💾 Saved scaler to {scaler_path}")
            print(f"💾 Saved label encoder to {encoder_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save models: {e}")
            return False
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray,
                    test_size: float = 0.2, validation_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                    np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data splits for training.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion for test set
            validation_size: Proportion for validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, X_val, y_train, y_test, y_val)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Data prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1] if len(X_train.shape) > 1 else 'Unknown'}")
        print(f"  Classes: {list(np.unique(labels))}")
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def train_model(self, save_model: bool = True) -> Dict:
        """
        Train the baby cry classification model using real data.
        
        Args:
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        print("🚀 Starting BabyWhisper model training...")
        
        # Load real data
        print("🎯 Loading REAL baby cry data...")
        try:
            features, labels = self.load_donateacry_dataset()
            data_source = "Real Donate-a-Cry Dataset"
        except Exception as e:
            print(f"❌ Failed to load real data: {e}")
            raise ValueError("Real dataset not available. Please ensure the Donate-a-Cry dataset is properly set up.")
        
        # Rest of training logic remains the same
        X_train, X_test, X_val, y_train, y_test, y_val = self.prepare_data(features, labels)
        
        # Train models
        results = self.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Evaluate
        final_results = self.evaluate_models(X_test, y_test)
        
        # Combine results
        training_summary = {
            'data_source': data_source,
            'total_samples': len(features),
            'features_per_sample': features.shape[1] if len(features.shape) > 1 else len(features[0]),
            'classes': list(np.unique(labels)),
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'training_results': results,
            'test_results': final_results
        }
        
        if save_model:
            self.save_models()
            print("💾 Models saved successfully!")
        
        print(f"✅ Training completed using: {data_source}")
        return training_summary
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                             model_type: str):
        """Perform hyperparameter tuning using GridSearchCV."""
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42)
        elif model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
            model = SVC(random_state=42)
        else:
            raise ValueError(f"Hyperparameter tuning not supported for {model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_type}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Create classifier with best parameters
        best_model = grid_search.best_estimator_
        return BabyCryClassifier(custom_model=best_model, model_type=model_type)
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.classifier is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'test_samples': len(y_test)
        }
        
        print(f"\n📊 Model Evaluation Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        return results
    
    def save_model(self, filepath: str = None):
        """Save the trained model to disk."""
        if self.classifier is None:
            raise ValueError("No trained model to save.")
        
        if filepath is None:
            os.makedirs("models", exist_ok=True)
            filepath = f"models/{self.model_name}.pkl"
        
        self.classifier.save_model(filepath)
        print(f"Model saved to {filepath}")
        
        # Also save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_count': len(self.feature_names) if hasattr(self, 'feature_names') else None,
            'training_samples': len(self.labels) if hasattr(self, 'labels') else None,
            'classes': list(np.unique(self.labels)) if hasattr(self, 'labels') else None
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
    
    def quick_demo_training(self):
        """Quick training for demonstration purposes."""
        print("🚀 Quick Demo Training...")
        
        # Load real dataset
        try:
            features, labels = self.load_donateacry_dataset()
            print(f"✅ Loaded {len(features)} samples from real dataset")
        except Exception as e:
            print(f"❌ Failed to load real dataset: {e}")
            raise ValueError("Real dataset not available for demo training.")
        
        # Prepare data
        X_train, X_test, X_val, y_train, y_test, y_val = self.prepare_data(features, labels)
        
        # Train model
        results = self.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Evaluate
        final_results = self.evaluate_models(X_test, y_test)
        
        print(f"🎯 Demo training completed! Accuracy: {final_results['ensemble']['test_accuracy']:.3f}")
        return results 