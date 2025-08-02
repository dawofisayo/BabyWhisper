"""Baby cry classification models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class BabyCryClassifier:
    """Multi-model baby cry classifier with ensemble capabilities."""
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ('rf', 'svm', 'mlp', 'cnn', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.classes = ['hunger', 'pain', 'discomfort', 'tiredness', 'normal']
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different model architectures."""
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Support Vector Machine
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Multi-layer Perceptron
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
    
    def _create_cnn_model(self, input_shape: tuple) -> keras.Model:
        """
        Create a CNN model for spectrogram-based classification.
        
        Args:
            input_shape: Shape of input spectrograms
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            spectrograms: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the classifier(s).
        
        Args:
            X: Feature matrix
            y: Target labels
            spectrograms: Optional spectrograms for CNN training
            
        Returns:
            Dictionary of model performances
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        performances = {}
        
        if self.model_type in ['rf', 'ensemble']:
            # Train Random Forest
            self.models['rf'].fit(X_scaled, y_encoded)
            rf_score = cross_val_score(self.models['rf'], X_scaled, y_encoded, cv=5).mean()
            performances['rf'] = rf_score
            print(f"Random Forest CV Score: {rf_score:.3f}")
        
        if self.model_type in ['svm', 'ensemble']:
            # Train SVM
            self.models['svm'].fit(X_scaled, y_encoded)
            svm_score = cross_val_score(self.models['svm'], X_scaled, y_encoded, cv=5).mean()
            performances['svm'] = svm_score
            print(f"SVM CV Score: {svm_score:.3f}")
        
        if self.model_type in ['mlp', 'ensemble']:
            # Train MLP
            self.models['mlp'].fit(X_scaled, y_encoded)
            mlp_score = cross_val_score(self.models['mlp'], X_scaled, y_encoded, cv=5).mean()
            performances['mlp'] = mlp_score
            print(f"MLP CV Score: {mlp_score:.3f}")
        
        if self.model_type in ['cnn', 'ensemble'] and spectrograms is not None:
            # Train CNN
            y_categorical = keras.utils.to_categorical(y_encoded, len(self.classes))
            
            self.models['cnn'] = self._create_cnn_model(spectrograms.shape[1:])
            
            # Split data for validation
            split_idx = int(0.8 * len(spectrograms))
            X_train, X_val = spectrograms[:split_idx], spectrograms[split_idx:]
            y_train, y_val = y_categorical[:split_idx], y_categorical[split_idx:]
            
            history = self.models['cnn'].fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            cnn_score = max(history.history['val_accuracy'])
            performances['cnn'] = cnn_score
            print(f"CNN Validation Accuracy: {cnn_score:.3f}")
        
        self.is_trained = True
        return performances
    
    def predict(self, X: np.ndarray, 
                spectrograms: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained model(s).
        
        Args:
            X: Feature matrix
            spectrograms: Optional spectrograms for CNN prediction
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == 'ensemble':
            return self._ensemble_predict(X, spectrograms)
        else:
            X_scaled = self.scaler.transform(X)
            if self.model_type == 'cnn' and spectrograms is not None:
                predictions = self.models['cnn'].predict(spectrograms, verbose=0)
                return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
            else:
                predictions = self.models[self.model_type].predict(X_scaled)
                return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray, 
                      spectrograms: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            spectrograms: Optional spectrograms for CNN prediction
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == 'ensemble':
            return self._ensemble_predict_proba(X, spectrograms)
        else:
            X_scaled = self.scaler.transform(X)
            if self.model_type == 'cnn' and spectrograms is not None:
                return self.models['cnn'].predict(spectrograms, verbose=0)
            else:
                return self.models[self.model_type].predict_proba(X_scaled)
    
    def _ensemble_predict(self, X: np.ndarray, 
                         spectrograms: Optional[np.ndarray] = None) -> np.ndarray:
        """Ensemble prediction using multiple models."""
        probabilities = self._ensemble_predict_proba(X, spectrograms)
        return self.label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
    
    def _ensemble_predict_proba(self, X: np.ndarray, 
                               spectrograms: Optional[np.ndarray] = None) -> np.ndarray:
        """Ensemble probability prediction using weighted average."""
        X_scaled = self.scaler.transform(X)
        all_probas = []
        weights = []
        
        # Traditional ML models
        for model_name in ['rf', 'svm', 'mlp']:
            if model_name in self.models:
                probas = self.models[model_name].predict_proba(X_scaled)
                all_probas.append(probas)
                
                # Weight based on model type (RF tends to be more reliable)
                weight = 0.4 if model_name == 'rf' else 0.3
                weights.append(weight)
        
        # CNN model if available
        if 'cnn' in self.models and spectrograms is not None:
            cnn_probas = self.models['cnn'].predict(spectrograms, verbose=0)
            all_probas.append(cnn_probas)
            weights.append(0.5)  # Higher weight for CNN
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average
        ensemble_probas = np.average(all_probas, axis=0, weights=weights)
        return ensemble_probas
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 spectrograms_test: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            spectrograms_test: Test spectrograms
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.predict(X_test, spectrograms_test)
        probabilities = self.predict_proba(X_test, spectrograms_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # Classification report
        report = classification_report(y_test, predictions, target_names=self.classes)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=self.classes)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from trained models."""
        importance_dict = {}
        
        if 'rf' in self.models:
            importance_dict['rf'] = self.models['rf'].feature_importances_
        
        # For other models, we could use permutation importance
        # This is a simplified version
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        model_data = {
            'models': {},
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'classes': self.classes,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        # Save traditional ML models
        for name, model in self.models.items():
            if name != 'cnn':
                model_data['models'][name] = model
        
        # Save the main model data
        joblib.dump(model_data, f"{filepath}_main.pkl")
        
        # Save CNN separately if it exists
        if 'cnn' in self.models:
            self.models['cnn'].save(f"{filepath}_cnn.h5")
    
    def load_model(self, filepath: str):
        """Load a trained model from file."""
        # Load main model data
        model_data = joblib.load(f"{filepath}_main.pkl")
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.classes = model_data['classes']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        # Load CNN if it exists
        cnn_path = f"{filepath}_cnn.h5"
        if os.path.exists(cnn_path):
            self.models['cnn'] = keras.models.load_model(cnn_path)
    
    def classify_with_confidence(self, X: np.ndarray, 
                               spectrograms: Optional[np.ndarray] = None,
                               confidence_threshold: float = 0.6) -> Dict:
        """
        Classify with confidence scoring and uncertainty handling.
        
        Args:
            X: Feature matrix
            spectrograms: Optional spectrograms
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Classification result with confidence and explanation
        """
        probabilities = self.predict_proba(X, spectrograms)
        
        # Get prediction and confidence
        prediction_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][prediction_idx]
        predicted_class = self.classes[prediction_idx]
        
        # Create detailed result
        result = {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                self.classes[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            },
            'is_confident': confidence >= confidence_threshold,
            'uncertainty_flag': confidence < confidence_threshold
        }
        
        # Add explanation based on confidence
        if confidence >= 0.8:
            result['explanation'] = f"High confidence prediction: {predicted_class}"
        elif confidence >= 0.6:
            result['explanation'] = f"Moderate confidence prediction: {predicted_class}"
        else:
            # Get top 2 predictions for uncertain cases
            top_2_idx = np.argsort(probabilities[0])[-2:][::-1]
            top_2_classes = [self.classes[i] for i in top_2_idx]
            top_2_probs = [probabilities[0][i] for i in top_2_idx]
            
            result['explanation'] = (
                f"Uncertain prediction. Most likely: {top_2_classes[0]} "
                f"({top_2_probs[0]:.2f}), also possible: {top_2_classes[1]} "
                f"({top_2_probs[1]:.2f})"
            )
            result['alternative_prediction'] = top_2_classes[1]
            result['alternative_confidence'] = float(top_2_probs[1])
        
        return result 