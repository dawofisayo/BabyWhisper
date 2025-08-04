"""Main application interface for BabyWhisper - AI-powered baby cry classification."""

import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

from .audio_processing import AudioFeatureExtractor, AudioPreprocessor
from .models import BabyCryClassifier, ModelTrainer
from .context import BabyProfile, ContextManager
from .utils import DataLoader, ModelEvaluator


class BabyWhisperClassifier:
    """Main interface for the BabyWhisper baby cry classification system."""
    
    def __init__(self):
        """Initialize BabyWhisper classifier with all components."""
        # Core components
        self.feature_extractor = AudioFeatureExtractor()
        self.preprocessor = AudioPreprocessor()
        self.classifier = None
        self.context_manager = ContextManager()
        
        # For real-trained ensemble model
        self.ensemble_model = None
        self.scaler = None
        self.label_encoder = None
        self.model_loaded = False
        
        # Define standard classes for baby cries
        self.classes = ['hunger', 'pain', 'discomfort', 'tiredness', 'normal']
        
        print("BabyWhisper Classifier initialized!")
        print("Ready to analyze baby cries with AI-powered insights.")
    
    def _setup_ensemble_classifier(self):
        """Setup classifier interface for ensemble model."""
        if self.ensemble_model and self.scaler and self.label_encoder:
            # Create a wrapper for the ensemble model
            class EnsembleWrapper:
                def __init__(self, ensemble, scaler, label_encoder):
                    self.ensemble = ensemble
                    self.scaler = scaler
                    self.label_encoder = label_encoder
                    self.classes = label_encoder.classes_
                    self.model_type = "ensemble"  # Add missing attribute
                
                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    predictions_encoded = self.ensemble.predict(X_scaled)
                    return self.label_encoder.inverse_transform(predictions_encoded)
                
                def predict_proba(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.ensemble.predict_proba(X_scaled)
                
                def classify_with_confidence(self, X):
                    predictions = self.predict(X)
                    probabilities = self.predict_proba(X)
                    
                    result = {
                        'prediction': predictions[0],
                        'confidence': float(np.max(probabilities[0])),
                        'all_probabilities': {
                            class_name: float(prob) 
                            for class_name, prob in zip(self.classes, probabilities[0])
                        }
                    }
                    return result
            
            # Set up the classifier interface
            self.classifier = EnsembleWrapper(self.ensemble_model, self.scaler, self.label_encoder)
            self.classes = self.classifier.classes
            self.model_loaded = True
            print("✅ Ensemble model wrapper created successfully!")
    
    def classify_cry(self, 
                    audio_path: str,
                    baby_profile: Optional[BabyProfile] = None,
                    return_detailed: bool = True) -> Dict:
        """
        Classify a baby cry from audio file.
        
        Args:
            audio_path: Path to audio file
            baby_profile: Optional baby profile for context-aware prediction
            return_detailed: Whether to return detailed analysis
            
        Returns:
            Dictionary with classification results
        """
        if self.classifier is None:
            raise ValueError("No model loaded. Please train a model or load a pre-trained one.")
        
        try:
            # Extract features
            features = self.feature_extractor.extract_feature_vector(audio_path)
            
            # Get base prediction from the classifier
            base_result = self.classifier.classify_with_confidence(features.reshape(1, -1))
            
            # Apply context if a baby profile is provided
            if baby_profile:
                context_result = self.context_manager.apply_context_to_prediction(
                    base_result['all_probabilities'], 
                    baby_profile, 
                    self.classes
                )
                
                # Merge base result with context result
                final_result = {**base_result, **context_result}
                
                # Generate recommendations based on the final prediction and context
                final_result['recommendations'] = self._get_specific_recommendations(
                    final_result['prediction'], baby_profile
                )
                
                # Include explanation from context result
                if 'explanation' in context_result:
                    final_result['explanation'] = context_result['explanation']
                
                return final_result
            
            # If no baby profile, return base result with general recommendations
            base_result['recommendations'] = self._get_general_recommendations(base_result['prediction'])
            
            # Add general explanation
            base_result['explanation'] = f"AI analysis based on audio features only. Prediction: {base_result['prediction']} with {base_result['confidence']:.2f} confidence. For more personalized insights, create a baby profile with feeding, sleep, and care history."
            
            return base_result
            
        except Exception as e:
            print(f"Error classifying cry: {e}")
            raise e
    
    def create_baby_profile(self, 
                           baby_name: str,
                           age_months: int = 6,
                           **kwargs) -> str:
        """
        Create a new baby profile.
        
        Args:
            baby_name: Name of the baby
            age_months: Age in months
            **kwargs: Additional profile parameters
            
        Returns:
            Profile ID for future reference
        """
        profile = BabyProfile(
            baby_name=baby_name,
            age_months=age_months,
            **kwargs
        )
        
        profile_id = self.context_manager.add_baby_profile(profile)
        print(f"Created baby profile for {baby_name} (ID: {profile_id})")
        
        return profile_id
    
    def update_baby_context(self, 
                           profile_id: str,
                           feeding_time: Optional[datetime] = None,
                           nap_time: Optional[datetime] = None,
                           diaper_change_time: Optional[datetime] = None,
                           **kwargs):
        """
        Update baby's contextual information.
        
        Args:
            profile_id: Baby profile ID
            feeding_time: Last feeding time
            nap_time: Last nap time
            diaper_change_time: Last diaper change time
            **kwargs: Additional context updates
        """
        profile = self.context_manager.get_baby_profile(profile_id)
        if not profile:
            raise ValueError(f"Baby profile {profile_id} not found")
        
        if feeding_time:
            profile.update_feeding(feeding_time)
        if nap_time:
            profile.update_sleep(sleep_start=nap_time)
        if diaper_change_time:
            profile.update_diaper_change(diaper_change_time)
        
        print(f"Updated context for {profile.baby_name}")
    
    def get_baby_insights(self, profile_id: str) -> Dict:
        """
        Get personalized insights for a baby.
        
        Args:
            profile_id: Baby profile ID
            
        Returns:
            Dictionary with insights and recommendations
        """
        return self.context_manager.get_personalized_insights(profile_id)
    
    def provide_feedback(self, 
                        profile_id: str,
                        predicted_cause: str,
                        actual_cause: str,
                        resolution_method: str,
                        resolution_time_minutes: float = 5.0):
        """
        Provide feedback to improve future predictions.
        
        Args:
            profile_id: Baby profile ID
            predicted_cause: What the system predicted
            actual_cause: What actually caused the crying
            resolution_method: How the issue was resolved
            resolution_time_minutes: Time it took to resolve
        """
        self.context_manager.learn_from_feedback(
            profile_id=profile_id,
            prediction=predicted_cause,
            confidence=0.8,  # Default confidence
            actual_cause=actual_cause,
            resolution_method=resolution_method,
            resolution_time_minutes=resolution_time_minutes
        )
        
        print(f"Feedback recorded. This will help improve future predictions!")
    
    def train_new_model(self, 
                       dataset_path: Optional[str] = None,
                       model_type: str = 'ensemble') -> Dict:
        """
        Train a new classification model.
        
        Args:
            dataset_path: Path to training dataset (optional)
            model_type: Type of model to train
            
        Returns:
            Training results
        """
        print("Starting model training...")
        
        trainer = ModelTrainer()
        
        # Use real dataset
        if dataset_path is None:
            # Use default Donate-a-Cry dataset
            print("Using default Donate-a-Cry dataset for training...")
            classifier = trainer.quick_demo_training()
        else:
            # Load real dataset from specified path
            data_loader = DataLoader()
            file_paths, labels = data_loader.load_dataset_from_directory(dataset_path)
            
            if not file_paths:
                raise ValueError(f"No audio files found in {dataset_path}")
            
            # Extract features
            print("Extracting features from audio files...")
            features = []
            valid_labels = []
            
            for i, file_path in enumerate(file_paths):
                try:
                    feature_vector = self.feature_extractor.extract_feature_vector(file_path)
                    features.append(feature_vector)
                    valid_labels.append(labels[i])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            if not features:
                raise ValueError("No valid features extracted from dataset")
            
            features = np.array(features)
            valid_labels = np.array(valid_labels)
            
            # Prepare data and train
            data_splits = trainer.prepare_data(features, valid_labels)
            classifier = trainer.train_model(data_splits, model_type=model_type)
            results = trainer.evaluate_model(data_splits)
            trainer.save_training_results(results)
        
        # Load the trained model
        self.classifier = classifier
        
        print("Model training completed!")
        return {
            'model_type': model_type,
            'classes': classifier.classes,
            'training_completed': True
        }
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model files
        """
        self.classifier = BabyCryClassifier()
        self.classifier.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str):
        """
        Save the current model.
        
        Args:
            model_path: Path to save the model
        """
        if self.classifier is None:
            raise ValueError("No model to save. Train a model first.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.classifier.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    def analyze_audio_features(self, audio_path: str, save_visualization: bool = True) -> Dict:
        """
        Analyze and visualize audio features.
        
        Args:
            audio_path: Path to audio file
            save_visualization: Whether to save feature visualizations
            
        Returns:
            Dictionary with audio analysis
        """
        try:
            # Extract comprehensive features
            features = self.feature_extractor.extract_features(audio_path)
            
            # Create visualizations
            if save_visualization:
                viz_path = os.path.join("plots", f"audio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                os.makedirs("plots", exist_ok=True)
                self.feature_extractor.visualize_features(audio_path, save_path=viz_path)
                print(f"Audio analysis visualization saved to {viz_path}")
            
            # Summary statistics
            analysis = {
                'file_path': audio_path,
                'duration_seconds': features['duration'],
                'energy_level': features['energy'],
                'dominant_frequency': features['f0_mean'] if features['f0_mean'] > 0 else None,
                'spectral_characteristics': {
                    'centroid_mean': features['spectral_centroid_mean'],
                    'rolloff_mean': features['spectral_rolloff_mean'],
                    'bandwidth_mean': features['spectral_bandwidth_mean']
                },
                'temporal_characteristics': {
                    'zero_crossing_rate': features['zcr_mean'],
                    'rms_energy': features['rms_mean'],
                    'tempo': features['tempo']
                }
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f"Error analyzing audio: {str(e)}"}
    
    def _get_specific_recommendations(self, prediction: str, profile: BabyProfile) -> List[str]:
        """Get specific recommendations based on prediction and baby profile."""
        recommendations = []
        
        if prediction == 'hunger':
            if profile.is_likely_hungry():
                recommendations.append("Baby is likely hungry - try feeding")
                recommendations.append("Check for hunger cues: rooting, sucking motions, fussiness")
            else:
                recommendations.append("Consider if baby might be thirsty or wanting comfort sucking")
        
        elif prediction == 'tiredness':
            if profile.is_likely_tired():
                recommendations.append("Create a calm, dark environment for sleep")
                if profile.comfort_preferences.get('responds_to_rocking'):
                    recommendations.append("Try gentle rocking or swaying")
                if profile.comfort_preferences.get('responds_to_white_noise'):
                    recommendations.append("Use white noise or soft music")
                if profile.comfort_preferences.get('prefers_swaddling'):
                    recommendations.append("Consider swaddling for comfort")
            else:
                recommendations.append("Look for other sleep cues: yawning, rubbing eyes, fussiness")
        
        elif prediction == 'discomfort':
            recommendations.append("Check diaper and change if needed")
            recommendations.append("Verify room temperature and clothing comfort")
            recommendations.append("Look for signs of gas or need to burp")
            recommendations.append("Check for hair wrapped around fingers/toes")
        
        elif prediction == 'pain':
            recommendations.append("Check for signs of illness: fever, rash, unusual behavior")
            recommendations.append("Look for sources of pain: tight clothing, scratches, diaper rash")
            recommendations.append("Consider consulting healthcare provider if crying persists")
            recommendations.append("Monitor for other concerning symptoms")
        
        else:  # normal or uncertain
            recommendations.append("Baby may just need attention or comfort")
            recommendations.append("Try gentle soothing: talking, singing, or holding")
        
        return recommendations
    
    def _get_general_recommendations(self, prediction: str) -> List[str]:
        """Get general recommendations without baby profile context."""
        recommendations = []
        
        if prediction == 'hunger':
            recommendations.append("Try feeding the baby")
            recommendations.append("Check for hunger cues")
        elif prediction == 'tiredness':
            recommendations.append("Create a calm environment")
            recommendations.append("Try gentle rocking or soothing sounds")
        elif prediction == 'discomfort':
            recommendations.append("Check diaper, temperature, and clothing")
            recommendations.append("Look for sources of discomfort")
        elif prediction == 'pain':
            recommendations.append("Check for signs of illness or injury")
            recommendations.append("Consider consulting healthcare provider")
        else:
            recommendations.append("Try general soothing techniques")
        
        return recommendations
    
    def get_system_status(self) -> Dict:
        """Get current system status and capabilities."""
        return {
            'model_loaded': self.classifier is not None,
            'model_type': self.classifier.model_type if self.classifier else None,
            'classes': list(self.classifier.classes) if self.classifier else [],
            'active_baby_profiles': len(self.context_manager.active_profiles),
            'baby_profiles': list(self.context_manager.active_profiles.keys()),
            'feature_extractor_ready': True,
            'preprocessor_ready': True,
            'context_manager_ready': True
        }


def create_demo_setup() -> BabyWhisperClassifier:
    """
    Create a demo setup with a trained model and sample baby profile.
    
    Returns:
        Configured BabyWhisper classifier ready for demonstration
    """
    print("Setting up BabyWhisper demo...")
    
    # Initialize classifier
    classifier = BabyWhisperClassifier()
    
    # Train a demo model with real data
    classifier.train_new_model()
    
    # Create a sample baby profile
    profile_id = classifier.create_baby_profile(
        baby_name="Demo Baby",
        age_months=4
    )
    
    # Set some sample context
    now = datetime.now()
    classifier.update_baby_context(
        profile_id=profile_id,
        feeding_time=now - timedelta(hours=2.5),
        nap_time=now - timedelta(hours=1.5),
        diaper_change_time=now - timedelta(hours=1)
    )
    
    print("Demo setup complete!")
    print(f"Sample baby profile created: {profile_id}")
    print("You can now classify baby cries with context-aware predictions!")
    
    return classifier


if __name__ == "__main__":
    # Demo usage
    demo_classifier = create_demo_setup()
    print("\nBabyWhisper is ready!")
    print("Use demo_classifier.classify_cry(audio_path) to analyze baby cries.")
    print("Use demo_classifier.get_system_status() to check system status.") 