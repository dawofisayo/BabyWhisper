"""Integration tests for the complete BabyWhisper system."""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.classifier import BabyCryClassifier
from context.baby_profile import BabyProfile
from context.context_manager import ContextManager
from audio_processing.feature_extractor import AudioFeatureExtractor


class TestCompleteSystemIntegration(unittest.TestCase):
    """Integration tests for the complete BabyWhisper system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = BabyCryClassifier()
        self.context_manager = ContextManager()
        self.feature_extractor = AudioFeatureExtractor()
        
        # Create test baby profile
        self.baby_profile = BabyProfile(
            baby_name="Integration Test Baby",
            age_months=6
        )
        
        # Create test audio file
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, 'test_audio.wav')
        
        # Generate test audio
        import soundfile as sf
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        sf.write(self.test_audio_path, test_signal, sample_rate)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_complete_audio_processing_pipeline(self):
        """Test complete audio processing pipeline."""
        # 1. Extract features from audio
        features = self.feature_extractor.extract_feature_vector(self.test_audio_path)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)
        self.assertGreater(len(features), 0)
        
        # 2. Verify feature consistency
        features2 = self.feature_extractor.extract_feature_vector(self.test_audio_path)
        np.testing.assert_array_almost_equal(features, features2, decimal=5)
    
    def test_complete_classification_pipeline(self):
        """Test complete classification pipeline with context."""
        # 1. Set up baby profile with context
        self.baby_profile.update_feeding(datetime.now() - timedelta(hours=3))
        self.baby_profile.last_wake_time = datetime.now() - timedelta(hours=2)
        self.baby_profile.update_diaper_change(datetime.now() - timedelta(hours=1))
        
        # 2. Add baby to context manager
        profile_id = self.context_manager.add_baby_profile(self.baby_profile)
        
        # 3. Mock the classification to avoid model loading issues
        with patch.object(self.classifier, 'predict_proba') as mock_predict:
            mock_predict.return_value = np.array([[0.85, 0.10, 0.05, 0.0, 0.0]])
            
            # 4. Extract features and classify
            features = self.feature_extractor.extract_feature_vector(self.test_audio_path)
            result = self.classifier.classify_with_confidence(features)
            
            # 5. Verify results
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertIn('probabilities', result)
    
    def test_context_aware_prediction_workflow(self):
        """Test context-aware prediction workflow."""
        # 1. Create multiple baby profiles with different contexts
        hungry_baby = BabyProfile("Hungry Baby", 6)
        hungry_baby.update_feeding(datetime.now() - timedelta(hours=4))
        
        tired_baby = BabyProfile("Tired Baby", 6)
        tired_baby.last_wake_time = datetime.now() - timedelta(hours=3)
        
        comfortable_baby = BabyProfile("Comfortable Baby", 6)
        comfortable_baby.update_feeding(datetime.now() - timedelta(hours=1))
        comfortable_baby.update_diaper_change(datetime.now() - timedelta(hours=0.5))
        
        # 2. Add all babies to context manager
        hungry_id = self.context_manager.add_baby_profile(hungry_baby)
        tired_id = self.context_manager.add_baby_profile(tired_baby)
        comfortable_id = self.context_manager.add_baby_profile(comfortable_baby)
        
        # 3. Test context-aware predictions
        base_probabilities = np.array([0.2, 0.3, 0.4, 0.1, 0.0])  # 5 classes
        classes = ['hunger', 'tiredness', 'discomfort', 'pain', 'normal']
        
        # Test with hungry baby
        hungry_result = self.context_manager.apply_context_to_prediction(
            base_probabilities, hungry_baby, classes
        )
        self.assertIn('hunger', hungry_result['context_adjusted_probabilities'])
        
        # Test with tired baby
        tired_result = self.context_manager.apply_context_to_prediction(
            base_probabilities, tired_baby, classes
        )
        self.assertIn('tiredness', tired_result['context_adjusted_probabilities'])
        
        # Test with comfortable baby
        comfortable_result = self.context_manager.apply_context_to_prediction(
            base_probabilities, comfortable_baby, classes
        )
        self.assertIn('normal', comfortable_result['context_adjusted_probabilities'])
    
    def test_learning_from_feedback_workflow(self):
        """Test learning from feedback workflow."""
        # 1. Set up baby profile
        profile_id = self.context_manager.add_baby_profile(self.baby_profile)
        
        # 2. Simulate multiple cry events with feedback
        cry_events = [
            {
                'prediction': 'hunger',
                'confidence': 0.85,
                'actual_cause': 'hunger',
                'resolution': 'feeding',
                'resolution_time': 3.0
            },
            {
                'prediction': 'tiredness',
                'confidence': 0.75,
                'actual_cause': 'discomfort',
                'resolution': 'diaper_change',
                'resolution_time': 2.0
            },
            {
                'prediction': 'hunger',
                'confidence': 0.90,
                'actual_cause': 'hunger',
                'resolution': 'feeding',
                'resolution_time': 4.0
            }
        ]
        
        # 3. Add feedback for each event
        for event in cry_events:
            self.context_manager.learn_from_feedback(
                profile_id=profile_id,
                prediction=event['prediction'],
                confidence=event['confidence'],
                actual_cause=event['actual_cause'],
                resolution_method=event['resolution'],
                resolution_time_minutes=event['resolution_time']
            )
        
        # 4. Verify learning occurred
        self.assertEqual(len(self.baby_profile.cry_history), 3)
        
        # 5. Check that patterns can be analyzed
        patterns = self.baby_profile.analyze_patterns()
        self.assertIsInstance(patterns, dict)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        # 1. Test invalid audio file handling
        invalid_audio_path = os.path.join(self.temp_dir, 'invalid.wav')
        with open(invalid_audio_path, 'w') as f:
            f.write('This is not an audio file')
        
        with self.assertRaises(Exception):
            self.feature_extractor.extract_feature_vector(invalid_audio_path)
        
        # 2. Test missing baby profile handling
        nonexistent_profile = self.context_manager.get_baby_profile("nonexistent_id")
        self.assertIsNone(nonexistent_profile)
        
        # Clean up
        os.remove(invalid_audio_path)
    
    def test_performance_and_scalability(self):
        """Test performance and scalability aspects."""
        # 1. Test feature extraction performance
        import time
        
        start_time = time.time()
        features = self.feature_extractor.extract_feature_vector(self.test_audio_path)
        extraction_time = time.time() - start_time
        
        # Feature extraction should be reasonably fast (< 5 seconds)
        self.assertLess(extraction_time, 5.0)
        
        # 2. Test multiple baby profiles
        profiles = []
        for i in range(10):
            profile = BabyProfile(f"Baby {i}", 6)
            profiles.append(profile)
            self.context_manager.add_baby_profile(profile)
        
        # Should handle multiple profiles without issues
        self.assertEqual(len(self.context_manager.active_profiles), 10)
        
        # 3. Test context calculation performance
        start_time = time.time()
        for profile in profiles:
            profile.update_feeding(datetime.now() - timedelta(hours=2))
            profile.get_time_since_feeding()
        context_time = time.time() - start_time
        
        # Context calculations should be very fast (< 1 second for 10 profiles)
        self.assertLess(context_time, 1.0)
    
    def test_data_persistence_and_recovery(self):
        """Test data persistence and recovery."""
        # 1. Set up baby profile with data
        self.baby_profile.update_feeding(datetime.now() - timedelta(hours=2))
        self.baby_profile.update_diaper_change(datetime.now() - timedelta(hours=1))
        self.baby_profile.add_cry_record('hunger', 0.8, 'feeding', 3.0)
        
        # 2. Save profile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp_file.close()
        
        try:
            self.baby_profile.save_profile(temp_file.name)
            
            # 3. Load profile
            loaded_profile = BabyProfile.load_profile(temp_file.name)
            
            # 4. Verify data integrity
            self.assertEqual(loaded_profile.baby_name, self.baby_profile.baby_name)
            self.assertEqual(loaded_profile.age_months, self.baby_profile.age_months)
            self.assertEqual(len(loaded_profile.cry_history), 
                           len(self.baby_profile.cry_history))
            
            # 5. Test context manager persistence
            profile_id = self.context_manager.add_baby_profile(loaded_profile)
            retrieved_profile = self.context_manager.get_baby_profile(profile_id)
            
            self.assertEqual(retrieved_profile.baby_name, loaded_profile.baby_name)
            
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
    
    def test_real_world_scenarios(self):
        """Test real-world usage scenarios."""
        # Scenario 1: Newborn baby (0-3 months)
        newborn = BabyProfile("Newborn", 1)
        newborn.typical_feeding_interval_hours = 2.0
        newborn.typical_sleep_interval_hours = 1.5
        
        # Test newborn context
        newborn.update_feeding(datetime.now() - timedelta(hours=3))
        self.assertTrue(newborn.is_likely_hungry())
        
        # Scenario 2: Older baby (6-12 months)
        older_baby = BabyProfile("Older Baby", 9)
        older_baby.typical_feeding_interval_hours = 4.0
        older_baby.typical_sleep_interval_hours = 3.0
        
        # Test older baby context
        older_baby.update_feeding(datetime.now() - timedelta(hours=2))
        self.assertFalse(older_baby.is_likely_hungry())
        
        # Scenario 3: Sick baby
        sick_baby = BabyProfile("Sick Baby", 6)
        sick_baby.update_feeding(datetime.now() - timedelta(hours=1))
        sick_baby.update_diaper_change(datetime.now() - timedelta(hours=0.5))
        
        # Even with recent care, sick baby might still be uncomfortable
        # This tests the system's ability to handle edge cases
        context_probs = sick_baby.get_context_probabilities()
        self.assertIsInstance(context_probs, dict)


if __name__ == '__main__':
    unittest.main() 