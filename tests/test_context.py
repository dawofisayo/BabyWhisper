"""Tests for context management and baby profile functionality."""

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

from context.baby_profile import BabyProfile
from context.context_manager import ContextManager


class TestBabyProfile(unittest.TestCase):
    """Test cases for BabyProfile class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.baby_profile = BabyProfile(
            baby_name="Test Baby",
            age_months=6
        )
    
    def test_initialization(self):
        """Test baby profile initialization."""
        self.assertEqual(self.baby_profile.baby_name, "Test Baby")
        self.assertEqual(self.baby_profile.age_months, 6)
        self.assertIsInstance(self.baby_profile.feeding_history, list)
        self.assertIsInstance(self.baby_profile.sleep_history, list)
        self.assertIsInstance(self.baby_profile.diaper_change_history, list)
    
    def test_update_feeding(self):
        """Test feeding time update."""
        feeding_time = datetime.now()
        self.baby_profile.update_feeding(feeding_time)
        
        self.assertEqual(self.baby_profile.last_feeding_time, feeding_time)
        self.assertIn(feeding_time, self.baby_profile.feeding_history)
    
    def test_update_sleep(self):
        """Test sleep time update."""
        sleep_start = datetime.now() - timedelta(hours=2)
        sleep_end = datetime.now()
        
        self.baby_profile.update_sleep(sleep_start, sleep_end)
        
        self.assertEqual(self.baby_profile.last_nap_time, sleep_end)
        self.assertIn({'start': sleep_start, 'end': sleep_end, 'is_nap': True}, 
                     self.baby_profile.sleep_history)
    
    def test_update_diaper_change(self):
        """Test diaper change time update."""
        change_time = datetime.now()
        self.baby_profile.update_diaper_change(change_time)
        
        self.assertEqual(self.baby_profile.last_diaper_change, change_time)
        self.assertIn(change_time, self.baby_profile.diaper_change_history)
    
    def test_get_time_since_feeding(self):
        """Test time since feeding calculation."""
        # Test with no feeding time
        self.assertIsNone(self.baby_profile.get_time_since_feeding())
        
        # Test with recent feeding
        feeding_time = datetime.now() - timedelta(hours=2)
        self.baby_profile.update_feeding(feeding_time)
        
        time_since = self.baby_profile.get_time_since_feeding()
        self.assertIsInstance(time_since, float)
        self.assertAlmostEqual(time_since, 2.0, delta=0.1)
    
    def test_get_time_since_feeding_just_fed(self):
        """Test time since feeding when just fed."""
        # Test with very recent feeding (should return 0.0)
        feeding_time = datetime.now() + timedelta(seconds=1)  # Slightly in future
        self.baby_profile.update_feeding(feeding_time)
        
        time_since = self.baby_profile.get_time_since_feeding()
        self.assertEqual(time_since, 0.0)
    
    def test_get_time_awake(self):
        """Test time awake calculation."""
        # Test with no wake time
        self.assertIsNone(self.baby_profile.get_time_awake())
        
        # Test with recent wake time
        wake_time = datetime.now() - timedelta(hours=1.5)
        self.baby_profile.last_wake_time = wake_time
        
        time_awake = self.baby_profile.get_time_awake()
        self.assertIsInstance(time_awake, float)
        self.assertAlmostEqual(time_awake, 1.5, delta=0.1)
    
    def test_is_likely_hungry(self):
        """Test hunger likelihood calculation."""
        # Test with no feeding time (should be True)
        self.assertTrue(self.baby_profile.is_likely_hungry())
        
        # Test with recent feeding (should be False)
        feeding_time = datetime.now() - timedelta(hours=1)
        self.baby_profile.update_feeding(feeding_time)
        self.assertFalse(self.baby_profile.is_likely_hungry())
        
        # Test with old feeding (should be True)
        feeding_time = datetime.now() - timedelta(hours=4)
        self.baby_profile.update_feeding(feeding_time)
        self.assertTrue(self.baby_profile.is_likely_hungry())
    
    def test_is_likely_tired(self):
        """Test tiredness likelihood calculation."""
        # Test with no wake time (should be False)
        self.assertFalse(self.baby_profile.is_likely_tired())
        
        # Test with recent wake time (should be False)
        wake_time = datetime.now() - timedelta(hours=1)
        self.baby_profile.last_wake_time = wake_time
        self.assertFalse(self.baby_profile.is_likely_tired())
        
        # Test with long awake time (should be True)
        wake_time = datetime.now() - timedelta(hours=3)
        self.baby_profile.last_wake_time = wake_time
        self.assertTrue(self.baby_profile.is_likely_tired())
    
    def test_is_likely_uncomfortable(self):
        """Test discomfort likelihood calculation."""
        # Test with no diaper change (should be True)
        self.assertTrue(self.baby_profile.is_likely_uncomfortable())
        
        # Test with recent diaper change (should be False)
        change_time = datetime.now() - timedelta(hours=1)
        self.baby_profile.update_diaper_change(change_time)
        self.assertFalse(self.baby_profile.is_likely_uncomfortable())
        
        # Test with old diaper change (should be True)
        change_time = datetime.now() - timedelta(hours=3)
        self.baby_profile.update_diaper_change(change_time)
        self.assertTrue(self.baby_profile.is_likely_uncomfortable())
    
    def test_get_context_probabilities(self):
        """Test context probability calculations."""
        # Set up a scenario where baby is hungry and tired
        feeding_time = datetime.now() - timedelta(hours=4)
        self.baby_profile.update_feeding(feeding_time)
        
        wake_time = datetime.now() - timedelta(hours=3)
        self.baby_profile.last_wake_time = wake_time
        
        probabilities = self.baby_profile.get_context_probabilities()
        
        self.assertIsInstance(probabilities, dict)
        self.assertIn('hunger', probabilities)
        self.assertIn('tiredness', probabilities)
        self.assertIn('discomfort', probabilities)
        self.assertIn('pain', probabilities)
        self.assertIn('normal', probabilities)
        
        # Should have higher probability for hunger and tiredness
        self.assertGreater(probabilities['hunger'], 0)
        self.assertGreater(probabilities['tiredness'], 0)
    
    def test_add_cry_record(self):
        """Test adding cry records for learning."""
        cry_type = 'hunger'
        confidence = 0.85
        resolution = 'feeding'
        resolution_time = 5.0
        
        self.baby_profile.add_cry_record(
            cry_type, confidence, resolution, resolution_time
        )
        
        self.assertEqual(len(self.baby_profile.cry_history), 1)
        record = self.baby_profile.cry_history[0]
        self.assertEqual(record['cry_type'], cry_type)
        self.assertEqual(record['confidence'], confidence)
        self.assertEqual(record['resolution'], resolution)
        self.assertEqual(record['resolution_time_minutes'], resolution_time)
    
    def test_save_and_load_profile(self):
        """Test profile saving and loading."""
        # Set up profile with some data
        self.baby_profile.update_feeding(datetime.now() - timedelta(hours=2))
        self.baby_profile.update_diaper_change(datetime.now() - timedelta(hours=1))
        self.baby_profile.add_cry_record('hunger', 0.8, 'feeding', 3.0)
        
        # Save profile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp_file.close()
        
        try:
            self.baby_profile.save_profile(temp_file.name)
            
            # Load profile
            loaded_profile = BabyProfile.load_profile(temp_file.name)
            
            # Check that data was preserved
            self.assertEqual(loaded_profile.baby_name, self.baby_profile.baby_name)
            self.assertEqual(loaded_profile.age_months, self.baby_profile.age_months)
            self.assertEqual(len(loaded_profile.cry_history), 
                           len(self.baby_profile.cry_history))
            
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)


class TestContextManager(unittest.TestCase):
    """Test cases for ContextManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context_manager = ContextManager()
        self.baby_profile = BabyProfile(
            baby_name="Test Baby",
            age_months=6
        )
    
    def test_initialization(self):
        """Test context manager initialization."""
        self.assertIsInstance(self.context_manager.active_profiles, dict)
        self.assertIsInstance(self.context_manager.context_weights, dict)
    
    def test_add_baby_profile(self):
        """Test adding baby profile."""
        profile_id = self.context_manager.add_baby_profile(self.baby_profile)
        
        self.assertIsInstance(profile_id, str)
        self.assertIn(profile_id, self.context_manager.active_profiles)
        self.assertEqual(self.context_manager.active_profiles[profile_id], 
                        self.baby_profile)
    
    def test_get_baby_profile(self):
        """Test retrieving baby profile."""
        profile_id = self.context_manager.add_baby_profile(self.baby_profile)
        retrieved_profile = self.context_manager.get_baby_profile(profile_id)
        
        self.assertEqual(retrieved_profile, self.baby_profile)
    
    def test_get_nonexistent_baby_profile(self):
        """Test retrieving nonexistent baby profile."""
        profile = self.context_manager.get_baby_profile("nonexistent_id")
        self.assertIsNone(profile)
    
    def test_apply_context_to_prediction(self):
        """Test context application to predictions."""
        # Set up baby profile with some context
        self.baby_profile.update_feeding(datetime.now() - timedelta(hours=4))
        self.baby_profile.last_wake_time = datetime.now() - timedelta(hours=3)
        
        profile_id = self.context_manager.add_baby_profile(self.baby_profile)
        
        # Create dummy prediction probabilities
        base_probabilities = np.array([0.2, 0.3, 0.4, 0.1, 0.0])  # 5 classes
        classes = ['hunger', 'tiredness', 'discomfort', 'pain', 'normal']
        
        result = self.context_manager.apply_context_to_prediction(
            base_probabilities, self.baby_profile, classes
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('base_prediction', result)
        self.assertIn('context_adjusted_prediction', result)
        self.assertIn('base_confidence', result)
        self.assertIn('context_adjusted_confidence', result)
        self.assertIn('explanation', result)
        
        # Check that probabilities are normalized
        adjusted_probs = result['context_adjusted_probabilities']
        total_prob = sum(adjusted_probs.values())
        self.assertAlmostEqual(total_prob, 1.0, delta=0.01)
    
    def test_generate_context_explanation(self):
        """Test context explanation generation."""
        # Set up baby profile with context
        self.baby_profile.update_feeding(datetime.now() - timedelta(hours=3))
        self.baby_profile.last_wake_time = datetime.now() - timedelta(hours=2)
        self.baby_profile.update_diaper_change(datetime.now() - timedelta(hours=1))
        
        base_prediction = 'hunger'
        adjusted_prediction = 'hunger'
        base_confidence = 0.8
        adjusted_confidence = 0.85
        context_factors = {'hunger': 0.3, 'tiredness': 0.2}
        
        explanation = self.context_manager._generate_context_explanation(
            self.baby_profile, base_prediction, adjusted_prediction,
            base_confidence, adjusted_confidence, context_factors
        )
        
        self.assertIsInstance(explanation, str)
        self.assertIn('Initial AI prediction', explanation)
        self.assertIn('Final prediction', explanation)
        self.assertIn('Context factors', explanation)
    
    def test_get_recommendations(self):
        """Test recommendation generation."""
        # Test hunger recommendations
        recommendations = self.context_manager._get_recommendations(
            'hunger', self.baby_profile
        )
        self.assertIsInstance(recommendations, str)
        self.assertIn('Try feeding', recommendations)
        
        # Test tiredness recommendations
        recommendations = self.context_manager._get_recommendations(
            'tiredness', self.baby_profile
        )
        self.assertIsInstance(recommendations, str)
        self.assertIn('Create calm environment', recommendations)
        
        # Test discomfort recommendations
        recommendations = self.context_manager._get_recommendations(
            'discomfort', self.baby_profile
        )
        self.assertIsInstance(recommendations, str)
        self.assertIn('Check diaper', recommendations)
    
    def test_learn_from_feedback(self):
        """Test learning from feedback."""
        profile_id = self.context_manager.add_baby_profile(self.baby_profile)
        
        # Add some feedback
        self.context_manager.learn_from_feedback(
            profile_id=profile_id,
            prediction='hunger',
            confidence=0.8,
            actual_cause='hunger',
            resolution_method='feeding',
            resolution_time_minutes=3.0
        )
        
        # Check that feedback was recorded
        self.assertEqual(len(self.baby_profile.cry_history), 1)
        record = self.baby_profile.cry_history[0]
        self.assertEqual(record['cry_type'], 'hunger')
        self.assertEqual(record['confidence'], 0.8)
        self.assertEqual(record['resolution'], 'feeding')
        self.assertEqual(record['resolution_time_minutes'], 3.0)


if __name__ == '__main__':
    unittest.main() 