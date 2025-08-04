"""Tests for audio processing functionality."""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import librosa
import soundfile as sf

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_processing.feature_extractor import AudioFeatureExtractor


class TestAudioFeatureExtractor(unittest.TestCase):
    """Test cases for AudioFeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AudioFeatureExtractor()
        
        # Create a dummy audio file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, 'test_audio.wav')
        
        # Generate a simple test audio signal
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a signal with multiple frequencies to test feature extraction
        test_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        sf.write(self.test_audio_path, test_signal, sample_rate)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_extract_features(self):
        """Test comprehensive feature extraction."""
        features = self.extractor.extract_features(self.test_audio_path)
        
        self.assertIsInstance(features, dict)
        self.assertIn('mfcc', features)
        self.assertIn('mfcc_mean', features)
        self.assertIn('mfcc_std', features)
        self.assertIn('spectral_centroid_mean', features)
        self.assertIn('spectral_centroid_std', features)
        self.assertIn('zcr_mean', features)
        self.assertIn('zcr_std', features)
        self.assertIn('rms_mean', features)
        self.assertIn('rms_std', features)
        self.assertIn('tempo', features)
        self.assertIn('chroma_mean', features)
        self.assertIn('chroma_std', features)
        self.assertIn('mel_spectrogram', features)
        self.assertIn('mel_spec_mean', features)
        self.assertIn('mel_spec_std', features)
        self.assertIn('f0_mean', features)
        self.assertIn('f0_std', features)
        self.assertIn('spectral_bandwidth_mean', features)
        self.assertIn('spectral_bandwidth_std', features)
        self.assertIn('duration', features)
        self.assertIn('energy', features)
    
    def test_extract_feature_vector(self):
        """Test complete feature vector extraction."""
        feature_vector = self.extractor.extract_feature_vector(self.test_audio_path)
        
        self.assertIsInstance(feature_vector, np.ndarray)
        self.assertEqual(feature_vector.ndim, 1)  # Should be 1D
        self.assertGreater(len(feature_vector), 0)  # Should have features
        self.assertIsInstance(feature_vector[0], (int, float))  # Should be numeric
    
    def test_feature_vector_consistency(self):
        """Test that feature vectors are consistent for same audio."""
        vector1 = self.extractor.extract_feature_vector(self.test_audio_path)
        vector2 = self.extractor.extract_feature_vector(self.test_audio_path)
        
        np.testing.assert_array_almost_equal(vector1, vector2, decimal=5)
    
    def test_invalid_audio_file(self):
        """Test handling of invalid audio file."""
        invalid_path = "nonexistent_file.wav"
        
        with self.assertRaises(Exception):
            self.extractor.extract_feature_vector(invalid_path)
    
    def test_get_feature_names(self):
        """Test feature names retrieval."""
        feature_names = self.extractor.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertIsInstance(feature_names[0], str)


if __name__ == '__main__':
    unittest.main() 