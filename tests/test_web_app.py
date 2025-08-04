"""Tests for web application API functionality."""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Flask app
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'web_app', 'backend'))
from app import app


class TestWebAppAPI(unittest.TestCase):
    """Test cases for web application API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a temporary audio file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, 'test_audio.wav')
        
        # Create a simple test audio file
        import numpy as np
        import soundfile as sf
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_signal = np.sin(2 * np.pi * 440 * t)
        sf.write(self.test_audio_path, test_signal, sample_rate)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.app.get('/api/health')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_system_status_endpoint(self):
        """Test system status endpoint."""
        response = self.app.get('/api/system-status')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('model_info', data)
        self.assertIn('system_info', data)
    
    def test_get_babies_endpoint(self):
        """Test getting baby profiles endpoint."""
        response = self.app.get('/api/babies')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
    
    def test_create_baby_endpoint(self):
        """Test creating a new baby profile."""
        baby_data = {
            'name': 'Test Baby',
            'age_months': 6,
            'birth_date': '2024-01-01'
        }
        
        response = self.app.post('/api/babies',
                               data=json.dumps(baby_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertIn('id', data)
        self.assertEqual(data['name'], 'Test Baby')
    
    def test_update_baby_context_endpoint(self):
        """Test updating baby context."""
        # First create a baby
        baby_data = {
            'name': 'Test Baby',
            'age_months': 6
        }
        create_response = self.app.post('/api/babies',
                                      data=json.dumps(baby_data),
                                      content_type='application/json')
        baby_id = json.loads(create_response.data)['id']
        
        # Update context
        context_data = {
            'feeding_time': datetime.now().isoformat(),
            'sleep_time': datetime.now().isoformat(),
            'diaper_time': datetime.now().isoformat()
        }
        
        response = self.app.put(f'/api/babies/{baby_id}',
                               data=json.dumps(context_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
    
    def test_classify_audio_endpoint(self):
        """Test audio classification endpoint."""
        with open(self.test_audio_path, 'rb') as audio_file:
            response = self.app.post('/api/classify-audio',
                                   data={'audio': (audio_file, 'test_audio.wav')},
                                   content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertIn('explanation', data)
    
    def test_classify_audio_with_baby_profile(self):
        """Test audio classification with baby profile."""
        # First create a baby
        baby_data = {
            'name': 'Test Baby',
            'age_months': 6
        }
        create_response = self.app.post('/api/babies',
                                      data=json.dumps(baby_data),
                                      content_type='application/json')
        baby_id = json.loads(create_response.data)['id']
        
        # Classify audio with baby profile
        with open(self.test_audio_path, 'rb') as audio_file:
            response = self.app.post('/api/classify-audio',
                                   data={
                                       'audio': (audio_file, 'test_audio.wav'),
                                       'baby_id': baby_id
                                   },
                                   content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertIn('explanation', data)
    
    def test_feedback_endpoint(self):
        """Test feedback endpoint."""
        feedback_data = {
            'prediction': 'hunger',
            'confidence': 0.85,
            'actual_cause': 'hunger',
            'resolution_method': 'feeding',
            'resolution_time_minutes': 3.0
        }
        
        response = self.app.post('/api/feedback',
                               data=json.dumps(feedback_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
    
    def test_invalid_audio_file(self):
        """Test handling of invalid audio file."""
        # Create an invalid audio file
        invalid_audio_path = os.path.join(self.temp_dir, 'invalid.txt')
        with open(invalid_audio_path, 'w') as f:
            f.write('This is not an audio file')
        
        with open(invalid_audio_path, 'rb') as audio_file:
            response = self.app.post('/api/classify-audio',
                                   data={'audio': (audio_file, 'invalid.txt')},
                                   content_type='multipart/form-data')
        
        # Should handle gracefully
        self.assertIn(response.status_code, [400, 500])
        
        # Clean up
        os.remove(invalid_audio_path)
    
    def test_missing_audio_file(self):
        """Test handling of missing audio file."""
        response = self.app.post('/api/classify-audio',
                               data={},
                               content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
    
    def test_invalid_baby_id(self):
        """Test handling of invalid baby ID."""
        response = self.app.put('/api/babies/invalid-id',
                               data=json.dumps({'feeding_time': datetime.now().isoformat()}),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 404)
    
    def test_baby_insights_endpoint(self):
        """Test baby insights endpoint."""
        # First create a baby
        baby_data = {
            'name': 'Test Baby',
            'age_months': 6
        }
        create_response = self.app.post('/api/babies',
                                      data=json.dumps(baby_data),
                                      content_type='application/json')
        baby_id = json.loads(create_response.data)['id']
        
        # Get insights
        response = self.app.get(f'/api/babies/{baby_id}/insights')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('current_status', data)
        self.assertIn('recommendations', data)
        self.assertIn('patterns', data)
    
    def test_baby_insights_invalid_id(self):
        """Test baby insights with invalid ID."""
        response = self.app.get('/api/babies/invalid-id/insights')
        
        self.assertEqual(response.status_code, 404)


class TestWebAppIntegration(unittest.TestCase):
    """Integration tests for web application."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_complete_workflow(self):
        """Test complete workflow: create baby, update context, classify audio."""
        # 1. Create baby profile
        baby_data = {
            'name': 'Integration Test Baby',
            'age_months': 8
        }
        create_response = self.app.post('/api/babies',
                                      data=json.dumps(baby_data),
                                      content_type='application/json')
        self.assertEqual(create_response.status_code, 201)
        baby_id = json.loads(create_response.data)['id']
        
        # 2. Update baby context
        context_data = {
            'feeding_time': (datetime.now() - timedelta(hours=3)).isoformat(),
            'sleep_time': (datetime.now() - timedelta(hours=2)).isoformat(),
            'diaper_time': (datetime.now() - timedelta(hours=1)).isoformat()
        }
        update_response = self.app.put(f'/api/babies/{baby_id}',
                                     data=json.dumps(context_data),
                                     content_type='application/json')
        self.assertEqual(update_response.status_code, 200)
        
        # 3. Get baby insights
        insights_response = self.app.get(f'/api/babies/{baby_id}/insights')
        self.assertEqual(insights_response.status_code, 200)
        insights_data = json.loads(insights_response.data)
        self.assertIn('current_status', insights_data)
        
        # 4. Provide feedback
        feedback_data = {
            'prediction': 'hunger',
            'confidence': 0.85,
            'actual_cause': 'hunger',
            'resolution_method': 'feeding',
            'resolution_time_minutes': 3.0
        }
        feedback_response = self.app.post('/api/feedback',
                                        data=json.dumps(feedback_data),
                                        content_type='application/json')
        self.assertEqual(feedback_response.status_code, 200)
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test invalid JSON
        response = self.app.post('/api/babies',
                               data='invalid json',
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test missing required fields
        response = self.app.post('/api/babies',
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test invalid date format
        baby_data = {
            'name': 'Test Baby',
            'age_months': 6,
            'birth_date': 'invalid-date'
        }
        response = self.app.post('/api/babies',
                               data=json.dumps(baby_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main() 