#!/usr/bin/env python3
"""
BabyWhisper Web Dashboard - Flask Backend API
Provides REST API endpoints for the BabyWhisper web interface.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
import logging
import traceback

# Add the parent directory to the path to import BabyWhisper modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.main import BabyWhisperClassifier
    from src.context import BabyProfile
except ImportError as e:
    print(f"Error importing BabyWhisper modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global BabyWhisper instance
baby_whisper = None
baby_profiles = {}  # Store baby profiles in memory (in production, use a database)

def initialize_baby_whisper():
    """Initialize the BabyWhisper system with real-trained models."""
    global baby_whisper
    try:
        baby_whisper = BabyWhisperClassifier()
        
        # Try to load the real-trained ensemble model
        ensemble_model_path = os.path.join('..', '..', 'models', 'ensemble_model.pkl')
        scaler_path = os.path.join('..', '..', 'models', 'scaler.pkl')
        encoder_path = os.path.join('..', '..', 'models', 'label_encoder.pkl')
        
        # Check if real-trained models exist
        if all(os.path.exists(path) for path in [ensemble_model_path, scaler_path, encoder_path]):
            logger.info("Loading real-trained ensemble model...")
            try:
                import joblib
                # Load the real ensemble model components
                baby_whisper.ensemble_model = joblib.load(ensemble_model_path)
                baby_whisper.scaler = joblib.load(scaler_path)
                baby_whisper.label_encoder = joblib.load(encoder_path)
                
                # Setup the classifier interface for ensemble model
                baby_whisper._setup_ensemble_classifier()
                
                logger.info("✅ Real-trained ensemble model loaded successfully!")
                logger.info("🎯 Model trained on 457 real baby cry recordings")
                logger.info("🎯 Test accuracy: 83.7%")
                return True
            except Exception as e:
                logger.error(f"Failed to load real-trained model: {e}")
                logger.error(traceback.format_exc())
        
        # Fallback: Try old model format
        old_model_path = os.path.join('..', '..', 'models', 'baby_cry_classifier')
        if baby_whisper.load_model(old_model_path):
            logger.info("⚠️  Loaded fallback model (not real-trained)")
            return True
        
        # Last resort: Train new model
        logger.info("No models found, training new model...")
        baby_whisper.train_new_model(use_synthetic_data=False)  # Try real data first
        logger.info("Model training completed!")
        
        logger.info("BabyWhisper system initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize BabyWhisper: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "BabyWhisper API",
        "timestamp": datetime.now().isoformat(),
        "baby_whisper_ready": baby_whisper is not None
    })

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """Get BabyWhisper system status."""
    global baby_whisper
    
    # Initialize if not already done
    if not baby_whisper:
        logger.info("Lazy initialization of BabyWhisper system...")
        if not initialize_baby_whisper():
            return jsonify({"error": "Failed to initialize BabyWhisper system"}), 500
    
    try:
        status = baby_whisper.get_system_status()
        status['active_profiles'] = len(baby_profiles)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/babies', methods=['GET'])
def get_babies():
    """Get all baby profiles."""
    try:
        babies_list = []
        for profile_id, profile in baby_profiles.items():
            babies_list.append({
                "id": profile_id,
                "name": profile.baby_name,
                "age_months": profile.age_months,
                "birth_date": profile.birth_date.isoformat() if profile.birth_date else None,
                "last_feeding": profile.last_feeding_time.isoformat() if profile.last_feeding_time else None,
                "last_sleep": profile.last_nap_time.isoformat() if profile.last_nap_time else None,
                "last_diaper_change": profile.last_diaper_change.isoformat() if profile.last_diaper_change else None
            })
        return jsonify(babies_list)
    except Exception as e:
        logger.error(f"Error getting babies: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/babies', methods=['POST'])
def create_baby():
    """Create a new baby profile."""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('name'):
            return jsonify({"error": "Baby name is required"}), 400
        
        # Generate unique ID
        profile_id = str(uuid.uuid4())
        
        # Parse birth date if provided
        birth_date = None
        if data.get('birth_date'):
            birth_date = datetime.fromisoformat(data['birth_date'].replace('Z', '+00:00'))
        
        # Create baby profile
        profile = BabyProfile(
            baby_name=data['name'],
            age_months=data.get('age_months', 0),
            birth_date=birth_date
        )
        
        # Store in memory (in production, save to database)
        baby_profiles[profile_id] = profile
        
        logger.info(f"Created baby profile: {data['name']} (ID: {profile_id})")
        
        return jsonify({
            "id": profile_id,
            "name": profile.baby_name,
            "age_months": profile.age_months,
            "birth_date": profile.birth_date.isoformat() if profile.birth_date else None,
            "message": "Baby profile created successfully"
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating baby: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/babies/<profile_id>', methods=['PUT'])
def update_baby_context(profile_id):
    """Update baby's contextual information."""
    try:
        if profile_id not in baby_profiles:
            return jsonify({"error": "Baby profile not found"}), 404
        
        data = request.get_json()
        profile = baby_profiles[profile_id]
        
        # Update feeding time
        if data.get('feeding_time'):
            feeding_time_str = data['feeding_time'].replace('Z', '+00:00')
            feeding_time = datetime.fromisoformat(feeding_time_str)
            # Convert to naive datetime to avoid comparison issues
            feeding_time = feeding_time.replace(tzinfo=None)
            profile.last_feeding_time = feeding_time
            profile.feeding_history.append(feeding_time)
        
        # Update sleep time
        if data.get('sleep_time'):
            sleep_time_str = data['sleep_time'].replace('Z', '+00:00')
            sleep_time = datetime.fromisoformat(sleep_time_str)
            # Convert to naive datetime to avoid comparison issues
            sleep_time = sleep_time.replace(tzinfo=None)
            profile.last_nap_time = sleep_time
        
        # Update diaper change time
        if data.get('diaper_time'):
            diaper_time_str = data['diaper_time'].replace('Z', '+00:00')
            diaper_time = datetime.fromisoformat(diaper_time_str)
            # Convert to naive datetime to avoid comparison issues
            diaper_time = diaper_time.replace(tzinfo=None)
            profile.last_diaper_change = diaper_time
            profile.diaper_change_history.append(diaper_time)
        
        logger.info(f"Updated context for baby: {profile.baby_name}")
        
        return jsonify({
            "message": "Baby context updated successfully",
            "profile": {
                "id": profile_id,
                "name": profile.baby_name,
                "last_feeding": profile.last_feeding_time.isoformat() if profile.last_feeding_time else None,
                "last_sleep": profile.last_nap_time.isoformat() if profile.last_nap_time else None,
                "last_diaper_change": profile.last_diaper_change.isoformat() if profile.last_diaper_change else None
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating baby context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/babies/<profile_id>/insights', methods=['GET'])
def get_baby_insights(profile_id):
    """Get insights for a specific baby."""
    try:
        if profile_id not in baby_profiles:
            return jsonify({"error": "Baby profile not found"}), 404
        
        profile = baby_profiles[profile_id]
        
        # Calculate current status
        insights = {
            "baby_name": profile.baby_name,
            "age_months": profile.age_months,
            "current_status": {
                "time_since_feeding": profile.get_time_since_feeding() if profile.last_feeding_time else None,
                "time_awake": profile.get_time_awake() if profile.last_nap_time else None,
                "time_since_diaper_change": profile.get_time_since_diaper_change() if profile.last_diaper_change else None,
                "likely_hungry": profile.is_likely_hungry(),
                "likely_tired": profile.is_likely_tired(),
                "likely_uncomfortable": profile.is_likely_uncomfortable()
            },
            "context_probabilities": profile.get_context_probabilities()
        }
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Error getting baby insights: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/classify-audio', methods=['POST'])
def classify_audio():
    """Classify uploaded audio file."""
    try:
        if not baby_whisper:
            return jsonify({"error": "BabyWhisper not initialized"}), 500
        
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Get baby profile ID if provided
        profile_id = request.form.get('baby_id')
        baby_profile = None
        if profile_id and profile_id in baby_profiles:
            baby_profile = baby_profiles[profile_id]
        
        # Save audio file temporarily
        temp_dir = tempfile.gettempdir()
        temp_filename = f"baby_cry_{uuid.uuid4()}.wav"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        audio_file.save(temp_path)
        
        try:
            # Classify the audio
            result = baby_whisper.classify_cry(
                audio_path=temp_path,
                baby_profile=baby_profile,
                return_detailed=True
            )
            
            # Clean up temporary file
            os.remove(temp_path)
            
            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()
            result['baby_used'] = baby_profile.baby_name if baby_profile else None
            
            logger.info(f"Audio classified: {result.get('final_prediction', 'unknown')}")
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error classifying audio: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def provide_feedback():
    """Provide feedback on a prediction."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['baby_id', 'predicted_cause', 'actual_cause', 'resolution_method']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"{field} is required"}), 400
        
        profile_id = data['baby_id']
        if profile_id not in baby_profiles:
            return jsonify({"error": "Baby profile not found"}), 404
        
        # Record feedback
        baby_whisper.provide_feedback(
            profile_id=profile_id,
            predicted_cause=data['predicted_cause'],
            actual_cause=data['actual_cause'],
            resolution_method=data['resolution_method'],
            resolution_time_minutes=data.get('resolution_time_minutes', 5.0)
        )
        
        logger.info(f"Feedback recorded: {data['predicted_cause']} -> {data['actual_cause']}")
        
        return jsonify({"message": "Feedback recorded successfully"})
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For development
    logger.info("Starting BabyWhisper Web API in development mode...")
    initialize_baby_whisper()
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    ) 