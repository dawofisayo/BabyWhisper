#!/usr/bin/env python3
"""
Example usage of BabyWhisper - AI-powered baby cry classification system.

This script demonstrates the main features of BabyWhisper including:
- Training a model with synthetic data
- Creating baby profiles
- Context-aware cry classification
- Providing feedback for continuous learning
"""

import os
from datetime import datetime, timedelta
from src.main import BabyWhisperClassifier, create_demo_setup


def main():
    """Main demonstration of BabyWhisper functionality."""
    
    print("🍼 Welcome to BabyWhisper Demo! 🍼")
    print("=" * 50)
    
    # 1. Setup the system with demo data
    print("\n1. Setting up BabyWhisper with demo model...")
    baby_whisper = create_demo_setup()
    
    # 2. Check system status
    print("\n2. System Status:")
    status = baby_whisper.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # 3. Create additional baby profile
    print("\n3. Creating a new baby profile...")
    profile_id = baby_whisper.create_baby_profile(
        baby_name="Emma",
        age_months=3
    )
    
    # Update baby context
    now = datetime.now()
    baby_whisper.update_baby_context(
        profile_id=profile_id,
        feeding_time=now - timedelta(hours=2),      # Fed 2 hours ago
        nap_time=now - timedelta(hours=1),          # Napped 1 hour ago  
        diaper_change_time=now - timedelta(minutes=30)  # Changed 30 min ago
    )
    
    # 4. Get baby insights
    print("\n4. Baby Profile Insights:")
    insights = baby_whisper.get_baby_insights(profile_id)
    print(f"   Baby: {insights['baby_name']} ({insights['age_months']} months)")
    print("   Current Status:")
    for key, value in insights['current_status'].items():
        if value is not None:
            if isinstance(value, float):
                print(f"     {key}: {value:.1f}")
            else:
                print(f"     {key}: {value}")
    
    print("   Recommendations:")
    for rec in insights['recommendations']:
        print(f"     - {rec}")
    
    # 5. Create synthetic audio for testing
    print("\n5. Creating synthetic test audio...")
    from src.utils import DataLoader
    
    data_loader = DataLoader()
    dataset_path = data_loader.download_sample_dataset('infant_cry_classification')
    
    # Find sample files
    import glob
    sample_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)
    
    if not sample_files:
        print("   No audio files found. Please check the dataset creation.")
        return
    
    print(f"   Created {len(sample_files)} synthetic audio samples")
    
    # 6. Test cry classification
    print("\n6. Testing Cry Classification:")
    baby_profile = baby_whisper.context_manager.get_baby_profile(profile_id)
    
    # Test different types of cries
    cry_types = ['hunger', 'pain', 'discomfort', 'tiredness', 'normal']
    
    for cry_type in cry_types:
        # Find a sample file for this cry type
        type_files = [f for f in sample_files if cry_type in f.lower()]
        if type_files:
            test_file = type_files[0]
            
            print(f"\n   Testing {cry_type.upper()} cry:")
            print(f"   File: {os.path.basename(test_file)}")
            
            # Classify without context
            basic_result = baby_whisper.classify_cry(test_file, baby_profile=None)
            
            # Check if there's an error in the result
            if 'error' in basic_result:
                print(f"   ⚠️  Error processing audio: {basic_result['error']}")
                continue
                
            print(f"   Basic prediction: {basic_result['final_prediction']} "
                  f"({basic_result['final_confidence']:.2f})")
            
            # Classify with context
            context_result = baby_whisper.classify_cry(test_file, baby_profile=baby_profile)
            
            # Check if there's an error in the context result
            if 'error' in context_result:
                print(f"   ⚠️  Error with context processing: {context_result['error']}")
                continue
                
            print(f"   Context-aware: {context_result['final_prediction']} "
                  f"({context_result['final_confidence']:.2f})")
            
            if 'context_enhanced' in context_result:
                changed = context_result['context_enhanced']['prediction_changed']
                print(f"   Context changed prediction: {'Yes' if changed else 'No'}")
            
            print(f"   Top recommendations:")
            for i, rec in enumerate(context_result['recommendations'][:2]):
                print(f"     {i+1}. {rec}")
    
    # 7. Demonstrate feedback learning
    print("\n7. Providing Feedback for Learning:")
    
    # Simulate some feedback scenarios
    feedback_scenarios = [
        {
            'predicted': 'hunger',
            'actual': 'hunger',
            'resolution': 'feeding',
            'time': 8
        },
        {
            'predicted': 'tiredness', 
            'actual': 'discomfort',
            'resolution': 'diaper_change',
            'time': 3
        },
        {
            'predicted': 'pain',
            'actual': 'pain',
            'resolution': 'medical_attention',
            'time': 15
        }
    ]
    
    for scenario in feedback_scenarios:
        baby_whisper.provide_feedback(
            profile_id=profile_id,
            predicted_cause=scenario['predicted'],
            actual_cause=scenario['actual'],
            resolution_method=scenario['resolution'],
            resolution_time_minutes=scenario['time']
        )
        
        accuracy = "✓" if scenario['predicted'] == scenario['actual'] else "✗"
        print(f"   {accuracy} Predicted: {scenario['predicted']}, "
              f"Actual: {scenario['actual']}, "
              f"Resolved by: {scenario['resolution']} ({scenario['time']}min)")
    
    # 8. Audio feature analysis
    print("\n8. Audio Feature Analysis:")
    if sample_files:
        sample_file = sample_files[0]
        try:
            analysis = baby_whisper.analyze_audio_features(sample_file, save_visualization=False)
            
            if 'error' not in analysis:
                print(f"   File: {os.path.basename(sample_file)}")
                print(f"   Duration: {analysis['duration_seconds']:.2f} seconds")
                print(f"   Energy level: {analysis['energy_level']:.3f}")
                if analysis['dominant_frequency']:
                    print(f"   Dominant frequency: {analysis['dominant_frequency']:.1f} Hz")
                print(f"   Spectral centroid: {analysis['spectral_characteristics']['centroid_mean']:.1f}")
            else:
                print(f"   ⚠️  Error analyzing audio: {analysis['error']}")
        except Exception as e:
            print(f"   ⚠️  Error analyzing audio: {str(e)}")
    
    # 9. Model performance evaluation
    print("\n9. Quick Model Evaluation:")
    
    # Test on a few samples
    correct_predictions = 0
    total_predictions = 0
    
    for cry_type in cry_types:
        type_files = [f for f in sample_files if cry_type in f.lower()]
        if type_files:
            test_file = type_files[0]
            result = baby_whisper.classify_cry(test_file, baby_profile=None)
            
            # Check for errors
            if 'error' in result:
                print(f"   ⚠️  Error evaluating {cry_type}: {result['error']}")
                continue
            
            predicted = result['final_prediction']
            actual = cry_type
            
            if predicted == actual:
                correct_predictions += 1
            total_predictions += 1
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"   Sample accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})")
    
    # 10. Save model
    print("\n10. Saving Model:")
    model_path = "models/example_trained_model"
    baby_whisper.save_model(model_path)
    print(f"   Model saved to: {model_path}")
    
    # 11. Summary
    print("\n" + "=" * 50)
    print("🎉 BabyWhisper Demo Complete! 🎉")
    print("\nKey Features Demonstrated:")
    print("✓ AI-powered cry classification with multiple models")
    print("✓ Context-aware predictions using baby profiles") 
    print("✓ Personalized recommendations based on baby's needs")
    print("✓ Continuous learning through user feedback")
    print("✓ Comprehensive audio feature analysis")
    print("✓ Model persistence and reusability")
    
    print(f"\nActive baby profiles: {len(baby_whisper.context_manager.active_profiles)}")
    print(f"Available cry categories: {', '.join(baby_whisper.classifier.classes)}")
    
    print("\nNext Steps:")
    print("- Train with real baby cry datasets")
    print("- Implement real-time audio processing") 
    print("- Create mobile/web application")
    print("- Add more contextual features")
    print("- Integrate with baby care devices")
    
    print("\nBabyWhisper is ready to help parents understand their babies better! 👶💕")


if __name__ == "__main__":
    main() 