#!/usr/bin/env python3
"""
Demo script to showcase BabyWhisper's context-aware prediction capabilities.

This demonstrates how baby profiles and contextual information 
dramatically improve prediction accuracy and provide intelligent explanations.
"""

import os
from datetime import datetime, timedelta
from src.main import BabyWhisperClassifier
from src.context import BabyProfile


def demo_context_awareness():
    """Demonstrate context-aware predictions vs base predictions."""
    
    print("🍼 BabyWhisper Context-Aware Prediction Demo 🍼")
    print("=" * 60)
    
    # Initialize system
    print("\n1. Setting up BabyWhisper...")
    from src.models import ModelTrainer
    trainer = ModelTrainer()
    classifier_instance = trainer.quick_demo_training()
    
    baby_whisper = BabyWhisperClassifier()
    baby_whisper.classifier = classifier_instance
    
    # Create baby profiles with different contexts
    print("\n2. Creating baby profiles with different contexts...")
    
    now = datetime.now()
    
    # Profile 1: Recently fed baby
    profile1 = BabyProfile(baby_name="Recently Fed Baby", age_months=4)
    profile1.update_feeding(now - timedelta(minutes=20))  # Fed 20 min ago
    profile1.update_sleep(sleep_end=now - timedelta(hours=1))  # Awake 1 hour
    profile1.update_diaper_change(now - timedelta(hours=2))  # Changed 2h ago
    
    # Profile 2: Hungry baby
    profile2 = BabyProfile(baby_name="Hungry Baby", age_months=6)
    profile2.update_feeding(now - timedelta(hours=4))  # Fed 4 hours ago
    profile2.update_sleep(sleep_end=now - timedelta(hours=2))  # Awake 2 hours
    profile2.update_diaper_change(now - timedelta(minutes=30))  # Recently changed
    
    # Profile 3: Tired baby
    profile3 = BabyProfile(baby_name="Tired Baby", age_months=3)
    profile3.update_feeding(now - timedelta(hours=1.5))  # Fed 1.5h ago
    profile3.update_sleep(sleep_end=now - timedelta(hours=3))  # Awake 3 hours!
    profile3.update_diaper_change(now - timedelta(minutes=45))  # Recently changed
    
    profiles = [
        ("recently_fed", profile1),
        ("hungry", profile2),
        ("tired", profile3)
    ]
    
    # Get sample audio files
    print("\n3. Creating test audio samples...")
    from src.utils import DataLoader
    data_loader = DataLoader()
    dataset_path = data_loader.download_sample_dataset('infant_cry_classification')
    
    import glob
    sample_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)
    
    # Test each cry type with each profile
    cry_types = ['hunger', 'pain', 'discomfort', 'tiredness']
    
    print("\n4. Testing Context-Aware Predictions:")
    print("=" * 60)
    
    for cry_type in cry_types:
        # Find sample file for this cry type
        type_files = [f for f in sample_files if cry_type in f.lower()]
        if not type_files:
            continue
            
        test_file = type_files[0]
        
        print(f"\n🎵 TESTING {cry_type.upper()} CRY:")
        print(f"Audio file: {os.path.basename(test_file)}")
        print("-" * 40)
        
        # Base prediction (no context)
        base_result = baby_whisper.classify_cry(test_file, baby_profile=None)
        print(f"📊 BASE AI PREDICTION: {base_result['final_prediction']} "
              f"({base_result['final_confidence']:.2f} confidence)")
        
        # Test with each baby profile
        for profile_name, profile in profiles:
            result = baby_whisper.classify_cry(test_file, baby_profile=profile)
            
            context_changed = ""
            if 'context_enhanced' in result:
                if result['context_enhanced']['prediction_changed']:
                    context_changed = " 🔄 CHANGED BY CONTEXT!"
            
            print(f"\n👶 WITH {profile.baby_name.upper()}:")
            print(f"   Prediction: {result['final_prediction']} "
                  f"({result['final_confidence']:.2f}){context_changed}")
            
            # Show context factors
            if 'context_enhanced' in result:
                factors = result['context_enhanced']['context_factors']
                active_factors = [f"{k}: +{v:.2f}" for k, v in factors.items() if v > 0]
                if active_factors:
                    print(f"   Context boosts: {', '.join(active_factors)}")
            
            # Show key recommendations
            print(f"   Key recommendation: {result['recommendations'][0]}")
    
    print("\n" + "=" * 60)
    print("🎯 CONTEXT AWARENESS DEMONSTRATION COMPLETE!")
    print("\nKey Benefits Shown:")
    print("✅ Predictions adjust based on feeding, sleep, and care timing")
    print("✅ Age-appropriate expectations (newborns vs older babies)")
    print("✅ Time-of-day considerations")
    print("✅ Intelligent explanations with specific recommendations")
    print("✅ Confidence adjustments based on likelihood")
    
    # Show detailed context example
    print(f"\n📋 DETAILED CONTEXT EXAMPLE:")
    hungry_profile = profiles[1][1]  # Hungry baby profile
    insights = baby_whisper.context_manager.get_personalized_insights('hungry')
    baby_whisper.context_manager.add_baby_profile(hungry_profile, 'hungry')
    insights = baby_whisper.context_manager.get_personalized_insights('hungry')
    
    print(f"Baby: {hungry_profile.baby_name}")
    print(f"Time since feeding: {hungry_profile.get_time_since_feeding():.1f} hours")
    print(f"Time awake: {hungry_profile.get_time_awake():.1f} hours")
    print(f"Likely hungry: {hungry_profile.is_likely_hungry()}")
    print(f"Likely tired: {hungry_profile.is_likely_tired()}")
    print(f"Likely uncomfortable: {hungry_profile.is_likely_uncomfortable()}")


if __name__ == "__main__":
    demo_context_awareness() 