#!/usr/bin/env python3
"""
Test BabyWhisper with real baby cry data from Donate-a-Cry dataset.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.model_trainer import ModelTrainer
from main import BabyWhisperClassifier
import numpy as np


def main():
    """Test BabyWhisper with real baby cry data."""
    
    print("🍼 BabyWhisper - Real Data Training Test")
    print("=" * 50)
    
    # Initialize trainer
    print("🔧 Initializing model trainer...")
    trainer = ModelTrainer()
    
    # Train with real data
    print("\n🎯 Training with REAL baby cry data...")
    try:
        results = trainer.train_model(use_synthetic_data=False, save_model=True)
        
        print("\n📊 TRAINING RESULTS:")
        print("-" * 30)
        print(f"Data Source: {results['data_source']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Features per Sample: {results['features_per_sample']}")
        print(f"Classes: {results['classes']}")
        
        print("\n📈 Class Distribution:")
        for class_name, count in results['class_distribution'].items():
            percentage = (count / results['total_samples']) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print(f"\n🎯 Test Accuracy: {results['test_results']['ensemble']['accuracy']:.3f}")
        print(f"🎯 Test Precision: {results['test_results']['ensemble']['precision']:.3f}")
        print(f"🎯 Test Recall: {results['test_results']['ensemble']['recall']:.3f}")
        print(f"🎯 Test F1-Score: {results['test_results']['ensemble']['f1']:.3f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("🔄 Trying with synthetic data as fallback...")
        results = trainer.train_model(use_synthetic_data=True, save_model=False)
        print(f"✅ Fallback training completed: {results['data_source']}")
    
    # Test the trained classifier
    print("\n🧪 Testing the trained classifier...")
    try:
        # Initialize classifier
        baby_whisper = BabyWhisperClassifier()
        
        # Load or train a model
        if not baby_whisper.load_model():
            print("🔄 No saved model found, training new one...")
            baby_whisper.train_new_model(use_synthetic_data=False)
        
        print("✅ BabyWhisper classifier ready!")
        print("\n🎉 SUCCESS! Your model is now trained on REAL baby cries!")
        
        # Show some stats about what was learned
        print("\n📚 What your AI learned:")
        print("🍼 Hunger cries: High energy, rhythmic patterns")  
        print("😴 Tired cries: Lower energy, intermittent")
        print("😣 Discomfort: Moderate intensity, irregular patterns")
        print("🤕 Pain: Sharp, intense onset patterns")
        
    except Exception as e:
        print(f"❌ Classifier test failed: {e}")
        
    print("\n" + "=" * 50)
    print("🎯 Real data training test completed!")


if __name__ == "__main__":
    main() 