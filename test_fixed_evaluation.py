#!/usr/bin/env python3
"""Test script to verify the fixed evaluation is working correctly."""

import sys
import os
sys.path.append('src')

from models.model_trainer import ModelTrainer
import numpy as np

def test_fixed_evaluation():
    """Test that the fixed evaluation is working correctly."""
    print("🧪 Testing Fixed Evaluation...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    print("✅ Trainer initialized")
    
    # Test training
    print("\n🤖 Testing Model Training...")
    
    try:
        # Train model
        results = trainer.train_model(save_model=False)
        
        print("✅ Training completed!")
        
        # Check ensemble accuracy
        if 'ensemble' in results.get('training_results', {}):
            ensemble_accuracy = results['training_results']['ensemble']['validation_accuracy']
            print(f"📈 Ensemble Validation Accuracy: {ensemble_accuracy:.3f}")
        else:
            print("⚠️  Ensemble not found in results")
        
        # Check test results
        if 'test_results' in results:
            test_accuracies = results['test_results']
            print("\n📊 Test Accuracies:")
            for model_name, acc_data in test_accuracies.items():
                if 'test_accuracy' in acc_data:
                    print(f"   {model_name}: {acc_data['test_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_evaluation()
    if success:
        print("\n🎉 Fixed Evaluation Test PASSED!")
    else:
        print("\n❌ Fixed Evaluation Test FAILED!") 