#!/usr/bin/env python3
"""
Test script to verify BabyWhisper installation and basic functionality.

This script performs basic tests to ensure all components are working correctly.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        # Test main components
        from src.main import BabyWhisperClassifier, create_demo_setup
        print("✓ Main BabyWhisper components imported successfully")
        
        # Test audio processing
        from src.audio_processing import AudioFeatureExtractor, AudioPreprocessor
        print("✓ Audio processing modules imported successfully")
        
        # Test models
        from src.models import BabyCryClassifier, ModelTrainer
        print("✓ Model components imported successfully")
        
        # Test context
        from src.context import BabyProfile, ContextManager
        print("✓ Context management modules imported successfully")
        
        # Test utilities
        from src.utils import DataLoader, ModelEvaluator
        print("✓ Utility modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring audio files."""
    print("\nTesting basic functionality...")
    
    try:
        from src.context import BabyProfile
        from src.utils import DataLoader
        
        # Test baby profile creation
        profile = BabyProfile(baby_name="Test Baby", age_months=6)
        print("✓ Baby profile creation works")
        
        # Test data loader
        data_loader = DataLoader()
        print("✓ Data loader initialization works")
        
        # Test feature extractor initialization
        from src.audio_processing import AudioFeatureExtractor
        extractor = AudioFeatureExtractor()
        print("✓ Audio feature extractor initialization works")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_synthetic_data_generation():
    """Test synthetic data generation capability."""
    print("\nTesting synthetic data generation...")
    
    try:
        from src.utils import DataLoader
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            data_loader = DataLoader(temp_dir)
            
            # Test synthetic dataset creation
            dataset_path = data_loader.download_sample_dataset('infant_cry_classification')
            
            # Check if files were created
            if os.path.exists(dataset_path):
                print("✓ Synthetic dataset generation works")
                
                # Check dataset statistics
                stats = data_loader.get_dataset_statistics(dataset_path)
                if stats.get('total_files', 0) > 0:
                    print(f"✓ Dataset contains {stats['total_files']} files")
                    print(f"✓ Classes: {stats.get('classes', [])}")
                else:
                    print("✗ No files found in generated dataset")
                    return False
            else:
                print("✗ Dataset directory not created")
                return False
                
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"✗ Synthetic data generation failed: {e}")
        return False

def test_requirements():
    """Test that required packages are available."""
    print("\nTesting required packages...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    optional_packages = [
        'librosa', 'soundfile', 'tensorflow'
    ]
    
    all_available = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing (required)")
            all_available = False
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"⚠ {package} is missing (optional - some features may not work)")
    
    return all_available

def main():
    """Run all tests."""
    print("🍼 BabyWhisper Installation Test 🍼")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_tests_passed = False
    
    # Test 2: Basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    # Test 3: Required packages
    if not test_requirements():
        all_tests_passed = False
    
    # Test 4: Synthetic data generation
    if not test_synthetic_data_generation():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("🎉 All tests passed! BabyWhisper is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_usage.py")
        print("2. Or try the Jupyter notebook: notebooks/demo_notebook.ipynb")
        print("3. For full demo with model training, ensure librosa and tensorflow are installed")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTo fix issues:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check Python path and module structure")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 