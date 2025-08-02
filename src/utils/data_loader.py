"""Data loading utilities for baby cry classification."""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import requests
from tqdm import tqdm
import zipfile
import tarfile


class DataLoader:
    """Utility class for loading and managing baby cry datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary data directories."""
        dirs = [
            self.data_dir,
            os.path.join(self.data_dir, "raw"),
            os.path.join(self.data_dir, "processed"),
            os.path.join(self.data_dir, "models"),
            os.path.join(self.data_dir, "samples")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_sample_dataset_info(self) -> Dict:
        """
        Create information about sample datasets available for download.
        
        Returns:
            Dictionary with dataset information
        """
        sample_datasets = {
            "baby_cry_detection": {
                "description": "Sample baby cry vs non-cry detection dataset",
                "url": "https://github.com/giuliomorina/baby_cry_detection/archive/refs/heads/master.zip",
                "format": "zip",
                "size_mb": 15,
                "files": ["cry", "no_cry"],
                "classes": ["cry", "no_cry"]
            },
            "infant_cry_classification": {
                "description": "Synthetic infant cry classification dataset",
                "url": "synthetic",
                "format": "generated",
                "size_mb": 5,
                "files": ["hunger", "pain", "discomfort", "tiredness", "normal"],
                "classes": ["hunger", "pain", "discomfort", "tiredness", "normal"]
            }
        }
        
        return sample_datasets
    
    def download_sample_dataset(self, dataset_name: str) -> str:
        """
        Download a sample dataset for demonstration.
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            Path to downloaded dataset
        """
        datasets = self.create_sample_dataset_info()
        
        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not available. Available: {list(datasets.keys())}")
        
        dataset_info = datasets[dataset_name]
        output_dir = os.path.join(self.data_dir, "raw", dataset_name)
        
        if dataset_info["url"] == "synthetic":
            return self._create_synthetic_audio_dataset(output_dir, dataset_info["classes"])
        
        print(f"Downloading {dataset_name}...")
        
        # Download file
        response = requests.get(dataset_info["url"], stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = os.path.join(self.data_dir, "temp_download")
        
        with open(temp_file, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        
        # Extract based on format
        if dataset_info["format"] == "zip":
            self._extract_zip(temp_file, output_dir)
        elif dataset_info["format"] == "tar":
            self._extract_tar(temp_file, output_dir)
        
        # Clean up
        os.remove(temp_file)
        
        print(f"Dataset {dataset_name} downloaded to {output_dir}")
        return output_dir
    
    def _extract_zip(self, zip_path: str, output_dir: str):
        """Extract zip file to output directory."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    
    def _extract_tar(self, tar_path: str, output_dir: str):
        """Extract tar file to output directory."""
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(output_dir)
    
    def _create_synthetic_audio_dataset(self, output_dir: str, classes: List[str]) -> str:
        """
        Create a synthetic audio dataset for demonstration.
        
        Args:
            output_dir: Output directory
            classes: List of classes to create
            
        Returns:
            Path to created dataset
        """
        print("Creating synthetic audio dataset...")
        
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            print("librosa and soundfile required for synthetic dataset creation")
            return output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create class directories
        for class_name in classes:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # Generate synthetic audio samples
        sample_rate = 22050
        duration = 3.0  # 3 seconds
        samples_per_class = 20
        
        for class_name in classes:
            class_dir = os.path.join(output_dir, class_name)
            
            for i in range(samples_per_class):
                # Generate synthetic audio based on class
                audio = self._generate_synthetic_cry(class_name, sample_rate, duration)
                
                # Save audio file
                filename = f"{class_name}_{i:03d}.wav"
                filepath = os.path.join(class_dir, filename)
                sf.write(filepath, audio, sample_rate)
        
        # Create labels CSV
        self._create_labels_csv(output_dir, classes, samples_per_class)
        
        print(f"Synthetic dataset created with {len(classes)} classes, {samples_per_class} samples each")
        return output_dir
    
    def _generate_synthetic_cry(self, class_name: str, sample_rate: int, duration: float) -> np.ndarray:
        """Generate synthetic cry audio based on class characteristics."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if class_name == 'hunger':
            # Rhythmic crying pattern
            freq = 300 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Varying frequency
            audio = np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.3)
            # Add rhythm
            rhythm = (np.sin(2 * np.pi * 2 * t) > 0).astype(float)
            audio *= rhythm
            
        elif class_name == 'pain':
            # Sharp, intense cry
            freq = 400 + 100 * np.random.random(len(t))
            audio = np.sin(2 * np.pi * freq * t) * (1 - np.exp(-t * 2))
            # Add sharp onset
            audio *= (1 + 2 * np.exp(-t * 5))
            
        elif class_name == 'discomfort':
            # Moderate intensity, irregular
            freq = 250 + 30 * np.sin(2 * np.pi * 0.3 * t + np.random.random() * 2 * np.pi)
            audio = np.sin(2 * np.pi * freq * t) * (0.7 + 0.3 * np.random.random(len(t)))
            
        elif class_name == 'tiredness':
            # Lower intensity, whimpering
            freq = 200 + 20 * np.sin(2 * np.pi * 0.2 * t)
            audio = np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.5) * 0.6
            # Add breaks
            breaks = (np.sin(2 * np.pi * 1.5 * t + np.pi/4) > 0.5).astype(float)
            audio *= breaks
            
        else:  # normal or other
            # Quiet background noise
            audio = 0.1 * np.random.normal(0, 1, len(t))
        
        # Add some background noise
        noise = 0.05 * np.random.normal(0, 1, len(t))
        audio += noise
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    def _create_labels_csv(self, output_dir: str, classes: List[str], samples_per_class: int):
        """Create a CSV file with labels for the dataset."""
        labels_data = []
        
        for class_name in classes:
            for i in range(samples_per_class):
                filename = f"{class_name}_{i:03d}.wav"
                relative_path = os.path.join(class_name, filename)
                labels_data.append({
                    'filename': relative_path,
                    'label': class_name,
                    'class_id': classes.index(class_name)
                })
        
        df = pd.DataFrame(labels_data)
        csv_path = os.path.join(output_dir, 'labels.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Labels CSV created: {csv_path}")
    
    def load_dataset_from_directory(self, dataset_dir: str) -> Tuple[List[str], List[str]]:
        """
        Load dataset from directory structure.
        
        Args:
            dataset_dir: Directory containing the dataset
            
        Returns:
            Tuple of (file_paths, labels)
        """
        file_paths = []
        labels = []
        
        # Check if labels.csv exists
        labels_csv = os.path.join(dataset_dir, 'labels.csv')
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                full_path = os.path.join(dataset_dir, row['filename'])
                if os.path.exists(full_path):
                    file_paths.append(full_path)
                    labels.append(row['label'])
        else:
            # Load from directory structure
            for class_name in os.listdir(dataset_dir):
                class_dir = os.path.join(dataset_dir, class_name)
                if os.path.isdir(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                            file_paths.append(os.path.join(class_dir, filename))
                            labels.append(class_name)
        
        return file_paths, labels
    
    def get_dataset_statistics(self, dataset_dir: str) -> Dict:
        """
        Get statistics about a dataset.
        
        Args:
            dataset_dir: Directory containing the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        file_paths, labels = self.load_dataset_from_directory(dataset_dir)
        
        if not file_paths:
            return {"error": "No audio files found"}
        
        # Count by class
        from collections import Counter
        class_counts = Counter(labels)
        
        # Calculate total duration (if possible)
        total_duration = 0
        valid_files = 0
        
        try:
            import librosa
            for file_path in file_paths[:10]:  # Sample first 10 files
                try:
                    duration = librosa.get_duration(filename=file_path)
                    total_duration += duration
                    valid_files += 1
                except:
                    continue
            
            if valid_files > 0:
                avg_duration = total_duration / valid_files
                estimated_total = avg_duration * len(file_paths)
            else:
                estimated_total = None
        except ImportError:
            estimated_total = None
        
        stats = {
            'total_files': len(file_paths),
            'classes': list(class_counts.keys()),
            'class_distribution': dict(class_counts),
            'estimated_total_duration_seconds': estimated_total,
            'dataset_directory': dataset_dir
        }
        
        return stats
    
    def create_train_test_split_files(self, dataset_dir: str, 
                                    test_size: float = 0.2,
                                    random_state: int = 42) -> Tuple[str, str]:
        """
        Create train/test split files for a dataset.
        
        Args:
            dataset_dir: Directory containing the dataset
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (train_csv_path, test_csv_path)
        """
        file_paths, labels = self.load_dataset_from_directory(dataset_dir)
        
        if not file_paths:
            raise ValueError("No audio files found in dataset")
        
        # Create DataFrame
        df = pd.DataFrame({
            'filepath': file_paths,
            'label': labels
        })
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['label']
        )
        
        # Save split files
        train_csv = os.path.join(dataset_dir, 'train_split.csv')
        test_csv = os.path.join(dataset_dir, 'test_split.csv')
        
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        
        print(f"Train split: {len(train_df)} samples -> {train_csv}")
        print(f"Test split: {len(test_df)} samples -> {test_csv}")
        
        return train_csv, test_csv
    
    def validate_dataset(self, dataset_dir: str) -> Dict:
        """
        Validate a dataset for common issues.
        
        Args:
            dataset_dir: Directory containing the dataset
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'file_count': 0,
            'classes': []
        }
        
        try:
            file_paths, labels = self.load_dataset_from_directory(dataset_dir)
            validation_results['file_count'] = len(file_paths)
            validation_results['classes'] = list(set(labels))
            
            if len(file_paths) == 0:
                validation_results['valid'] = False
                validation_results['issues'].append("No audio files found")
                return validation_results
            
            # Check class balance
            from collections import Counter
            class_counts = Counter(labels)
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            
            if max_count / min_count > 5:
                validation_results['warnings'].append(
                    f"Significant class imbalance detected (max: {max_count}, min: {min_count})"
                )
            
            # Check if all files exist and are readable
            missing_files = 0
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    missing_files += 1
            
            if missing_files > 0:
                validation_results['issues'].append(f"{missing_files} files are missing")
                validation_results['valid'] = False
            
            # Check minimum samples per class
            if min_count < 5:
                validation_results['warnings'].append(
                    f"Some classes have very few samples (minimum: {min_count})"
                )
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Error during validation: {str(e)}")
        
        return validation_results 