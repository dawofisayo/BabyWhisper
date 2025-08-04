"""Audio feature extraction for baby cry classification."""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract various audio features for baby cry classification."""
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13, 
                 n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features from baby cry.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing various audio features
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        features = {}
        
        # 1. MFCC Features (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        features['mfcc'] = mfcc
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # 2. Temporal Features
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 3. Tempo and Rhythm
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # 4. Mel-scaled Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        features['mel_spectrogram'] = mel_spec
        features['mel_spec_mean'] = np.mean(mel_spec, axis=1)
        features['mel_spec_std'] = np.std(mel_spec, axis=1)
        
        # 5. Fundamental Frequency (F0)
        f0 = librosa.piptrack(y=y, sr=sr, threshold=0.1)[0]
        f0_values = f0[f0 > 0]
        if len(f0_values) > 0:
            features['f0_mean'] = np.mean(f0_values)
            features['f0_std'] = np.std(f0_values)
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
        
        # 6. Audio statistics
        features['duration'] = len(y) / sr
        features['energy'] = np.sum(y ** 2)
        features['max_amplitude'] = np.max(np.abs(y))
        features['min_amplitude'] = np.min(np.abs(y))
        
        return features
    
    def extract_feature_vector(self, audio_path: str) -> np.ndarray:
        """
        Extract features and return as a single feature vector for ML models.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            1D numpy array of features
        """
        features = self.extract_features(audio_path)
        
        # Combine all scalar features into a single vector
        feature_vector = []
        
        # Add MFCC statistics (ensure they're flattened)
        mfcc_mean = np.array(features['mfcc_mean']).flatten()
        mfcc_std = np.array(features['mfcc_std']).flatten()
        feature_vector.extend(mfcc_mean.tolist())
        feature_vector.extend(mfcc_std.tolist())
        
        # Add temporal features (ensure scalars)
        temporal_features = [
            features['zcr_mean'],
            features['zcr_std'],
            features['rms_mean'],
            features['rms_std'],
            features['tempo']
        ]
        # Convert to scalars if they're arrays
        for feat in temporal_features:
            if hasattr(feat, 'item'):
                feature_vector.append(feat.item())
            elif np.isscalar(feat):
                feature_vector.append(float(feat))
            else:
                feature_vector.append(float(np.mean(feat)))
        
        # Add mel spectrogram statistics (ensure they're flattened)
        mel_mean = np.array(features['mel_spec_mean']).flatten()
        mel_std = np.array(features['mel_spec_std']).flatten()
        feature_vector.extend(mel_mean.tolist())
        feature_vector.extend(mel_std.tolist())
        
        # Add fundamental frequency (ensure scalars)
        f0_features = [features['f0_mean'], features['f0_std']]
        for feat in f0_features:
            if hasattr(feat, 'item'):
                feature_vector.append(feat.item())
            elif np.isscalar(feat):
                feature_vector.append(float(feat))
            else:
                feature_vector.append(float(np.mean(feat)))
        
        # Add audio statistics (ensure scalars)
        audio_stats = [
            features['duration'],
            features['energy'],
            features['max_amplitude'],
            features['min_amplitude']
        ]
        for feat in audio_stats:
            if hasattr(feat, 'item'):
                feature_vector.append(feat.item())
            elif np.isscalar(feat):
                feature_vector.append(float(feat))
            else:
                feature_vector.append(float(np.mean(feat)))
        
        # Convert to numpy array with proper dtype
        feature_array = np.array(feature_vector, dtype=np.float64)
        # Ensure it's 1D
        if feature_array.ndim > 1:
            feature_array = feature_array.flatten()
        return feature_array
    
    def visualize_features(self, audio_path: str, save_path: Optional[str] = None):
        """
        Create visualizations of audio features.
        
        Args:
            audio_path: Path to audio file
            save_path: Optional path to save the visualization
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Baby Cry Audio Analysis', fontsize=16)
        
        # Waveform
        axes[0, 0].plot(y)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Samples')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        librosa.display.specshow(mfcc, x_axis='time', ax=axes[0, 2])
        axes[0, 2].set_title('MFCC')
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr, ax=axes[1, 0])
        axes[1, 0].set_title('Mel Spectrogram')
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        frames = range(len(zcr))
        t = librosa.frames_to_time(frames)
        axes[1, 1].plot(t, zcr)
        axes[1, 1].set_title('Zero Crossing Rate')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Rate')
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        frames = range(len(rms))
        t = librosa.frames_to_time(frames)
        axes[1, 2].plot(t, rms)
        axes[1, 2].set_title('RMS Energy')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Energy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_names(self) -> list:
        """Return list of feature names for the feature vector."""
        feature_names = []
        
        # MFCC features
        for i in range(self.n_mfcc):
            feature_names.append(f'mfcc_{i}_mean')
        for i in range(self.n_mfcc):
            feature_names.append(f'mfcc_{i}_std')
        
        # Temporal features
        feature_names.extend([
            'zcr_mean', 'zcr_std', 'rms_mean', 'rms_std', 'tempo'
        ])
        
        # Mel spectrogram features (128 bands by default)
        for i in range(128):
            feature_names.append(f'mel_{i}_mean')
        for i in range(128):
            feature_names.append(f'mel_{i}_std')
            
        # Fundamental frequency
        feature_names.extend(['f0_mean', 'f0_std'])
        
        # Audio statistics
        feature_names.extend([
            'duration', 'energy', 'max_amplitude', 'min_amplitude'
        ])
        
        return feature_names 