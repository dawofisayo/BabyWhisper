"""Audio preprocessing utilities for baby cry analysis."""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import os
from scipy import signal


class AudioPreprocessor:
    """Preprocess audio data for baby cry classification."""
    
    def __init__(self, target_sr: int = 22050, target_duration: float = 3.0):
        """
        Initialize the audio preprocessor.
        
        Args:
            target_sr: Target sample rate
            target_duration: Target duration in seconds for standardization
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
    
    def load_and_preprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # Remove silence
        audio = self.remove_silence(audio)
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Standardize duration
        audio = self.standardize_duration(audio)
        
        # Apply noise reduction
        audio = self.reduce_noise(audio)
        
        return audio, self.target_sr
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Remove silence from the beginning and end of audio.
        
        Args:
            audio: Input audio signal
            threshold: Amplitude threshold for silence detection
            
        Returns:
            Audio with silence removed
        """
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold
        
        if not np.any(non_silent):
            return audio  # Return original if all silent
        
        # Find start and end of non-silent regions
        start_idx = np.argmax(non_silent)
        end_idx = len(audio) - np.argmax(non_silent[::-1]) - 1
        
        return audio[start_idx:end_idx + 1]
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def standardize_duration(self, audio: np.ndarray) -> np.ndarray:
        """
        Standardize audio duration by padding or truncating.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Audio with standardized duration
        """
        current_samples = len(audio)
        
        if current_samples > self.target_samples:
            # Truncate from center to preserve important parts
            start_idx = (current_samples - self.target_samples) // 2
            return audio[start_idx:start_idx + self.target_samples]
        elif current_samples < self.target_samples:
            # Pad with zeros
            padding = self.target_samples - current_samples
            pad_before = padding // 2
            pad_after = padding - pad_before
            return np.pad(audio, (pad_before, pad_after), mode='constant')
        
        return audio
    
    def reduce_noise(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Apply simple noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio signal
            noise_factor: Factor for noise reduction
            
        Returns:
            Denoised audio
        """
        # Estimate noise from the first 0.5 seconds
        noise_duration = min(int(0.5 * self.target_sr), len(audio) // 4)
        noise_sample = audio[:noise_duration]
        
        # Apply spectral subtraction
        stft = librosa.stft(audio)
        noise_stft = librosa.stft(noise_sample)
        
        # Estimate noise spectrum
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
        
        # Apply spectral subtraction
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Subtract noise estimate
        clean_magnitude = magnitude - noise_factor * np.sqrt(noise_power)
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Reconstruct signal
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft)
        
        return clean_audio
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str = 'noise') -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Args:
            audio: Input audio signal
            augmentation_type: Type of augmentation ('noise', 'pitch', 'speed', 'shift')
            
        Returns:
            Augmented audio
        """
        if augmentation_type == 'noise':
            # Add random noise
            noise_factor = np.random.uniform(0.005, 0.015)
            noise = np.random.normal(0, noise_factor, len(audio))
            return audio + noise
            
        elif augmentation_type == 'pitch':
            # Pitch shifting
            pitch_factor = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=pitch_factor)
            
        elif augmentation_type == 'speed':
            # Speed/tempo change
            speed_factor = np.random.uniform(0.9, 1.1)
            return librosa.effects.time_stretch(audio, rate=speed_factor)
            
        elif augmentation_type == 'shift':
            # Time shifting
            shift_samples = np.random.randint(-len(audio)//10, len(audio)//10)
            return np.roll(audio, shift_samples)
            
        return audio
    
    def detect_cry_segments(self, audio: np.ndarray, 
                           min_duration: float = 0.5) -> list:
        """
        Detect cry segments in longer audio recordings.
        
        Args:
            audio: Input audio signal
            min_duration: Minimum duration for a cry segment
            
        Returns:
            List of (start_time, end_time) tuples for cry segments
        """
        # Calculate energy in sliding windows
        window_length = int(0.1 * self.target_sr)  # 100ms windows
        hop_length = window_length // 2
        
        energy = []
        for i in range(0, len(audio) - window_length, hop_length):
            window = audio[i:i + window_length]
            energy.append(np.sum(window ** 2))
        
        energy = np.array(energy)
        
        # Normalize energy
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        
        # Find segments above threshold
        threshold = np.percentile(energy, 70)  # Top 30% energy
        active_segments = energy > threshold
        
        # Find continuous segments
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_active in enumerate(active_segments):
            if is_active and not in_segment:
                segment_start = i
                in_segment = True
            elif not is_active and in_segment:
                segment_duration = (i - segment_start) * hop_length / self.target_sr
                if segment_duration >= min_duration:
                    start_time = segment_start * hop_length / self.target_sr
                    end_time = i * hop_length / self.target_sr
                    segments.append((start_time, end_time))
                in_segment = False
        
        return segments
    
    def extract_cry_segments(self, audio_path: str, 
                           output_dir: str) -> list:
        """
        Extract cry segments from longer audio file and save them.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save extracted segments
            
        Returns:
            List of paths to extracted segments
        """
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        
        # Detect cry segments
        segments = self.detect_cry_segments(audio)
        
        # Extract and save segments
        os.makedirs(output_dir, exist_ok=True)
        segment_paths = []
        
        for i, (start_time, end_time) in enumerate(segments):
            start_sample = int(start_time * self.target_sr)
            end_sample = int(end_time * self.target_sr)
            
            segment = audio[start_sample:end_sample]
            
            # Preprocess segment
            segment = self.remove_silence(segment)
            segment = self.normalize_audio(segment)
            segment = self.standardize_duration(segment)
            
            # Save segment
            segment_filename = f"cry_segment_{i:03d}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            sf.write(segment_path, segment, self.target_sr)
            segment_paths.append(segment_path)
        
        return segment_paths
    
    def batch_preprocess(self, input_dir: str, output_dir: str) -> list:
        """
        Batch preprocess all audio files in a directory.
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save processed audio files
            
        Returns:
            List of processed file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_files = []
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                
                try:
                    # Load and preprocess
                    audio, sr = self.load_and_preprocess(input_path)
                    
                    # Save processed audio
                    sf.write(output_path, audio, sr)
                    processed_files.append(output_path)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return processed_files 