"""
Feature Extractor for audio analysis and classification
"""

import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract various audio features for analysis and classification.
    """
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the FeatureExtractor.
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        
    def extract_mfcc(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features.
        
        Args:
            audio_data: Input audio data
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features (n_mfcc, n_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio_data, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Dictionary of spectral features
        """
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
            'zero_crossing_rate_std': float(np.std(zero_crossing_rate))
        }
    
    def extract_temporal_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Dictionary of temporal features
        """
        # RMS energy
        rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'tempo': float(tempo),
            'duration': len(audio_data) / self.sample_rate
        }
    
    def extract_chroma_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract chroma features.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Chroma features (12, n_frames)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio_data, 
            sr=self.sample_rate, 
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma
    
    def extract_mel_spectrogram(self, audio_data: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Extract mel spectrogram.
        
        Args:
            audio_data: Input audio data
            n_mels: Number of mel bands
            
        Returns:
            Mel spectrogram (n_mels, n_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels
        )
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def extract_pitch_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Dictionary of pitch features
        """
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(
            y=audio_data, 
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Get dominant pitch
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            return {
                'pitch_mean': float(np.mean(pitch_values)),
                'pitch_std': float(np.std(pitch_values)),
                'pitch_min': float(np.min(pitch_values)),
                'pitch_max': float(np.max(pitch_values))
            }
        else:
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_min': 0.0,
                'pitch_max': 0.0
            }
    
    def extract_all_features(self, audio_data: np.ndarray) -> Dict:
        """
        Extract all available features from audio data.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Dictionary containing all features
        """
        features = {}
        
        # MFCC features (mean and std)
        mfcc = self.extract_mfcc(audio_data)
        features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
        features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio_data)
        features.update(spectral_features)
        
        # Temporal features
        temporal_features = self.extract_temporal_features(audio_data)
        features.update(temporal_features)
        
        # Chroma features (mean)
        chroma = self.extract_chroma_features(audio_data)
        features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
        features['chroma_std'] = np.std(chroma, axis=1).tolist()
        
        # Pitch features
        pitch_features = self.extract_pitch_features(audio_data)
        features.update(pitch_features)
        
        # Additional features
        features['brightness'] = float(np.mean(self.extract_mel_spectrogram(audio_data)))
        
        return features
    
    def extract_features_for_classification(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract features specifically for classification tasks.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Feature vector as numpy array
        """
        features = self.extract_all_features(audio_data)
        
        # Flatten features into a single vector
        feature_vector = []
        
        # MFCC mean (13 features)
        feature_vector.extend(features['mfcc_mean'])
        
        # Spectral features (8 features)
        spectral_keys = ['spectral_centroid_mean', 'spectral_centroid_std', 
                          'spectral_rolloff_mean', 'spectral_rolloff_std',
                          'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                          'zero_crossing_rate_mean', 'zero_crossing_rate_std']
        feature_vector.extend([features[key] for key in spectral_keys])
        
        # Temporal features (4 features)
        temporal_keys = ['rms_mean', 'rms_std', 'tempo', 'duration']
        feature_vector.extend([features[key] for key in temporal_keys])
        
        # Chroma mean (12 features)
        feature_vector.extend(features['chroma_mean'])
        
        # Pitch features (4 features)
        pitch_keys = ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max']
        feature_vector.extend([features[key] for key in pitch_keys])
        
        # Brightness (1 feature)
        feature_vector.append(features['brightness'])
        
        return np.array(feature_vector)


if __name__ == "__main__":
    # Example usage
    from audio_processor import AudioProcessor
    
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    # Test with sample audio
    sample_path = "path/to/your/audio.wav"
    if Path(sample_path).exists():
        audio, sr = processor.load_audio(sample_path)
        
        # Extract features
        features = extractor.extract_all_features(audio)
        classification_features = extractor.extract_features_for_classification(audio)
        
        print(f"Extracted {len(features)} feature groups")
        print(f"Classification feature vector shape: {classification_features.shape}")
        
        # Print some sample features
        print(f"Tempo: {features['tempo']}")
        print(f"Spectral centroid mean: {features['spectral_centroid_mean']}")
        print(f"Zero crossing rate mean: {features['zero_crossing_rate_mean']}")
    else:
        print(f"Sample file {sample_path} not found")