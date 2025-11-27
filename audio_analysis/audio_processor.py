"""
Audio Processor for handling audio files and basic preprocessing
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Core audio processing class for loading, preprocessing, and manipulating audio files.
    """
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the AudioProcessor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def load_audio(self, file_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            file_path: Path to audio file
            duration: Duration to load (seconds), None for full file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            logger.info(f"Loaded audio from {file_path}: shape={audio.shape}, sr={sr}")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio from {file_path}: {e}")
            raise
    
    def save_audio(self, audio_data: np.ndarray, file_path: str, sample_rate: Optional[int] = None) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data array
            file_path: Output file path
            sample_rate: Sample rate (uses instance default if None)
        """
        sr = sample_rate or self.sample_rate
        try:
            sf.write(file_path, audio_data, sr)
            logger.info(f"Saved audio to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")
            raise
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to [-1, 1] range.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
        """
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def trim_silence(self, audio_data: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio_data: Input audio data
            top_db: Threshold for silence detection
            
        Returns:
            Trimmed audio data
        """
        trimmed, _ = librosa.effects.trim(audio_data, top_db=top_db)
        return trimmed
    
    def apply_preemphasis(self, audio_data: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """
        Apply preemphasis filter to audio.
        
        Args:
            audio_data: Input audio data
            coeff: Preemphasis coefficient
            
        Returns:
            Preemphasized audio data
        """
        return np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])
    
    def segment_audio(self, audio_data: np.ndarray, segment_length: float, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Segment audio into overlapping chunks.
        
        Args:
            audio_data: Input audio data
            segment_length: Length of each segment in seconds
            overlap: Overlap ratio between segments
            
        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for i in range(0, len(audio_data) - segment_samples + 1, hop_samples):
            segment = audio_data[i:i + segment_samples]
            segments.append(segment)
        
        return segments
    
    def get_audio_info(self, file_path: str) -> Dict:
        """
        Get basic information about an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            info = sf.info(file_path)
            audio, sr = self.load_audio(file_path)
            
            return {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'frames': info.frames
            }
        except Exception as e:
            logger.error(f"Error getting audio info for {file_path}: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor()
    
    # Test with sample audio
    sample_path = "path/to/your/audio.wav"
    if Path(sample_path).exists():
        audio, sr = processor.load_audio(sample_path)
        print(f"Loaded audio: shape={audio.shape}, sample_rate={sr}")
        
        # Normalize and trim
        normalized = processor.normalize_audio(audio)
        trimmed = processor.trim_silence(normalized)
        
        print(f"Original length: {len(audio)}, Trimmed length: {len(trimmed)}")
    else:
        print(f"Sample file {sample_path} not found")