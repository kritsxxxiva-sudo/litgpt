"""
Audio Analysis and Processing Pipeline for Thai Dialect Classification

This package provides comprehensive tools for:
- Audio preprocessing and feature extraction
- Spectrogram generation and analysis
- Audio classification for Thai dialects (Phuthai, Toei, Kaleang, Khmer, Lao)
- Synthetic audio evaluation and validation
- Visualization and reporting tools
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .audio_processor import AudioProcessor
from .feature_extractor import FeatureExtractor
from .classifier import DialectClassifier
from .synthetic_evaluator import SyntheticAudioEvaluator
from .visualizer import AudioVisualizer

__all__ = [
    'AudioProcessor',
    'FeatureExtractor', 
    'DialectClassifier',
    'SyntheticAudioEvaluator',
    'AudioVisualizer'
]