#!/usr/bin/env python3
"""
Demo script for Thai Dialect Audio Analysis System
Tests the implementation with the provided audio dataset
"""

import sys
import logging
from pathlib import Path
import json
import numpy as np

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from audio_analysis import (
    AudioProcessor,
    FeatureExtractor,
    DialectClassifier,
    SyntheticAudioEvaluator,
    AudioVisualizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_audio_processing(audio_data_path: str):
    """Demonstrate basic audio processing capabilities."""
    logger.info("=== Audio Processing Demo ===")
    
    processor = AudioProcessor()
    
    # Find a sample audio file
    audio_path = Path(audio_data_path)
    sample_file = None
    
    for dialect_dir in audio_path.iterdir():
        if dialect_dir.is_dir():
            wav_files = list(dialect_dir.glob("*.wav"))
            if wav_files:
                sample_file = str(wav_files[0])
                break
    
    if not sample_file:
        logger.error("No audio files found in dataset")
        return
    
    logger.info(f"Processing sample file: {sample_file}")
    
    # Load and process audio
    audio, sr = processor.load_audio(sample_file)
    logger.info(f"Loaded audio: shape={audio.shape}, sample_rate={sr}")
    
    # Get audio info
    info = processor.get_audio_info(sample_file)
    logger.info(f"Audio info: {json.dumps(info, indent=2, default=str)}")
    
    # Demonstrate preprocessing
    normalized = processor.normalize_audio(audio)
    trimmed = processor.trim_silence(audio)
    segments = processor.segment_audio(audio, segment_length=2.0, overlap=0.5)
    
    logger.info(f"Original length: {len(audio)}, Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    logger.info(f"Trimmed length: {len(trimmed)}, Segments created: {len(segments)}")
    
    return audio, info


def demo_feature_extraction(audio_data_path: str):
    """Demonstrate feature extraction capabilities."""
    logger.info("\n=== Feature Extraction Demo ===")
    
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    # Find a sample audio file
    audio_path = Path(audio_data_path)
    sample_file = None
    
    for dialect_dir in audio_path.iterdir():
        if dialect_dir.is_dir():
            wav_files = list(dialect_dir.glob("*.wav"))
            if wav_files:
                sample_file = str(wav_files[0])
                break
    
    if not sample_file:
        logger.error("No audio files found in dataset")
        return
    
    # Load and process audio
    audio, sr = processor.load_audio(sample_file)
    
    # Extract features
    logger.info("Extracting features...")
    features = extractor.extract_all_features(audio)
    
    # Display some key features
    key_features = {
        'tempo': features.get('tempo'),
        'spectral_centroid_mean': features.get('spectral_centroid_mean'),
        'zero_crossing_rate_mean': features.get('zero_crossing_rate_mean'),
        'rms_mean': features.get('rms_mean'),
        'brightness': features.get('brightness')
    }
    
    logger.info("Extracted features:")
    for key, value in key_features.items():
        if value is not None:
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Extract classification features
    classification_features = extractor.extract_features_for_classification(audio)
    logger.info(f"Classification feature vector shape: {classification_features.shape}")
    
    return features, classification_features


def demo_dataset_exploration(audio_data_path: str):
    """Explore the audio dataset structure."""
    logger.info("\n=== Dataset Exploration ===")
    
    processor = AudioProcessor()
    
    audio_path = Path(audio_data_path)
    dataset_info = {}
    
    logger.info(f"Exploring dataset at: {audio_data_path}")
    
    for dialect_dir in audio_path.iterdir():
        if dialect_dir.is_dir():
            dialect_name = dialect_dir.name
            logger.info(f"\nProcessing {dialect_name} dialect...")
            
            wav_files = list(dialect_dir.glob("*.wav"))
            logger.info(f"Found {len(wav_files)} audio files")
            
            if wav_files:
                # Analyze a few files
                durations = []
                for wav_file in wav_files[:3]:  # Analyze first 3 files
                    try:
                        info = processor.get_audio_info(str(wav_file))
                        durations.append(info.get('duration', 0))
                        logger.info(f"  {wav_file.name}: {info.get('duration', 0):.2f}s")
                    except Exception as e:
                        logger.warning(f"  Error analyzing {wav_file}: {e}")
                
                if durations:
                    avg_duration = np.mean(durations)
                    dataset_info[dialect_name] = {
                        'num_files': len(wav_files),
                        'avg_duration': avg_duration,
                        'total_duration': sum(durations)
                    }
    
    logger.info("\nDataset Summary:")
    for dialect, info in dataset_info.items():
        logger.info(f"  {dialect}: {info['num_files']} files, avg duration: {info['avg_duration']:.2f}s")
    
    return dataset_info


def demo_synthetic_evaluation(real_audio_dir: str, synthetic_audio_dir: str):
    """Demonstrate synthetic audio evaluation."""
    logger.info("\n=== Synthetic Audio Evaluation Demo ===")
    
    evaluator = SyntheticAudioEvaluator()
    
    # Check if directories exist
    real_path = Path(real_audio_dir)
    synthetic_path = Path(synthetic_audio_dir)
    
    if not real_path.exists():
        logger.error(f"Real audio directory not found: {real_audio_dir}")
        return
    
    if not synthetic_path.exists():
        logger.error(f"Synthetic audio directory not found: {synthetic_audio_dir}")
        return
    
    logger.info(f"Evaluating synthetic audio in {synthetic_audio_dir} against real audio in {real_audio_dir}")
    
    try:
        # Evaluate a small subset first
        results = evaluator.evaluate_dataset(real_audio_dir, synthetic_audio_dir)
        
        if 'summary_statistics' in results:
            summary = results['summary_statistics']
            logger.info("Synthetic Audio Evaluation Results:")
            logger.info(f"  Overall Quality Score: {summary.get('overall_quality_score_mean', 'N/A'):.3f}")
            logger.info(f"  MFCC Similarity: {summary.get('mfcc_mean_similarity_mean', 'N/A'):.3f}")
            logger.info(f"  Spectral Distance: {summary.get('cosine_distance_mean', 'N/A'):.3f}")
            logger.info(f"  Total pairs evaluated: {results.get('total_pairs_evaluated', 0)}")
        else:
            logger.warning("No evaluation results available")
            
    except Exception as e:
        logger.error(f"Error during synthetic evaluation: {e}")


def demo_visualization(audio_data_path: str, output_dir: str = "demo_visualizations"):
    """Demonstrate visualization capabilities."""
    logger.info("\n=== Visualization Demo ===")
    
    visualizer = AudioVisualizer()
    processor = AudioProcessor()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find sample audio files from different dialects
    audio_path = Path(audio_data_path)
    sample_files = {}
    
    for dialect_dir in audio_path.iterdir():
        if dialect_dir.is_dir():
            wav_files = list(dialect_dir.glob("*.wav"))
            if wav_files:
                sample_files[dialect_dir.name] = str(wav_files[0])
                if len(sample_files) >= 3:  # Limit to 3 samples
                    break
    
    if not sample_files:
        logger.error("No audio files found for visualization")
        return
    
    logger.info(f"Creating visualizations for: {list(sample_files.keys())}")
    
    # Load audio samples
    audio_samples = []
    for dialect, file_path in sample_files.items():
        try:
            audio, sr = processor.load_audio(file_path)
            audio_samples.append((audio, dialect))
            logger.info(f"Loaded {dialect}: shape={audio.shape}")
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    if not audio_samples:
        logger.error("No audio samples could be loaded")
        return
    
    # Create comparison visualizations
    try:
        # Waveform comparison
        visualizer.plot_comparison(audio_samples, plot_type='waveform',
                                   title='Waveform Comparison',
                                   save_path=f"{output_dir}/waveform_comparison.png")
        
        logger.info(f"Waveform comparison saved to {output_dir}/waveform_comparison.png")
        
    except Exception as e:
        logger.error(f"Error creating waveform comparison: {e}")
    
    # Create individual audio reports for the first sample
    if audio_samples:
        try:
            first_audio, first_dialect = audio_samples[0]
            report_path = visualizer.create_audio_report(
                first_audio, 
                sample_files[first_dialect],
                extractor=FeatureExtractor(),
                save_dir=output_dir
            )
            logger.info(f"Audio report created: {report_path}")
            
        except Exception as e:
            logger.error(f"Error creating audio report: {e}")


def main():
    """Main demo function."""
    logger.info("Starting Thai Dialect Audio Analysis System Demo")
    
    # Use the actual audio data path
    audio_data_path = "audio_data"  # This should point to your extracted audio data
    
    # Check if audio data exists
    if not Path(audio_data_path).exists():
        logger.error(f"Audio data directory not found: {audio_data_path}")
        logger.info("Please ensure audio data is extracted to the 'audio_data' directory")
        return
    
    try:
        # Run demos
        demo_dataset_exploration(audio_data_path)
        demo_audio_processing(audio_data_path)
        demo_feature_extraction(audio_data_path)
        demo_visualization(audio_data_path)
        
        # Test synthetic evaluation if synthetic audio exists
        synthetic_path = Path(audio_data_path) / "synthetic_audio"
        if synthetic_path.exists():
            real_path = Path(audio_data_path) / "raw_audio"
            if real_path.exists():
                demo_synthetic_evaluation(str(real_path), str(synthetic_path))
        
        logger.info("\n=== Demo Completed Successfully ===")
        logger.info("Check the 'demo_visualizations' directory for generated plots and reports")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()