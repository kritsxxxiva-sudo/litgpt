"""
Main application script for Thai Dialect Audio Analysis System

This script provides a complete pipeline for:
1. Audio preprocessing and feature extraction
2. Dialect classification model training and evaluation  
3. Synthetic audio quality assessment
4. Comprehensive visualization and reporting
"""

import argparse
import logging
import json
import numpy as np
from pathlib import Path
import sys

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from audio_analysis import (
    AudioProcessor,
    FeatureExtractor, 
    DialectClassifier,
    SyntheticAudioEvaluator,
    AudioVisualizer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: str = "audio_analysis_results") -> dict:
    """Create necessary directories for outputs."""
    directories = {
        'base': base_dir,
        'models': f"{base_dir}/models",
        'reports': f"{base_dir}/reports", 
        'visualizations': f"{base_dir}/visualizations",
        'features': f"{base_dir}/features"
    }
    
    for dir_path in directories.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    return directories


def extract_features_dataset(audio_data_path: str, output_dir: str) -> dict:
    """Extract features from the entire audio dataset."""
    logger.info("Starting feature extraction from dataset...")
    
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    audio_path = Path(audio_data_path)
    features_data = {}
    
    # Process each dialect/language directory
    for dialect_dir in audio_path.iterdir():
        if dialect_dir.is_dir():
            dialect_name = dialect_dir.name
            logger.info(f"Processing {dialect_name} dialect...")
            
            features_data[dialect_name] = []
            
            # Process each audio file
            for audio_file in dialect_dir.glob("*.wav"):
                try:
                    # Load and preprocess
                    audio, sr = processor.load_audio(str(audio_file))
                    audio = processor.normalize_audio(audio)
                    audio = processor.trim_silence(audio)
                    
                    # Extract features
                    features = extractor.extract_all_features(audio)
                    features['filename'] = audio_file.name
                    features['dialect'] = dialect_name
                    
                    features_data[dialect_name].append(features)
                    
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")
                    continue
    
    # Save features to JSON
    features_file = f"{output_dir}/extracted_features.json"
    with open(features_file, 'w', encoding='utf-8') as f:
        json.dump(features_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Features extracted and saved to {features_file}")
    return features_data


def train_classifier(data_path: str, model_output_path: str, test_size: float = 0.2) -> dict:
    """Train the dialect classifier model."""
    logger.info("Training dialect classifier...")
    
    classifier = DialectClassifier()
    
    # Extract features
    X, y = classifier.extract_features_from_dataset(data_path)
    
    if len(X) == 0:
        raise ValueError("No features extracted. Check your data path and audio files.")
    
    logger.info(f"Extracted features from {len(X)} audio files")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train multiple models and select best
    model_types = ['random_forest', 'gradient_boosting']
    best_model = None
    best_score = 0
    best_results = None
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        train_results = classifier.train_model(X_train, y_train, model_type)
        eval_results = classifier.evaluate_model(X_test, y_test)
        
        accuracy = eval_results['accuracy']
        logger.info(f"{model_type} accuracy: {accuracy:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model_type
            best_results = {
                'model_type': model_type,
                'training_results': train_results,
                'evaluation_results': eval_results
            }
    
    logger.info(f"Best model: {best_model} with accuracy: {best_score:.4f}")
    
    # Save the best model
    classifier.save_model(model_output_path)
    logger.info(f"Model saved to {model_output_path}")
    
    return best_results


def evaluate_synthetic_audio(real_audio_dir: str, synthetic_audio_dir: str, 
                           output_dir: str) -> dict:
    """Evaluate synthetic audio quality against real audio."""
    logger.info("Evaluating synthetic audio quality...")
    
    evaluator = SyntheticAudioEvaluator()
    visualizer = AudioVisualizer()
    
    # Evaluate dataset
    evaluation_results = evaluator.evaluate_dataset(real_audio_dir, synthetic_audio_dir)
    
    # Generate comprehensive report
    report_path = f"{output_dir}/synthetic_audio_evaluation"
    evaluator.create_evaluation_report(evaluation_results, report_path)
    
    logger.info(f"Synthetic audio evaluation completed. Report saved to {report_path}")
    
    return evaluation_results


def create_comprehensive_report(audio_data_path: str, results_dir: str) -> None:
    """Create a comprehensive analysis report."""
    logger.info("Creating comprehensive analysis report...")
    
    visualizer = AudioVisualizer()
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    # Sample a few audio files for detailed analysis
    audio_path = Path(audio_data_path)
    sample_files = []
    
    for dialect_dir in audio_path.iterdir():
        if dialect_dir.is_dir():
            wav_files = list(dialect_dir.glob("*.wav"))
            if wav_files:
                sample_files.append(str(wav_files[0]))
                if len(sample_files) >= 3:  # Limit to 3 samples
                    break
    
    # Create individual audio reports
    for audio_file in sample_files:
        try:
            audio, sr = processor.load_audio(audio_file)
            report_path = visualizer.create_audio_report(audio, audio_file, extractor, results_dir)
            logger.info(f"Created report for {audio_file}")
        except Exception as e:
            logger.warning(f"Error creating report for {audio_file}: {e}")
            continue
    
    # Create summary statistics
    summary_stats = {}
    for dialect_dir in audio_path.iterdir():
        if dialect_dir.is_dir():
            dialect_name = dialect_dir.name
            durations = []
            
            for audio_file in dialect_dir.glob("*.wav"):
                try:
                    info = processor.get_audio_info(str(audio_file))
                    durations.append(info.get('duration', 0))
                except:
                    continue
            
            if durations:
                summary_stats[dialect_name] = {
                    'num_files': len(durations),
                    'avg_duration': np.mean(durations),
                    'total_duration': np.sum(durations)
                }
    
    # Save summary
    summary_file = f"{results_dir}/dataset_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Comprehensive report created. Summary saved to {summary_file}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Thai Dialect Audio Analysis System')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['extract_features', 'train_model', 'evaluate_synthetic', 'full_pipeline'],
                       help='Operation mode')
    parser.add_argument('--data_path', type=str, 
                       help='Path to audio dataset')
    parser.add_argument('--real_audio_dir', type=str,
                       help='Directory with real audio files (for synthetic evaluation)')
    parser.add_argument('--synthetic_audio_dir', type=str,
                       help='Directory with synthetic audio files (for synthetic evaluation)')
    parser.add_argument('--model_output', type=str, default='models/dialect_classifier.pkl',
                       help='Path to save trained model')
    parser.add_argument('--results_dir', type=str, default='audio_analysis_results',
                       help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size for model training')
    
    args = parser.parse_args()
    
    # Setup directories
    directories = setup_directories(args.results_dir)
    
    try:
        if args.mode == 'extract_features':
            if not args.data_path:
                raise ValueError("--data_path required for feature extraction")
            extract_features_dataset(args.data_path, directories['features'])
            
        elif args.mode == 'train_model':
            if not args.data_path:
                raise ValueError("--data_path required for model training")
            results = train_classifier(args.data_path, args.model_output, args.test_size)
            
            # Save training results
            results_file = f"{directories['reports']}/training_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
        elif args.mode == 'evaluate_synthetic':
            if not args.real_audio_dir or not args.synthetic_audio_dir:
                raise ValueError("--real_audio_dir and --synthetic_audio_dir required")
            evaluate_synthetic_audio(args.real_audio_dir, args.synthetic_audio_dir, 
                                     directories['reports'])
                                     
        elif args.mode == 'full_pipeline':
            if not args.data_path:
                raise ValueError("--data_path required for full pipeline")
            
            logger.info("Running full analysis pipeline...")
            
            # Step 1: Extract features
            extract_features_dataset(args.data_path, directories['features'])
            
            # Step 2: Train classifier
            results = train_classifier(args.data_path, args.model_output, args.test_size)
            
            # Step 3: Create comprehensive report
            create_comprehensive_report(args.data_path, directories['reports'])
            
            # Step 4: Evaluate synthetic audio if directories provided
            if args.real_audio_dir and args.synthetic_audio_dir:
                evaluate_synthetic_audio(args.real_audio_dir, args.synthetic_audio_dir,
                                       directories['reports'])
            
            logger.info("Full pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()