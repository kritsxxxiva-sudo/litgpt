"""
Synthetic Audio Evaluator for validating generated audio quality
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SyntheticAudioEvaluator:
    """
    Evaluates synthetic audio quality compared to real audio samples.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the SyntheticAudioEvaluator.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
    def compute_spectral_distance(self, real_audio: np.ndarray, synthetic_audio: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral distance metrics between real and synthetic audio.
        
        Args:
            real_audio: Real audio sample
            synthetic_audio: Synthetic audio sample
            
        Returns:
            Dictionary of spectral distance metrics
        """
        # Compute spectrograms
        real_spec = np.abs(librosa.stft(real_audio, hop_length=512))
        synthetic_spec = np.abs(librosa.stft(synthetic_audio, hop_length=512))
        
        # Ensure same dimensions
        min_frames = min(real_spec.shape[1], synthetic_spec.shape[1])
        real_spec = real_spec[:, :min_frames]
        synthetic_spec = synthetic_spec[:, :min_frames]
        
        # Flatten for distance computation
        real_flat = real_spec.flatten()
        synthetic_flat = synthetic_spec.flatten()
        
        # Compute distances
        cosine_dist = cosine(real_flat, synthetic_flat)
        euclidean_dist = euclidean(real_flat, synthetic_flat)
        
        # Correlation
        correlation, _ = pearsonr(real_flat, synthetic_flat)
        
        return {
            'cosine_distance': float(cosine_dist),
            'euclidean_distance': float(euclidean_dist),
            'correlation': float(correlation),
            'spectral_mse': float(np.mean((real_flat - synthetic_flat) ** 2))
        }
    
    def compute_mfcc_similarity(self, real_audio: np.ndarray, synthetic_audio: np.ndarray) -> Dict[str, float]:
        """
        Compute MFCC-based similarity metrics.
        
        Args:
            real_audio: Real audio sample
            synthetic_audio: Synthetic audio sample
            
        Returns:
            Dictionary of MFCC similarity metrics
        """
        # Extract MFCC features
        real_mfcc = librosa.feature.mfcc(y=real_audio, sr=self.sample_rate, n_mfcc=13)
        synthetic_mfcc = librosa.feature.mfcc(y=synthetic_audio, sr=self.sample_rate, n_mfcc=13)
        
        # Ensure same dimensions
        min_frames = min(real_mfcc.shape[1], synthetic_mfcc.shape[1])
        real_mfcc = real_mfcc[:, :min_frames]
        synthetic_mfcc = synthetic_mfcc[:, :min_frames]
        
        # Compute MFCC statistics
        real_mfcc_mean = np.mean(real_mfcc, axis=1)
        real_mfcc_std = np.std(real_mfcc, axis=1)
        synthetic_mfcc_mean = np.mean(synthetic_mfcc, axis=1)
        synthetic_mfcc_std = np.std(synthetic_mfcc, axis=1)
        
        # Compute similarities
        mean_cosine = 1 - cosine(real_mfcc_mean, synthetic_mfcc_mean)
        std_cosine = 1 - cosine(real_mfcc_std, synthetic_mfcc_std)
        
        return {
            'mfcc_mean_similarity': float(mean_cosine),
            'mfcc_std_similarity': float(std_cosine),
            'mfcc_correlation': float(np.corrcoef(real_mfcc_mean, synthetic_mfcc_mean)[0, 1])
        }
    
    def compute_temporal_similarity(self, real_audio: np.ndarray, synthetic_audio: np.ndarray) -> Dict[str, float]:
        """
        Compute temporal similarity metrics.
        
        Args:
            real_audio: Real audio sample
            synthetic_audio: Synthetic audio sample
            
        Returns:
            Dictionary of temporal similarity metrics
        """
        # RMS energy
        real_rms = librosa.feature.rms(y=real_audio, hop_length=512)
        synthetic_rms = librosa.feature.rms(y=synthetic_audio, hop_length=512)
        
        # Zero crossing rate
        real_zcr = librosa.feature.zero_crossing_rate(real_audio, hop_length=512)
        synthetic_zcr = librosa.feature.zero_crossing_rate(synthetic_audio, hop_length=512)
        
        # Ensure same dimensions
        min_frames = min(real_rms.shape[1], synthetic_rms.shape[1])
        real_rms = real_rms[:, :min_frames]
        synthetic_rms = synthetic_rms[:, :min_frames]
        real_zcr = real_zcr[:, :min_frames]
        synthetic_zcr = synthetic_zcr[:, :min_frames]
        
        # Compute similarities
        rms_cosine = 1 - cosine(real_rms.flatten(), synthetic_rms.flatten())
        zcr_cosine = 1 - cosine(real_zcr.flatten(), synthetic_zcr.flatten())
        
        return {
            'rms_similarity': float(rms_cosine),
            'zcr_similarity': float(zcr_cosine),
            'duration_ratio': float(len(synthetic_audio)) / float(len(real_audio))
        }
    
    def evaluate_single_pair(self, real_audio: np.ndarray, synthetic_audio: np.ndarray) -> Dict:
        """
        Evaluate a single pair of real and synthetic audio.
        
        Args:
            real_audio: Real audio sample
            synthetic_audio: Synthetic audio sample
            
        Returns:
            Comprehensive evaluation results
        """
        # Ensure same length
        min_length = min(len(real_audio), len(synthetic_audio))
        real_audio = real_audio[:min_length]
        synthetic_audio = synthetic_audio[:min_length]
        
        # Compute all metrics
        spectral_metrics = self.compute_spectral_distance(real_audio, synthetic_audio)
        mfcc_metrics = self.compute_mfcc_similarity(real_audio, synthetic_audio)
        temporal_metrics = self.compute_temporal_similarity(real_audio, synthetic_audio)
        
        # Combine all metrics
        all_metrics = {
            **spectral_metrics,
            **mfcc_metrics,
            **temporal_metrics
        }
        
        # Compute overall quality score (weighted average)
        quality_score = (
            0.3 * mfcc_metrics['mfcc_mean_similarity'] +
            0.3 * mfcc_metrics['mfcc_std_similarity'] +
            0.2 * temporal_metrics['rms_similarity'] +
            0.2 * (1 - spectral_metrics['cosine_distance'])  # Convert distance to similarity
        )
        
        all_metrics['overall_quality_score'] = float(quality_score)
        
        return all_metrics
    
    def evaluate_dataset(self, real_audio_dir: str, synthetic_audio_dir: str, 
                        file_pattern: str = "*.wav") -> Dict:
        """
        Evaluate synthetic audio against real audio for an entire dataset.
        
        Args:
            real_audio_dir: Directory containing real audio files
            synthetic_audio_dir: Directory containing synthetic audio files  
            file_pattern: File pattern to match
            
        Returns:
            Comprehensive dataset evaluation results
        """
        import glob
        from pathlib import Path
        
        real_files = glob.glob(f"{real_audio_dir}/{file_pattern}")
        synthetic_files = glob.glob(f"{synthetic_audio_dir}/{file_pattern}")
        
        if not real_files or not synthetic_files:
            raise ValueError("No audio files found in specified directories")
        
        results = []
        
        # Match files by name
        real_dict = {Path(f).stem: f for f in real_files}
        synthetic_dict = {Path(f).stem: f for f in synthetic_files}
        
        # Find matching pairs
        matched_pairs = []
        for name in real_dict:
            # Look for corresponding synthetic file
            for synth_name in synthetic_dict:
                if name in synth_name or synth_name.replace('_synthetic', '').replace('_interpolation', '') == name:
                    matched_pairs.append((real_dict[name], synthetic_dict[synth_name]))
                    break
        
        logger.info(f"Found {len(matched_pairs)} matching audio pairs")
        
        for real_path, synthetic_path in matched_pairs:
            try:
                # Load audio files
                real_audio, _ = librosa.load(real_path, sr=self.sample_rate)
                synthetic_audio, _ = librosa.load(synthetic_path, sr=self.sample_rate)
                
                # Evaluate pair
                pair_results = self.evaluate_single_pair(real_audio, synthetic_audio)
                pair_results['real_file'] = real_path
                pair_results['synthetic_file'] = synthetic_path
                
                results.append(pair_results)
                
            except Exception as e:
                logger.warning(f"Error evaluating pair {real_path} - {synthetic_path}: {e}")
                continue
        
        # Compute summary statistics
        if results:
            summary_stats = self._compute_summary_statistics(results)
            return {
                'individual_results': results,
                'summary_statistics': summary_stats,
                'total_pairs_evaluated': len(results)
            }
        else:
            return {'error': 'No valid pairs could be evaluated'}
    
    def _compute_summary_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute summary statistics from individual evaluation results.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Summary statistics
        """
        import pandas as pd
        
        # Convert to DataFrame for easier statistics
        df = pd.DataFrame(results)
        
        # Compute mean and std for each metric
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_columns:
            summary[f"{col}_mean"] = float(df[col].mean())
            summary[f"{col}_std"] = float(df[col].std())
            summary[f"{col}_min"] = float(df[col].min())
            summary[f"{col}_max"] = float(df[col].max())
        
        return summary
    
    def create_evaluation_report(self, evaluation_results: Dict, output_path: str) -> None:
        """
        Create a comprehensive evaluation report with visualizations.
        
        Args:
            evaluation_results: Results from evaluate_dataset
            output_path: Path to save the report
        """
        if 'individual_results' not in evaluation_results:
            logger.error("No individual results found in evaluation_results")
            return
        
        results = evaluation_results['individual_results']
        summary = evaluation_results['summary_statistics']
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Synthetic Audio Quality Evaluation Report', fontsize=16)
        
        # Extract key metrics for plotting
        quality_scores = [r['overall_quality_score'] for r in results]
        mfcc_similarities = [r['mfcc_mean_similarity'] for r in results]
        spectral_distances = [r['cosine_distance'] for r in results]
        rms_similarities = [r['rms_similarity'] for r in results]
        
        # Plot 1: Overall Quality Score Distribution
        axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, color='blue')
        axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(quality_scores):.3f}')
        axes[0, 0].set_title('Overall Quality Score Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Plot 2: MFCC Similarity vs Spectral Distance
        axes[0, 1].scatter(mfcc_similarities, spectral_distances, alpha=0.6)
        axes[0, 1].set_xlabel('MFCC Similarity')
        axes[0, 1].set_ylabel('Spectral Cosine Distance')
        axes[0, 1].set_title('MFCC Similarity vs Spectral Distance')
        
        # Plot 3: RMS Similarity Distribution
        axes[1, 0].hist(rms_similarities, bins=20, alpha=0.7, color='green')
        axes[1, 0].axvline(np.mean(rms_similarities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(rms_similarities):.3f}')
        axes[1, 0].set_title('RMS Similarity Distribution')
        axes[1, 0].set_xlabel('RMS Similarity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Plot 4: Correlation Matrix of Key Metrics
        metrics_matrix = np.array([
            [r['mfcc_mean_similarity'], r['mfcc_std_similarity'], 
             r['rms_similarity'], r['overall_quality_score']]
            for r in results
        ])
        corr_matrix = np.corrcoef(metrics_matrix.T)
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=['MFCC Mean', 'MFCC Std', 'RMS Sim', 'Quality'],
                   yticklabels=['MFCC Mean', 'MFCC Std', 'RMS Sim', 'Quality'],
                   ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(f"{output_path}_visualizations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary report
        report_content = f"""
# Synthetic Audio Quality Evaluation Report

## Summary Statistics

- **Total Audio Pairs Evaluated**: {evaluation_results['total_pairs_evaluated']}
- **Overall Quality Score (Mean ± Std)**: {summary['overall_quality_score_mean']:.3f} ± {summary['overall_quality_score_std']:.3f}
- **MFCC Mean Similarity**: {summary['mfcc_mean_similarity_mean']:.3f} ± {summary['mfcc_mean_similarity_std']:.3f}
- **Spectral Cosine Distance**: {summary['cosine_distance_mean']:.3f} ± {summary['cosine_distance_std']:.3f}
- **RMS Similarity**: {summary['rms_similarity_mean']:.3f} ± {summary['rms_similarity_std']:.3f}

## Key Findings

1. **Quality Assessment**: The synthetic audio achieves an average quality score of {summary['overall_quality_score_mean']:.3f} (scale: 0-1, higher is better).

2. **Spectral Similarity**: The spectral cosine distance of {summary['cosine_distance_mean']:.3f} indicates {'good' if summary['cosine_distance_mean'] < 0.3 else 'moderate' if summary['cosine_distance_mean'] < 0.6 else 'poor'} spectral similarity.

3. **MFCC Features**: MFCC similarity scores of {summary['mfcc_mean_similarity_mean']:.3f} suggest {'high' if summary['mfcc_mean_similarity_mean'] > 0.7 else 'moderate' if summary['mfcc_mean_similarity_mean'] > 0.4 else 'low'} similarity in perceptually relevant features.

## Recommendations

Based on the evaluation results:
- {'✓' if summary['overall_quality_score_mean'] > 0.6 else '⚠'} The synthetic audio quality is {'suitable' if summary['overall_quality_score_mean'] > 0.6 else 'needs improvement'} for most applications.
- {'✓' if summary['mfcc_mean_similarity_mean'] > 0.7 else '⚠'} MFCC features show {'excellent' if summary['mfcc_mean_similarity_mean'] > 0.7 else 'adequate' if summary['mfcc_mean_similarity_mean'] > 0.4 else 'poor'} perceptual similarity.
- {'✓' if summary['cosine_distance_mean'] < 0.4 else '⚠'} Spectral characteristics are {'well preserved' if summary['cosine_distance_mean'] < 0.4 else 'moderately preserved' if summary['cosine_distance_mean'] < 0.7 else 'poorly preserved'}.
"""
        
        with open(f"{output_path}_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    evaluator = SyntheticAudioEvaluator()
    
    # Evaluate dataset
    real_dir = "path/to/real/audio"
    synthetic_dir = "path/to/synthetic/audio"
    
    if Path(real_dir).exists() and Path(synthetic_dir).exists():
        results = evaluator.evaluate_dataset(real_dir, synthetic_dir)
        
        # Create report
        output_path = "synthetic_audio_evaluation"
        evaluator.create_evaluation_report(results, output_path)
        
        print("Evaluation completed successfully!")
    else:
        print("Audio directories not found. Please check paths.")