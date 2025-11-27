"""
Audio Visualizer for creating comprehensive audio analysis plots and reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class AudioVisualizer:
    """
    Comprehensive audio visualization and analysis reporting.
    """
    
    def __init__(self, sample_rate: int = 22050, figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the AudioVisualizer.
        
        Args:
            sample_rate: Target sample rate for audio processing
            figure_size: Default figure size for plots
        """
        self.sample_rate = sample_rate
        self.figure_size = figure_size
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_waveform(self, audio_data: np.ndarray, title: str = "Audio Waveform", 
                     save_path: Optional[str] = None) -> None:
        """
        Plot audio waveform.
        
        Args:
            audio_data: Audio data
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
        ax.plot(time_axis, audio_data)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_spectrogram(self, audio_data: np.ndarray, title: str = "Spectrogram",
                        save_path: Optional[str] = None) -> None:
        """
        Plot spectrogram.
        
        Args:
            audio_data: Audio data
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        
        img = librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_mel_spectrogram(self, audio_data: np.ndarray, title: str = "Mel Spectrogram",
                           save_path: Optional[str] = None) -> None:
        """
        Plot mel spectrogram.
        
        Args:
            audio_data: Audio data
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(mel_spec_db, sr=self.sample_rate, 
                                     x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_mfcc(self, audio_data: np.ndarray, title: str = "MFCC Features",
                 save_path: Optional[str] = None) -> None:
        """
        Plot MFCC features.
        
        Args:
            audio_data: Audio data
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        
        img = librosa.display.specshow(mfcc, sr=self.sample_rate, x_axis='time', ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_chroma(self, audio_data: np.ndarray, title: str = "Chroma Features",
                   save_path: Optional[str] = None) -> None:
        """
        Plot chroma features.
        
        Args:
            audio_data: Audio data
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Compute chroma
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        
        img = librosa.display.specshow(chroma, sr=self.sample_rate, x_axis='time', 
                                     y_axis='chroma', ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_comparison(self, audio_list: List[Tuple[np.ndarray, str]], 
                       plot_type: str = 'waveform', title: str = "Audio Comparison",
                       save_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple audio samples.
        
        Args:
            audio_list: List of (audio_data, label) tuples
            plot_type: Type of plot ('waveform', 'spectrogram', 'mfcc')
            title: Plot title
            save_path: Path to save the plot
        """
        n_samples = len(audio_list)
        fig, axes = plt.subplots(n_samples, 1, figsize=(self.figure_size[0], 
                                                          self.figure_size[1] * n_samples))
        
        if n_samples == 1:
            axes = [axes]
            
        for i, (audio_data, label) in enumerate(audio_list):
            ax = axes[i]
            
            if plot_type == 'waveform':
                time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
                ax.plot(time_axis, audio_data)
                ax.set_ylabel('Amplitude')
                if i == n_samples - 1:
                    ax.set_xlabel('Time (s)')
                    
            elif plot_type == 'spectrogram':
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
                img = librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', 
                                             y_axis='hz', ax=ax)
                if i == 0:
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    
            elif plot_type == 'mfcc':
                mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
                img = librosa.display.specshow(mfcc, sr=self.sample_rate, x_axis='time', ax=ax)
                if i == 0:
                    fig.colorbar(img, ax=ax)
                    
            ax.set_title(f"{label}")
            ax.grid(True, alpha=0.3)
            
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_comparison(self, features_dict: Dict[str, List[float]], 
                              title: str = "Feature Comparison",
                              save_path: Optional[str] = None) -> None:
        """
        Plot comparison of extracted features across different audio samples.
        
        Args:
            features_dict: Dictionary of {sample_name: feature_values}
            title: Plot title
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data for plotting
        sample_names = list(features_dict.keys())
        feature_values = list(features_dict.values())
        
        # Bar plot of mean features
        mean_features = [np.mean(values) for values in feature_values]
        ax1.bar(sample_names, mean_features)
        ax1.set_title('Mean Feature Values')
        ax1.set_ylabel('Feature Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Box plot of feature distributions
        ax2.boxplot(feature_values, labels=sample_names)
        ax2.set_title('Feature Distributions')
        ax2.set_ylabel('Feature Value')
        ax2.tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_classification_results(self, y_true: List, y_pred: List, 
                                   class_names: List[str], title: str = "Classification Results",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot classification results including confusion matrix and performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            save_path: Path to save the plot
        """
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=class_names, yticklabels=class_names)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Classification report as heatmap
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :-3].T  # Remove support rows/cols
        sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Classification Metrics')
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        ax3.bar(class_names, per_class_acc)
        ax3.set_title('Per-Class Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
        # Prediction confidence distribution (if probabilities available)
        # This would require storing prediction probabilities
        ax4.text(0.1, 0.5, f"Overall Accuracy: {np.mean(y_true == y_pred):.3f}", 
                transform=ax4.transAxes, fontsize=14)
        ax4.text(0.1, 0.3, f"Total Samples: {len(y_true)}", transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_audio_report(self, audio_data: np.ndarray, audio_path: str,
                          feature_extractor=None, save_dir: str = "audio_reports") -> str:
        """
        Create a comprehensive audio analysis report.
        
        Args:
            audio_data: Audio data
            audio_path: Path to audio file
            feature_extractor: FeatureExtractor instance
            save_dir: Directory to save reports
            
        Returns:
            Path to generated report
        """
        from pathlib import Path
        import datetime
        
        # Create save directory
        Path(save_dir).mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(audio_path).stem
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Waveform
        ax1 = plt.subplot(4, 2, 1)
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
        ax1.plot(time_axis, audio_data)
        ax1.set_title('Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        
        # Spectrogram
        ax2 = plt.subplot(4, 2, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='hz', ax=ax2)
        ax2.set_title('Spectrogram')
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        # Mel spectrogram
        ax3 = plt.subplot(4, 2, 3)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_spec_db, sr=self.sample_rate, x_axis='time', y_axis='mel', ax=ax3)
        ax3.set_title('Mel Spectrogram')
        fig.colorbar(img, ax=ax3, format='%+2.0f dB')
        
        # MFCC
        ax4 = plt.subplot(4, 2, 4)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        img = librosa.display.specshow(mfcc, sr=self.sample_rate, x_axis='time', ax=ax4)
        ax4.set_title('MFCC Features')
        fig.colorbar(img, ax=ax4)
        
        # Chroma
        ax5 = plt.subplot(4, 2, 5)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        img = librosa.display.specshow(chroma, sr=self.sample_rate, x_axis='time', y_axis='chroma', ax=ax5)
        ax5.set_title('Chroma Features')
        fig.colorbar(img, ax=ax5)
        
        # Spectral features
        ax6 = plt.subplot(4, 2, 6)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(spectral_centroid[0]))
        ax6.plot(time_axis, spectral_centroid[0], label='Spectral Centroid')
        ax6.plot(time_axis, spectral_rolloff[0], label='Spectral Rolloff')
        ax6.set_title('Spectral Features')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Frequency (Hz)')
        ax6.legend()
        
        # Zero crossing rate
        ax7 = plt.subplot(4, 2, 7)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        ax7.plot(time_axis[:len(zcr[0])], zcr[0])
        ax7.set_title('Zero Crossing Rate')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('ZCR')
        
        # RMS energy
        ax8 = plt.subplot(4, 2, 8)
        rms = librosa.feature.rms(y=audio_data)
        ax8.plot(time_axis[:len(rms[0])], rms[0])
        ax8.set_title('RMS Energy')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('RMS')
        
        plt.suptitle(f'Audio Analysis Report - {audio_name}', fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        viz_path = f"{save_dir}/{audio_name}_analysis_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        if feature_extractor:
            features = feature_extractor.extract_all_features(audio_data)
            
            report_text = f"""
# Audio Analysis Report

**File**: {audio_path}
**Analysis Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Basic Information
- Duration: {len(audio_data) / self.sample_rate:.2f} seconds
- Sample Rate: {self.sample_rate} Hz
- Total Samples: {len(audio_data)}

## Extracted Features
- Tempo: {features.get('tempo', 'N/A')}
- Spectral Centroid Mean: {features.get('spectral_centroid_mean', 'N/A'):.2f}
- Zero Crossing Rate Mean: {features.get('zero_crossing_rate_mean', 'N/A'):.4f}
- RMS Energy Mean: {features.get('rms_mean', 'N/A'):.4f}
- Brightness: {features.get('brightness', 'N/A'):.2f}

## Audio Characteristics
Based on the extracted features, this audio sample shows:
- {'High' if features.get('spectral_centroid_mean', 0) > 1500 else 'Low' if features.get('spectral_centroid_mean', 0) < 800 else 'Moderate'} frequency content
- {'High' if features.get('zero_crossing_rate_mean', 0) > 0.1 else 'Low' if features.get('zero_crossing_rate_mean', 0) < 0.05 else 'Moderate'} zero crossing rate
- {'High' if features.get('rms_mean', 0) > 0.1 else 'Low' if features.get('rms_mean', 0) < 0.01 else 'Moderate'} energy content

Generated visualization saved to: {viz_path}
"""
            
            # Save text report
            report_path = f"{save_dir}/{audio_name}_report_{timestamp}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            logger.info(f"Audio report generated: {report_path}")
            return report_path
        
        return viz_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    visualizer = AudioVisualizer()
    
    # Test with sample audio
    sample_path = "path/to/your/audio.wav"
    if Path(sample_path).exists():
        # Load audio
        audio, sr = librosa.load(sample_path, sr=22050)
        
        # Create comprehensive report
        report_path = visualizer.create_audio_report(audio, sample_path)
        print(f"Report generated: {report_path}")
    else:
        print(f"Sample file {sample_path} not found")