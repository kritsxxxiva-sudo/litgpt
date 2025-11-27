"""
Dialect Classifier for Thai language variants using machine learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .feature_extractor import FeatureExtractor
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class DialectClassifier:
    """
    Multi-class classifier for Thai dialects and related languages.
    Supports: Phuthai, Toei, Kaleang, Khmer, Lao
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the DialectClassifier.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.scaler = StandardScaler()
        self.model = None
        self.label_mapping = {
            'phuthai': 0,
            'toei': 1,
            'kaleang': 2,
            'khmer': 3,
            'lao': 4
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def extract_features_from_dataset(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from all audio files in the dataset.
        
        Args:
            data_path: Path to the dataset directory
            
        Returns:
            Tuple of (features, labels)
        """
        data_path = Path(data_path)
        features = []
        labels = []
        
        # Process each dialect directory
        for dialect_dir in data_path.iterdir():
            if dialect_dir.is_dir() and dialect_dir.name in self.label_mapping:
                dialect_label = self.label_mapping[dialect_dir.name]
                logger.info(f"Processing {dialect_dir.name} dialect...")
                
                # Process each audio file
                for audio_file in dialect_dir.glob("*.wav"):
                    try:
                        # Load and preprocess audio
                        audio, _ = self.audio_processor.load_audio(str(audio_file))
                        audio = self.audio_processor.normalize_audio(audio)
                        audio = self.audio_processor.trim_silence(audio)
                        
                        # Extract features
                        feature_vector = self.feature_extractor.extract_features_for_classification(audio)
                        features.append(feature_vector)
                        labels.append(dialect_label)
                        
                    except Exception as e:
                        logger.warning(f"Error processing {audio_file}: {e}")
                        continue
        
        return np.array(features), np.array(labels)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   model_type: str = 'random_forest', cv_folds: int = 5) -> Dict:
        """
        Train the classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model ('random_forest', 'gradient_boosting', 'svm')
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Select and configure model
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
        elif model_type == 'svm':
            base_model = SVC(random_state=42, probability=True)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform grid search with cross-validation
        logger.info(f"Training {model_type} model with grid search...")
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_type': model_type
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=list(self.label_mapping.keys()),
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=list(self.label_mapping.keys()))}")
        
        return results
    
    def predict_dialect(self, audio_path: str) -> Dict:
        """
        Predict the dialect of a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Load and preprocess audio
        audio, _ = self.audio_processor.load_audio(audio_path)
        audio = self.audio_processor.normalize_audio(audio)
        audio = self.audio_processor.trim_silence(audio)
        
        # Extract features
        features = self.feature_extractor.extract_features_for_classification(audio)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top predictions
        top_predictions = []
        for i, prob in enumerate(probabilities):
            top_predictions.append({
                'dialect': self.reverse_mapping[i],
                'probability': float(prob)
            })
        
        # Sort by probability
        top_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'predicted_dialect': self.reverse_mapping[prediction],
            'confidence': float(probabilities[prediction]),
            'all_probabilities': top_predictions
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_mapping': self.label_mapping,
            'sample_rate': self.sample_rate
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_mapping = model_data['label_mapping']
        self.sample_rate = model_data['sample_rate']
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        logger.info(f"Model loaded from {model_path}")


def main():
    """
    Main function for training and evaluating the dialect classifier.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Thai Dialect Classifier')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='random_forest', 
                       choices=['random_forest', 'gradient_boosting', 'svm'],
                       help='Type of model to train')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--model_path', type=str, default='dialect_classifier.pkl',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = DialectClassifier()
    
    # Extract features
    logger.info("Extracting features from dataset...")
    X, y = classifier.extract_features_from_dataset(args.data_path)
    logger.info(f"Extracted features from {len(X)} audio files")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    train_results = classifier.train_model(X_train, y_train, args.model_type)
    
    # Evaluate model
    eval_results = classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model(args.model_path)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()