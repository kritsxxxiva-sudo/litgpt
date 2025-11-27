#!/usr/bin/env python3
"""
Thai Dialect Audio Analysis Web Interface
Interactive web application for exploring and analyzing Thai dialect audio datasets.
"""

import os
import json
import base64
import io
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.utils import secure_filename
import numpy as np
from audio_analysis.audio_processor import AudioProcessor
from audio_analysis.feature_extractor import FeatureExtractor
from audio_analysis.visualizer import AudioVisualizer as Visualizer
from audio_analysis.classifier import DialectClassifier
from audio_analysis.synthetic_evaluator import SyntheticAudioEvaluator as SyntheticEvaluator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thai-dialect-audio-analysis'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize analysis components
audio_processor = AudioProcessor()
feature_extractor = FeatureExtractor()
visualizer = Visualizer()
classifier = DialectClassifier()
synthetic_evaluator = SyntheticEvaluator()

# Configuration
AUDIO_DATA_DIR = 'audio_data'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_dialect_stats():
    """Get statistics about the available dialect data."""
    stats = {
        'raw_audio': {},
        'synthetic_audio': {},
        'spectrograms': {}
    }
    
    # Raw audio statistics
    raw_audio_dir = os.path.join(AUDIO_DATA_DIR, 'raw_audio')
    if os.path.exists(raw_audio_dir):
        for dialect in os.listdir(raw_audio_dir):
            dialect_path = os.path.join(raw_audio_dir, dialect)
            if os.path.isdir(dialect_path):
                audio_files = [f for f in os.listdir(dialect_path) if f.endswith(('.wav', '.mp3'))]
                stats['raw_audio'][dialect] = len(audio_files)
    
    # Synthetic audio statistics
    synthetic_dir = os.path.join(AUDIO_DATA_DIR, 'synthetic_audio')
    if os.path.exists(synthetic_dir):
        synthetic_files = [f for f in os.listdir(synthetic_dir) if f.endswith('.wav')]
        stats['synthetic_audio']['total'] = len(synthetic_files)
        
        # Count by dialect
        for filename in synthetic_files:
            if 'phuthai' in filename:
                stats['synthetic_audio']['phuthai'] = stats['synthetic_audio'].get('phuthai', 0) + 1
            elif 'toei' in filename:
                stats['synthetic_audio']['toei'] = stats['synthetic_audio'].get('toei', 0) + 1
            elif 'kaleang' in filename:
                stats['synthetic_audio']['kaleang'] = stats['synthetic_audio'].get('kaleang', 0) + 1
            elif 'khmer' in filename:
                stats['synthetic_audio']['khmer'] = stats['synthetic_audio'].get('khmer', 0) + 1
            elif 'lao' in filename:
                stats['synthetic_audio']['lao'] = stats['synthetic_audio'].get('lao', 0) + 1
    
    # Spectrogram statistics
    spectrograms_dir = os.path.join(AUDIO_DATA_DIR, 'spectrograms')
    if os.path.exists(spectrograms_dir):
        for dialect in os.listdir(spectrograms_dir):
            dialect_path = os.path.join(spectrograms_dir, dialect)
            if os.path.isdir(dialect_path):
                spec_files = [f for f in os.listdir(dialect_path) if f.endswith('.png')]
                stats['spectrograms'][dialect] = len(spec_files)
    
    return stats


@app.route('/')
def index():
    """Main dashboard page."""
    stats = get_dialect_stats()
    return render_template('index.html', stats=stats)


@app.route('/api/stats')
def api_stats():
    """API endpoint for dataset statistics."""
    stats = get_dialect_stats()
    return jsonify(stats)


@app.route('/api/dialects')
def api_dialects():
    """Get available dialects."""
    dialects = classifier.get_supported_dialects()
    return jsonify({'dialects': dialects})


@app.route('/api/audio-files/<data_type>')
def api_audio_files(data_type):
    """Get list of audio files for a specific data type."""
    if data_type not in ['raw_audio', 'synthetic_audio']:
        return jsonify({'error': 'Invalid data type'}), 400
    
    audio_dir = os.path.join(AUDIO_DATA_DIR, data_type)
    files = []
    
    if os.path.exists(audio_dir):
        if data_type == 'raw_audio':
            # Organized by dialect subdirectories
            for dialect in os.listdir(audio_dir):
                dialect_path = os.path.join(audio_dir, dialect)
                if os.path.isdir(dialect_path):
                    for filename in os.listdir(dialect_path):
                        if filename.endswith(('.wav', '.mp3')):
                            files.append({
                                'filename': filename,
                                'dialect': dialect,
                                'path': os.path.join(dialect, filename)
                            })
        else:
            # Flat structure for synthetic audio
            for filename in os.listdir(audio_dir):
                if filename.endswith('.wav'):
                    files.append({
                        'filename': filename,
                        'path': filename
                    })
    
    return jsonify({'files': files})


@app.route('/api/audio-file/<data_type>')
def api_audio_file(data_type):
    """Serve audio file."""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    if data_type == 'raw_audio':
        dialect = request.args.get('dialect')
        if not dialect:
            return jsonify({'error': 'Dialect required for raw audio'}), 400
        file_path = os.path.join(AUDIO_DATA_DIR, data_type, dialect, filename)
    else:
        file_path = os.path.join(AUDIO_DATA_DIR, data_type, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path)


@app.route('/api/analyze-audio', methods=['POST'])
def api_analyze_audio():
    """Analyze uploaded or existing audio file."""
    try:
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        else:
            # Handle existing file analysis
            data = request.get_json()
            data_type = data.get('data_type')
            filename = data.get('filename')
            
            if not all([data_type, filename]):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            if data_type == 'raw_audio':
                dialect = data.get('dialect')
                if not dialect:
                    return jsonify({'error': 'Dialect required for raw audio'}), 400
                filepath = os.path.join(AUDIO_DATA_DIR, data_type, dialect, filename)
            else:
                filepath = os.path.join(AUDIO_DATA_DIR, data_type, filename)
            
            if not os.path.exists(filepath):
                return jsonify({'error': 'File not found'}), 404
        
        # Analyze the audio
        audio_data, sample_rate = audio_processor.load_audio(filepath)
        
        # Extract features
        features = feature_extractor.extract_features(audio_data, sample_rate)
        
        # Classify dialect
        dialect_prediction = classifier.classify(audio_data, sample_rate)
        
        # Generate visualizations
        waveform_buffer = io.BytesIO()
        spectrogram_buffer = io.BytesIO()
        
        visualizer.plot_waveform(audio_data, sample_rate, save_path=waveform_buffer)
        visualizer.plot_spectrogram(audio_data, sample_rate, save_path=spectrogram_buffer)
        
        # Convert to base64 for web display
        waveform_b64 = base64.b64encode(waveform_buffer.getvalue()).decode()
        spectrogram_b64 = base64.b64encode(spectrogram_buffer.getvalue()).decode()
        
        # Clean up uploaded file
        if 'file' in request.files:
            os.remove(filepath)
        
        return jsonify({
            'features': features,
            'dialect_prediction': dialect_prediction,
            'visualizations': {
                'waveform': f'data:image/png;base64,{waveform_b64}',
                'spectrogram': f'data:image/png;base64,{spectrogram_b64}'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-audio', methods=['POST'])
def api_compare_audio():
    """Compare two audio files."""
    try:
        data = request.get_json()
        file1_info = data.get('file1')
        file2_info = data.get('file2')
        
        if not all([file1_info, file2_info]):
            return jsonify({'error': 'Both file1 and file2 required'}), 400
        
        # Load first audio file
        if file1_info['type'] == 'raw_audio':
            path1 = os.path.join(AUDIO_DATA_DIR, file1_info['type'], file1_info['dialect'], file1_info['filename'])
        else:
            path1 = os.path.join(AUDIO_DATA_DIR, file1_info['type'], file1_info['filename'])
        
        # Load second audio file
        if file2_info['type'] == 'raw_audio':
            path2 = os.path.join(AUDIO_DATA_DIR, file2_info['type'], file2_info['dialect'], file2_info['filename'])
        else:
            path2 = os.path.join(AUDIO_DATA_DIR, file2_info['type'], file2_info['filename'])
        
        audio1, sr1 = audio_processor.load_audio(path1)
        audio2, sr2 = audio_processor.load_audio(path2)
        
        # Extract features for both
        features1 = feature_extractor.extract_features(audio1, sr1)
        features2 = feature_extractor.extract_features(audio2, sr2)
        
        # Calculate similarity
        similarity = synthetic_evaluator.calculate_similarity(features1, features2)
        
        # Generate comparison visualizations
        comparison_buffer = io.BytesIO()
        visualizer.plot_comparison(features1, features2, save_path=comparison_buffer)
        comparison_b64 = base64.b64encode(comparison_buffer.getvalue()).decode()
        
        return jsonify({
            'similarity': similarity,
            'features1': features1,
            'features2': features2,
            'comparison_chart': f'data:image/png;base64,{comparison_b64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/synthetic-evaluation', methods=['POST'])
def api_synthetic_evaluation():
    """Evaluate synthetic audio quality."""
    try:
        data = request.get_json()
        synthetic_info = data.get('synthetic_file')
        reference_info = data.get('reference_file')
        
        if not all([synthetic_info, reference_info]):
            return jsonify({'error': 'Both synthetic and reference files required'}), 400
        
        # Load synthetic audio
        synthetic_path = os.path.join(AUDIO_DATA_DIR, synthetic_info['type'], synthetic_info['filename'])
        synthetic_audio, sr_synth = audio_processor.load_audio(synthetic_path)
        
        # Load reference audio
        if reference_info['type'] == 'raw_audio':
            reference_path = os.path.join(AUDIO_DATA_DIR, reference_info['type'], reference_info['dialect'], reference_info['filename'])
        else:
            reference_path = os.path.join(AUDIO_DATA_DIR, reference_info['type'], reference_info['filename'])
        
        reference_audio, sr_ref = audio_processor.load_audio(reference_path)
        
        # Evaluate synthetic audio
        evaluation_results = synthetic_evaluator.evaluate_synthetic_audio(
            synthetic_audio, reference_audio, sr_synth
        )
        
        # Generate evaluation visualizations
        eval_buffer = io.BytesIO()
        visualizer.plot_evaluation_results(evaluation_results, save_path=eval_buffer)
        eval_b64 = base64.b64encode(eval_buffer.getvalue()).decode()
        
        return jsonify({
            'evaluation': evaluation_results,
            'evaluation_chart': f'data:image/png;base64,{eval_b64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🎵 Thai Dialect Audio Analysis Web Interface")
    print("=" * 50)
    print(f"Audio data directory: {AUDIO_DATA_DIR}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("=" * 50)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)