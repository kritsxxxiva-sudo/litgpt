/**
 * Thai Dialect Audio Analysis System - Frontend JavaScript
 * Interactive web interface for audio dataset exploration and analysis
 */

// Global variables
let selectedFile = null;
let currentStats = null;
let dialects = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadDialects();
    setupEventListeners();
    initializeExplorer();
});

/**
 * Load dataset statistics
 */
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        currentStats = stats;
        displayStats(stats);
    } catch (error) {
        console.error('Error loading stats:', error);
        showAlert('Error loading dataset statistics', 'danger');
    }
}

/**
 * Display statistics in the dashboard
 */
function displayStats(stats) {
    const container = document.getElementById('stats-container');
    
    // Calculate totals
    const rawTotal = Object.values(stats.raw_audio).reduce((a, b) => a + b, 0);
    const syntheticTotal = Object.values(stats.synthetic_audio).reduce((a, b) => a + b, 0);
    const spectrogramTotal = Object.values(stats.spectrograms).reduce((a, b) => a + b, 0);
    
    const statsHtml = `
        <div class="col-md-3 mb-3">
            <div class="card stats-card">
                <div class="card-body">
                    <span class="stats-number">${rawTotal}</span>
                    <span class="stats-label">Raw Audio Files</span>
                </div>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="card stats-card">
                <div class="card-body">
                    <span class="stats-number">${syntheticTotal}</span>
                    <span class="stats-label">Synthetic Audio Files</span>
                </div>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="card stats-card">
                <div class="card-body">
                    <span class="stats-number">${spectrogramTotal}</span>
                    <span class="stats-label">Spectrograms</span>
                </div>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="card stats-card">
                <div class="card-body">
                    <span class="stats-number">${dialects.length}</span>
                    <span class="stats-label">Dialects</span>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = statsHtml;
}

/**
 * Load available dialects
 */
async function loadDialects() {
    try {
        const response = await fetch('/api/dialects');
        const data = await response.json();
        dialects = data.dialects;
        populateDialectSelects();
    } catch (error) {
        console.error('Error loading dialects:', error);
        showAlert('Error loading dialects', 'danger');
    }
}

/**
 * Populate dialect select elements
 */
function populateDialectSelects() {
    const dialectSelects = [
        'dialect-select',
        'compare1-dialect',
        'compare2-dialect'
    ];
    
    dialectSelects.forEach(selectId => {
        const select = document.getElementById(selectId);
        if (select) {
            // Clear existing options except the first one
            while (select.options.length > 1) {
                select.remove(1);
            }
            
            // Add dialect options
            dialects.forEach(dialect => {
                const option = document.createElement('option');
                option.value = dialect;
                option.textContent = dialect.charAt(0).toUpperCase() + dialect.slice(1);
                select.appendChild(option);
            });
        }
    });
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Data type selection changes
    document.getElementById('data-type-select').addEventListener('change', handleDataTypeChange);
    
    // Compare type changes
    document.getElementById('compare1-type').addEventListener('change', () => handleCompareTypeChange(1));
    document.getElementById('compare2-type').addEventListener('change', () => handleCompareTypeChange(2));
    
    // File input for analysis
    document.getElementById('audio-file-input').addEventListener('change', handleFileSelect);
}

/**
 * Handle data type selection change
 */
function handleDataTypeChange() {
    const dataType = document.getElementById('data-type-select').value;
    const dialectGroup = document.getElementById('dialect-selection');
    
    if (dataType === 'synthetic_audio') {
        dialectGroup.style.display = 'none';
    } else {
        dialectGroup.style.display = 'block';
    }
}

/**
 * Handle compare type change
 */
function handleCompareTypeChange(compareNum) {
    const type = document.getElementById(`compare${compareNum}-type`).value;
    const dialectGroup = document.getElementById(`compare${compareNum}-dialect-group`);
    
    if (type === 'synthetic_audio') {
        dialectGroup.style.display = 'none';
    } else {
        dialectGroup.style.display = 'block';
    }
    
    // Load available files for the selected type
    loadCompareFiles(compareNum);
}

/**
 * Handle file selection for analysis
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        // Enable analyze button
        const analyzeBtn = document.querySelector('[onclick="analyzeUploadedFile()"]');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }
    }
}

/**
 * Initialize the audio explorer
 */
function initializeExplorer() {
    // Set up initial state
    handleDataTypeChange();
    handleCompareTypeChange(1);
    handleCompareTypeChange(2);
}

/**
 * Load audio files based on selection
 */
async function loadAudioFiles() {
    const dataType = document.getElementById('data-type-select').value;
    const dialect = document.getElementById('dialect-select').value;
    const container = document.getElementById('audio-files-container');
    
    container.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Loading files...</div>';
    
    try {
        const response = await fetch(`/api/audio-files/${dataType}`);
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        let files = data.files;
        
        // Filter by dialect if specified
        if (dialect && dataType === 'raw_audio') {
            files = files.filter(file => file.dialect === dialect);
        }
        
        displayAudioFiles(files, dataType);
        
    } catch (error) {
        console.error('Error loading audio files:', error);
        container.innerHTML = `<div class="alert alert-danger">Error loading files: ${error.message}</div>`;
    }
}

/**
 * Display audio files in the container
 */
function displayAudioFiles(files, dataType) {
    const container = document.getElementById('audio-files-container');
    
    if (files.length === 0) {
        container.innerHTML = '<p class="text-muted">No audio files found for the selected criteria.</p>';
        return;
    }
    
    let html = '<div class="file-list">';
    
    files.forEach(file => {
        const fileId = `file-${dataType}-${file.filename.replace(/[^a-zA-Z0-9]/g, '')}`;
        const badgeType = dataType === 'raw_audio' ? 'raw' : 'synthetic';
        const badgeClass = dataType === 'raw_audio' ? 'badge-raw' : 'badge-synthetic';
        
        html += `
            <div class="file-item" id="${fileId}" onclick="selectFile('${fileId}', '${dataType}', '${file.filename}', '${file.dialect || ''}')">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="file-name">${file.filename}</div>
                        ${file.dialect ? `<div class="file-info"><span class="badge-dialect">${file.dialect}</span></div>` : ''}
                    </div>
                    <div>
                        <span class="badge ${badgeClass}">${badgeType}</span>
                        <button class="btn btn-sm btn-outline-primary ms-2" onclick="playAudio('${dataType}', '${file.filename}', '${file.dialect || ''}', event)">
                            <i class="fas fa-play"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

/**
 * Select a file for analysis
 */
function selectFile(fileId, dataType, filename, dialect) {
    // Remove previous selection
    document.querySelectorAll('.file-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Select current file
    document.getElementById(fileId).classList.add('selected');
    
    // Store selected file info
    selectedFile = {
        type: dataType,
        filename: filename,
        dialect: dialect
    };
    
    // Enable quick analyze button
    const quickAnalyzeBtn = document.getElementById('quick-analyze-btn');
    if (quickAnalyzeBtn) {
        quickAnalyzeBtn.disabled = false;
    }
}

/**
 * Play audio file
 */
function playAudio(dataType, filename, dialect, event) {
    event.stopPropagation();
    
    let audioUrl;
    if (dataType === 'raw_audio') {
        audioUrl = `/api/audio-file/${dataType}?filename=${encodeURIComponent(filename)}&dialect=${dialect}`;
    } else {
        audioUrl = `/api/audio-file/${dataType}?filename=${encodeURIComponent(filename)}`;
    }
    
    const modal = new bootstrap.Modal(document.getElementById('audioPlayerModal'));
    const audioPlayer = document.getElementById('modal-audio-player');
    
    audioPlayer.src = audioUrl;
    modal.show();
}

/**
 * Analyze uploaded file
 */
async function analyzeUploadedFile() {
    const fileInput = document.getElementById('audio-file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select an audio file to analyze.', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    await analyzeAudio(formData, 'upload');
}

/**
 * Analyze selected file
 */
async function analyzeSelectedFile() {
    if (!selectedFile) {
        showAlert('Please select an audio file from the explorer first.', 'warning');
        return;
    }
    
    const analysisData = {
        data_type: selectedFile.type,
        filename: selectedFile.filename
    };
    
    if (selectedFile.type === 'raw_audio') {
        analysisData.dialect = selectedFile.dialect;
    }
    
    await analyzeAudio(JSON.stringify(analysisData), 'existing');
}

/**
 * Analyze audio file
 */
async function analyzeAudio(data, type) {
    const resultsContainer = document.getElementById('analysis-results');
    const visualizationRow = document.getElementById('visualization-row');
    
    resultsContainer.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Analyzing audio...</div>';
    
    try {
        const options = {
            method: 'POST',
            headers: type === 'upload' ? {} : {'Content-Type': 'application/json'},
            body: data
        };
        
        const response = await fetch('/api/analyze-audio', options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Analysis failed');
        }
        
        displayAnalysisResults(result);
        visualizationRow.style.display = 'flex';
        
    } catch (error) {
        console.error('Error analyzing audio:', error);
        resultsContainer.innerHTML = `<div class="alert alert-danger">Error analyzing audio: ${error.message}</div>`;
    }
}

/**
 * Display analysis results
 */
function displayAnalysisResults(result) {
    const container = document.getElementById('analysis-results');
    const features = result.features;
    const prediction = result.dialect_prediction;
    
    let featuresHtml = '<div class="analysis-results">';
    
    // Dialect prediction
    if (prediction) {
        featuresHtml += `
            <div class="alert alert-info">
                <strong>Dialect Prediction:</strong> ${prediction.predicted_dialect} 
                (Confidence: ${(prediction.confidence * 100).toFixed(1)}%)
            </div>
        `;
    }
    
    // Audio features
    featuresHtml += '<h6>Audio Features:</h6>';
    for (const [key, value] of Object.entries(features)) {
        const displayName = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const displayValue = typeof value === 'number' ? value.toFixed(4) : value;
        
        featuresHtml += `
            <div class="feature-item">
                <span class="feature-name">${displayName}:</span>
                <span class="feature-value">${displayValue}</span>
            </div>
        `;
    }
    
    featuresHtml += '</div>';
    container.innerHTML = featuresHtml;
    
    // Display visualizations
    if (result.visualizations) {
        if (result.visualizations.waveform) {
            document.getElementById('waveform-img').src = result.visualizations.waveform;
        }
        if (result.visualizations.spectrogram) {
            document.getElementById('spectrogram-img').src = result.visualizations.spectrogram;
        }
    }
}

/**
 * Load files for comparison
 */
async function loadCompareFiles(compareNum) {
    const type = document.getElementById(`compare${compareNum}-type`).value;
    const fileSelect = document.getElementById(`compare${compareNum}-file`);
    
    // Clear existing options
    while (fileSelect.options.length > 1) {
        fileSelect.remove(1);
    }
    
    try {
        const response = await fetch(`/api/audio-files/${type}`);
        const data = await response.json();
        
        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file.filename;
                option.textContent = file.filename;
                fileSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error(`Error loading compare files for ${compareNum}:`, error);
    }
}

/**
 * Compare audio files
 */
async function compareAudio() {
    const file1Info = {
        type: document.getElementById('compare1-type').value,
        filename: document.getElementById('compare1-file').value
    };
    
    const file2Info = {
        type: document.getElementById('compare2-type').value,
        filename: document.getElementById('compare2-file').value
    };
    
    if (!file1Info.filename || !file2Info.filename) {
        showAlert('Please select both files to compare.', 'warning');
        return;
    }
    
    // Add dialect info for raw audio
    if (file1Info.type === 'raw_audio') {
        file1Info.dialect = document.getElementById('compare1-dialect').value;
        if (!file1Info.dialect) {
            showAlert('Please select dialect for file 1.', 'warning');
            return;
        }
    }
    
    if (file2Info.type === 'raw_audio') {
        file2Info.dialect = document.getElementById('compare2-dialect').value;
        if (!file2Info.dialect) {
            showAlert('Please select dialect for file 2.', 'warning');
            return;
        }
    }
    
    const comparisonData = {
        file1: file1Info,
        file2: file2Info
    };
    
    const resultsContainer = document.getElementById('comparison-results');
    const comparisonContent = document.getElementById('comparison-content');
    
    resultsContainer.style.display = 'none';
    comparisonContent.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Comparing audio files...</div>';
    
    try {
        const response = await fetch('/api/compare-audio', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(comparisonData)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Comparison failed');
        }
        
        displayComparisonResults(result);
        resultsContainer.style.display = 'block';
        
    } catch (error) {
        console.error('Error comparing audio:', error);
        comparisonContent.innerHTML = `<div class="alert alert-danger">Error comparing audio: ${error.message}</div>`;
    }
}

/**
 * Display comparison results
 */
function displayComparisonResults(result) {
    const container = document.getElementById('comparison-content');
    const similarity = result.similarity;
    const features1 = result.features1;
    const features2 = result.features2;
    
    let html = `
        <div class="comparison-result">
            <div class="similarity-score">
                <div class="similarity-value">${(similarity * 100).toFixed(1)}%</div>
                <div>Similarity Score</div>
            </div>
    `;
    
    if (result.comparison_chart) {
        html += `
            <div class="visualization-container">
                <img src="${result.comparison_chart}" alt="Comparison Chart" class="img-fluid">
            </div>
        `;
    }
    
    html += `
        <div class="feature-comparison">
            <div>
                <h6>File 1 Features:</h6>
                ${Object.entries(features1).map(([key, value]) => `
                    <div class="feature-item">
                        <span class="feature-name">${key}:</span>
                        <span class="feature-value">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                    </div>
                `).join('')}
            </div>
            <div>
                <h6>File 2 Features:</h6>
                ${Object.entries(features2).map(([key, value]) => `
                    <div class="feature-item">
                        <span class="feature-name">${key}:</span>
                        <span class="feature-value">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                    </div>
                `).join('')}
            </div>
        </div>
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the main container
    const mainContainer = document.querySelector('.container-fluid');
    mainContainer.insertBefore(alertContainer, mainContainer.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertContainer.parentNode) {
            alertContainer.remove();
        }
    }, 5000);
}