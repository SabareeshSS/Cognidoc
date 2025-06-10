// Retrieving DOM elements
const fileInput = document.getElementById('fileInput');
const processButton = document.getElementById('processButton');
const clearButton = document.getElementById('clearButton');
const clearAllButton = document.getElementById('clearAllButton');
// const statusOutput = document.getElementById('statusOutput'); // Old status textarea
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const queryInput = document.getElementById('queryInput');
const submitButton = document.getElementById('submitButton');
const answerOutput = document.getElementById('answerOutput');
const embeddingModel = document.getElementById('embeddingModel');
const imageEmbeddingModel = document.getElementById('imageEmbeddingModel');
const queryingModel = document.getElementById('queryingModel');
const selectedFilesList = document.getElementById('selectedFilesList');

// Helper function to update the progress bar and status text
function updateStatus(text, percentage = -1, statusType = 'info') { // statusType: 'info', 'processing', 'success', 'error'
    if (progressText) {
        progressText.textContent = text;
    }
    if (progressBar) {
        if (typeof percentage === 'number' && percentage >= 0 && percentage <= 100) {
            progressBar.style.width = `${percentage}%`;
        }

        switch (statusType) {
            case 'error':
                progressBar.style.backgroundColor = '#cc3333'; // Error red
                break;
            case 'success':
                progressBar.style.backgroundColor = 'var(--accent-green-dark)'; // Success green
                break;
            case 'processing':
                progressBar.style.backgroundColor = 'var(--accent-green-medium)'; // Processing green
                break;
            case 'info':
            default:
                progressBar.style.backgroundColor = 'var(--accent-green-medium)';
                // Optional: make it less visible or match track if 0% and info
                if (progressBar.style.width === '0%') {
                    // progressBar.style.backgroundColor = 'var(--input-background)'; // Example: match track
                }
                break;
        }
    }
}

// Initialize event listeners when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    if (!fileInput || !selectedFilesList) {
        console.error('Required elements not found:', {
            fileInput: !!fileInput,
            selectedFilesList: !!selectedFilesList
        });
        return;
    }

    // File input event listener
    fileInput.addEventListener('change', handleFileSelection);
    
    console.log('Event listeners initialized');
});

// File handling functions
function handleFileSelection(event) {
    console.log('File selection change event triggered');
    const files = event.target.files;
    let filesActuallyAdded = 0;
    
    if (files && files.length > 0) {
        console.log(`${files.length} files selected from input`);
        const newFilesInput = Array.from(files);
        
        newFilesInput.forEach(newFile => {
            // Add to selected files, preventing duplicates by name
            if (!currentFiles.some(existingFile => existingFile.name === newFile.name)) {
                currentFiles.push(newFile);
                filesActuallyAdded++;
            } else {
                console.log(`File ${newFile.name} already in selected list.`);
            }
        });
        
        if (filesActuallyAdded > 0) {
            updateFileList(currentFiles, selectedFilesList, false);
            updateStatus(`${filesActuallyAdded} file(s) added to selection. Add more, or choose models and Process.`, 0, 'info');
            processButton.disabled = false;
        } else if (newFilesInput.length > 0) { // Files were selected, but all were duplicates
            updateStatus(`Selected file(s) are already in the list. Add more, or choose models and Process.`, 0, 'info');
        }
        
        // Clear input to allow re-selecting the same file if user removes it and wants to add again
        fileInput.value = ''; 
        
        console.log('Current selected files:', currentFiles.map(f => f.name));
    } else {
        console.log('No files selected in this input event');
    }
    updateButtonStates(); // Update button states based on currentFiles
}

// State management
let currentFiles = [];
let isProcessing = false;
let isReadyForQuery = false;
let currentEmbeddingModel = '';
let currentImageEmbeddingModel = '';
let currentQueryingModel = '';

// Initialize model selectors
async function initializeModels() {
    console.log('Starting model initialization...');
    
    try {
        const modelCategories = {
            'text_embedding_models': embeddingModel,
            'image_embedding_models': imageEmbeddingModel,
            'querying_models': queryingModel
        };

        for (const [category, element] of Object.entries(modelCategories)) {
            console.log(`Fetching models for category: ${category}`);
            const result = await window.electronAPI.getModelsByCategory(category);
            console.log(`Result for ${category}:`, result);

            if (result && result.type === 'models_list' && Array.isArray(result.models)) {
                // Clear existing options
                element.innerHTML = '';
                
                // Add model options
                result.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    element.appendChild(option);
                });

                // Set default selection
                if (result.models.length > 0) {
                    element.value = result.models[0];
                    switch(element) {
                        case embeddingModel:
                            currentEmbeddingModel = result.models[0];
                            break;
                        case imageEmbeddingModel:
                            currentImageEmbeddingModel = result.models[0];
                            break;
                        case queryingModel:
                            currentQueryingModel = result.models[0];
                            break;
                    }
                }
            } else {
                console.error(`Invalid response for ${category}:`, result);
                updateStatus(`Error loading ${category} models. Invalid response.`, 0, 'error');
            }
        }
    } catch (error) {
        console.error('Error initializing models:', error);
        updateStatus(`Error initializing models: ${error.message}`, 0, 'error');
    }
}

// Model selection change handlers
embeddingModel.addEventListener('change', async () => {
    const selectedModel = embeddingModel.value;
    if (selectedModel && selectedModel !== currentEmbeddingModel) {
        try {
            embeddingModel.disabled = true;
            updateStatus(`Setting embedding model to ${selectedModel}...`, 0, 'info');
            
            const result = await window.electronAPI.setEmbeddingModel(selectedModel);
            console.log('Set embedding model result:', result);
            
            if (result && result.type === 'model_set') {
                currentEmbeddingModel = selectedModel;
                updateStatus(result.message, 0, 'success');
            } else {
                throw new Error(result.message || 'Failed to set embedding model');
            }
        } catch (error) {
            console.error('Error setting embedding model:', error);
            updateStatus(`Error setting embedding model: ${error.message}`, 0, 'error');
            embeddingModel.value = currentEmbeddingModel;
        } finally {
            embeddingModel.disabled = false;
        }
    }
});

imageEmbeddingModel.addEventListener('change', async () => {
    const selectedModel = imageEmbeddingModel.value;
    if (selectedModel && selectedModel !== currentImageEmbeddingModel) {
        try {
            imageEmbeddingModel.disabled = true;
            updateStatus(`Setting image embedding model to ${selectedModel}...`, 0, 'info');
            
            const result = await window.electronAPI.setImageEmbeddingModel(selectedModel);
            console.log('Set image embedding model result:', result);
            
            if (result && result.type === 'model_set') {
                currentImageEmbeddingModel = selectedModel;
                updateStatus(result.message, 0, 'success');
            } else {
                throw new Error(result.message || 'Failed to set image embedding model');
            }
        } catch (error) {
            console.error('Error setting image embedding model:', error);
            updateStatus(`Error setting image embedding model: ${error.message}`, 0, 'error');
            imageEmbeddingModel.value = currentImageEmbeddingModel;
        } finally {
            imageEmbeddingModel.disabled = false;
        }
    }
});

queryingModel.addEventListener('change', async () => {
    const selectedModel = queryingModel.value;
    if (selectedModel && selectedModel !== currentQueryingModel) {
        try {
            queryingModel.disabled = true;
            updateStatus(`Setting querying model to ${selectedModel}...`, 0, 'info');
            
            const result = await window.electronAPI.setQueryingModel(selectedModel);
            console.log('Set querying model result:', result);
            
            if (result && result.type === 'model_set') {
                currentQueryingModel = selectedModel;
                updateStatus(result.message, 0, 'success');
            } else {
                throw new Error(result.message || 'Failed to set querying model');
            }
        } catch (error) {
            console.error('Error setting querying model:', error);
            updateStatus(`Error setting querying model: ${error.message}`, 0, 'error');
            queryingModel.value = currentQueryingModel;
        } finally {
            queryingModel.disabled = false;
        }
    }
});

// File handling
let processedFiles = [];

function updateFileListItem(filename, status, details = '') {
    const fileItem = document.querySelector(`#selectedFilesList li[data-filename="${filename}"]`);
    if (fileItem) {
        const statusSpan = fileItem.querySelector('.file-status');
        if (statusSpan) {
            statusSpan.textContent = status;
            switch (status) {
                case 'Error':
                    statusSpan.style.backgroundColor = '#ff4444';
                    fileItem.style.borderLeftColor = '#ff4444';
                    break;
                case 'Processing':
                    statusSpan.style.backgroundColor = '#ffa500';
                    fileItem.style.borderLeftColor = '#ffa500';
                    break;
                default:
                    statusSpan.style.backgroundColor = 'var(--accent-green-dark)';
                    fileItem.style.borderLeftColor = 'var(--accent-green-dark)';
            }
        }
        fileItem.title = details || status;
    }
}

// File handling functions
function updateFileList(files, listElement) {
    if (!listElement) {
        console.error('Invalid list element');
        return;
    }

    try {
        // Clear current list
        while (listElement.firstChild) {
            listElement.removeChild(listElement.firstChild);
        }

        // Convert to array if needed
        const fileArray = Array.isArray(files) ? files : Array.from(files || []);
        
        console.log(`Updating selected files list with ${fileArray.length} files`);
        
        fileArray.forEach(file => {
            const li = document.createElement('li');
            li.className = 'file-item';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'file-name';
            nameSpan.textContent = file.name;
            li.appendChild(nameSpan);
            
            const removeBtn = document.createElement('button');
            removeBtn.textContent = '×';
            removeBtn.className = 'remove-file';
            removeBtn.onclick = (e) => {
                e.stopPropagation();
                currentFiles = currentFiles.filter(f => f.name !== file.name);
                updateFileList(currentFiles, selectedFilesList);
                processButton.disabled = currentFiles.length === 0;
                
                if (currentFiles.length === 0) {
                    updateStatus('All files removed. Please select files to process.', 0, 'info');
                }
            };
            li.appendChild(removeBtn);
            
            listElement.appendChild(li);
        });

        console.log(`Updated file list with ${fileArray.length} files`);
    } catch (error) {
        console.error('Error updating file list:', error);
    }
}

function updateButtonStates() {
    // Enable or disable process button based on file selection and processing state
    processButton.disabled = currentFiles.length === 0 || isProcessing;
    
    // Update query-related buttons based on processing state
    queryInput.disabled = !isReadyForQuery;
    submitButton.disabled = !isReadyForQuery;
    clearButton.disabled = !isReadyForQuery;
    
    // Update status message if needed
    if (currentFiles.length > 0 && !isProcessing && !isReadyForQuery) {
        updateStatus('Files selected. Choose models if needed, then click Process Documents.', 0, 'info');
    }
}

// File input event listener
// Process button event listener
processButton.addEventListener('click', async () => {
    if (currentFiles.length === 0) {
        updateStatus('No files selected for processing.', 0, 'error');
        return;
    }

    try {
        isProcessing = true;
        updateButtonStates();
        updateStatus('Processing files...', 50, 'processing');

        // Convert File objects to paths
        const filePaths = currentFiles.map(file => file.path);
        
        // Send to backend for processing
        const result = await window.electronAPI.processFiles(filePaths);
        console.log('Process files result:', result);

        if (result.success) {
            updateStatus(result.message, 100, 'success');
            isReadyForQuery = true;
        } else {
            throw new Error(result.message || 'Failed to process files');
        }
    } catch (error) {
        console.error('Error processing files:', error);
        updateStatus(`Error processing files: ${error.message}`, 100, 'error'); // Show 100% bar but in error color
        isReadyForQuery = false;
    } finally {
        isProcessing = false;
        updateButtonStates();
    }
});

// Clear button handlers
clearButton.addEventListener('click', () => {
    queryInput.value = '';
    answerOutput.value = '';
    clearButton.disabled = true;
});

clearAllButton.addEventListener('click', () => {
    // Clear all files and reset state
    currentFiles = [];
    fileInput.value = '';
    queryInput.value = '';
    answerOutput.value = '';    
    updateStatus("Ready. Upload files and click 'Process'.", 0, 'info');
    isReadyForQuery = false;
    updateFileList([], selectedFilesList, false);
    updateButtonStates();
});

// Reset query state
function resetQueryState() {
    isReadyForQuery = false;
    queryInput.disabled = true;
    submitButton.disabled = true;
    queryInput.value = '';
    queryInput.placeholder = 'Upload & Process files first...';
}

// Handle query submission
async function handleQuerySubmit() {
    const query = queryInput.value.trim();
    if (!query) {
        answerOutput.value = 'Please enter a question or request a summary.';
        return;
    }
    if (!isReadyForQuery) {
        answerOutput.value = 'Error: Please process files successfully first.';
        return;
    }

    answerOutput.value = 'Searching through documents...';
    submitButton.disabled = true;
    queryInput.disabled = true;
    updateStatus('Processing query...', 50, 'processing');

    try {
        const result = await window.electronAPI.submitQuery(query);
        console.log("Query Result from Main:", result);

        if (!result || result.type !== 'query_result') {
            throw new Error(result?.message || 'Invalid response from query');
        }

        if (!result.success) {
            throw new Error(result.message || 'Query processing failed');
        }

        // Always update answer output with the answer
        answerOutput.value = result.answer || 'No answer received';

        // If we have sources and chunks info, append them
        if (result.sources && result.chunks_used) {
            answerOutput.value += `\n\nSources: ${result.sources.join(', ')}\nBased on ${result.chunks_used} relevant chunks`;
        }

        // Update status
        updateStatus(result.message || 'Query processed successfully', 100, 'success');
    } catch (error) {
        console.error('Error processing query:', error);
        answerOutput.value = `Error: ${error.message}`;
        updateStatus(`Error: ${error.message}`, 100, 'error'); // Show 100% bar but in error color
    } finally {
        submitButton.disabled = false;
        queryInput.disabled = false;
    }
}

// Submit button click event
submitButton.addEventListener('click', handleQuerySubmit);

// Query input keypress event
queryInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleQuerySubmit();
    }
});

// Notification handling
window.electronAPI.onNotification(({ title, message }) => {
    // Update text, keep current progress bar percentage and color by not specifying type or percentage
    updateStatus(`${title}: ${message}`, -1); 
});

// Initialize
resetQueryState();
clearButton.disabled = true;
initializeModels();
updateStatus("Ready. Upload files and click 'Process'.", 0, 'info'); // Initial status