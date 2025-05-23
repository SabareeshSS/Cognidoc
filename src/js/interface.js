// Retrieving DOM elements
const fileInput = document.getElementById('fileInput');
const processButton = document.getElementById('processButton');
const clearButton = document.getElementById('clearButton');
const statusOutput = document.getElementById('statusOutput');
const queryInput = document.getElementById('queryInput');
const submitButton = document.getElementById('submitButton');
const answerOutput = document.getElementById('answerOutput');
const embeddingModel = document.getElementById('embeddingModel');
const imageEmbeddingModel = document.getElementById('imageEmbeddingModel');
const queryingModel = document.getElementById('queryingModel');

let currentFiles = [];
let isProcessing = false;
let isReadyForQuery = false;
let currentEmbeddingModel = '';
let currentImageEmbeddingModel = '';
let currentQueryingModel = '';

// Initialize model selectors
async function initializeModels() {
    try {
        console.log('Initializing model selectors...');
        
        // Clear existing options
        embeddingModel.innerHTML = '';
        imageEmbeddingModel.innerHTML = '';
        queryingModel.innerHTML = '';

        // Define model categories and their corresponding select elements
        const modelSelectors = [
            { category: 'text_embedding_models', select: embeddingModel, currentVar: 'currentEmbeddingModel' },
            { category: 'image_embedding_models', select: imageEmbeddingModel, currentVar: 'currentImageEmbeddingModel' },
            { category: 'querying_models', select: queryingModel, currentVar: 'currentQueryingModel' }
        ];

        // Populate each model selector
        for (const { category, select, currentVar } of modelSelectors) {
            try {
                console.log(`Fetching models for category: ${category}`);
                const result = await window.electronAPI.getModelsByCategory(category);
                
                if (result && result.type === 'models_list' && Array.isArray(result.models)) {
                    console.log(`Got models for ${category}:`, result.models);
                    
                    // Add options to select element
                    result.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        select.appendChild(option);
                    });

                    // Set initial selection if we have models
                    if (result.models.length > 0) {
                        const defaultModel = result.models[0];
                        select.value = defaultModel;
                        window[currentVar] = defaultModel;
                    }
                } else {
                    console.error(`Failed to get models for ${category}:`, result);
                    statusOutput.value = `Error loading ${category}. Check the backend connection.`;
                }
            } catch (error) {
                console.error(`Error loading ${category}:`, error);
                statusOutput.value = `Error loading ${category}. ${error.message}`;
            }
        }
    } catch (error) {
        console.error('Error initializing models:', error);
        statusOutput.value = 'Error initializing models. Check connection to backend.';
    }
}

// Call initialization on page load
initializeModels();

// Handle embedding model selection change
embeddingModel.addEventListener('change', async () => {
    const selectedModel = embeddingModel.value;
    if (selectedModel && selectedModel !== currentEmbeddingModel) {
        try {
            embeddingModel.disabled = true;
            statusOutput.value = `Setting embedding model to ${selectedModel}...`;
            
            const result = await window.electronAPI.setEmbeddingModel(selectedModel);
            console.log('Set embedding model result:', result);
            
            if (result && result.type === 'model_set') {
                currentEmbeddingModel = selectedModel;
                statusOutput.value = result.message;
            } else {
                throw new Error(result.message || 'Failed to set embedding model');
            }
        } catch (error) {
            console.error('Error setting embedding model:', error);
            statusOutput.value = `Error setting embedding model: ${error.message}`;
            embeddingModel.value = currentEmbeddingModel;
        } finally {
            embeddingModel.disabled = false;
        }
    }
});

// Handle image embedding model selection change
imageEmbeddingModel.addEventListener('change', async () => {
    const selectedModel = imageEmbeddingModel.value;
    if (selectedModel && selectedModel !== currentImageEmbeddingModel) {
        try {
            imageEmbeddingModel.disabled = true;
            statusOutput.value = `Setting image embedding model to ${selectedModel}...`;
            
            const result = await window.electronAPI.setImageEmbeddingModel(selectedModel);
            console.log('Set image embedding model result:', result);
            
            if (result && result.type === 'model_set') {
                currentImageEmbeddingModel = selectedModel;
                statusOutput.value = result.message;
            } else {
                throw new Error(result.message || 'Failed to set image embedding model');
            }
        } catch (error) {
            console.error('Error setting image embedding model:', error);
            statusOutput.value = `Error setting image embedding model: ${error.message}`;
            imageEmbeddingModel.value = currentImageEmbeddingModel;
        } finally {
            imageEmbeddingModel.disabled = false;
        }
    }
});

// Handle querying model selection change
queryingModel.addEventListener('change', async () => {
    const selectedModel = queryingModel.value;
    if (selectedModel && selectedModel !== currentQueryingModel) {
        try {
            queryingModel.disabled = true;
            statusOutput.value = `Setting querying model to ${selectedModel}...`;
            
            const result = await window.electronAPI.setQueryingModel(selectedModel);
            console.log('Set querying model result:', result);
            
            if (result && result.type === 'model_set') {
                currentQueryingModel = selectedModel;
                statusOutput.value = result.message;
            } else {
                throw new Error(result.message || 'Failed to set querying model');
            }
        } catch (error) {
            console.error('Error setting querying model:', error);
            statusOutput.value = `Error setting querying model: ${error.message}`;
            queryingModel.value = currentQueryingModel;
        } finally {
            queryingModel.disabled = false;
        }
    }
});

let processedFiles = [];

// File input change event
fileInput.addEventListener('change', (event) => {
    currentFiles = Array.from(event.target.files);
    if (currentFiles.length > 0) {
        resetQueryState();
        answerOutput.value = '';
        clearButton.disabled = true;
        const fileNames = currentFiles.map(f => f.name).join(', ');
        statusOutput.value = `Selected files: ${fileNames}\nClick 'Process' to begin.`;
    } else {
        statusOutput.value = "File selection cancelled.";
    }
});

// Process button click event
processButton.addEventListener('click', async () => {
    if (currentFiles.length === 0) {
        statusOutput.value = 'Error: No files selected.';
        return;
    }
    if (isProcessing) {
        statusOutput.value = 'Error: Already processing files.';
        return;
    }

    isProcessing = true;
    resetQueryState();
    processButton.disabled = true;
    clearButton.disabled = true;
    processedFiles = []; // Reset processed files list

    try {
        for (const file of currentFiles) {
            statusOutput.value = `Processing '${file.name}'...`;
            const result = await window.electronAPI.processFile(file.path);
            console.log("Process Result from Main:", result);
            
            if (result.type === 'process_complete') {
                processedFiles.push(file.name);
                statusOutput.value = result.status;
            } else {
                statusOutput.value += `\nError processing ${file.name}: ${result.message || 'Unknown error'}`;
            }
        }

        if (processedFiles.length > 0) {
            isReadyForQuery = true;
            queryInput.disabled = false;
            submitButton.disabled = false;
            clearButton.disabled = false;
            queryInput.placeholder = `Ask about the processed files or type 'summarize'...`;
            statusOutput.value += `\n\nReady to answer questions about: ${processedFiles.join(', ')}`;
        }
    } catch (error) {
        console.error('Error processing files:', error);
        statusOutput.value = `Error processing files: ${error.message}`;
    } finally {
        isProcessing = false;
        processButton.disabled = false;
    }
});

// Clear button click event
clearButton.addEventListener('click', async () => {
    try {
        statusOutput.value = "Clearing UI state...";
        await new Promise(resolve => setTimeout(resolve, 100));
        statusOutput.value = "UI Cleared. Upload new files.";
        resetQueryState();
        answerOutput.value = '';
        fileInput.value = '';
        currentFiles = [];
        processedFiles = [];
        clearButton.disabled = true;
    } catch (error) {
        console.error('Error clearing state:', error);
        statusOutput.value = `Error: Could not clear state.\nDetails: ${error.message}`;
    }
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

    answerOutput.value = 'Thinking...';
    submitButton.disabled = true;
    queryInput.disabled = true;

    try {
        const result = await window.electronAPI.submitQuery(query);
        console.log("Query Result from Main:", result);

        if (!result || result.type !== 'query_result') {
            throw new Error(result.message || 'Invalid response received from backend process.');
        }

        answerOutput.value = result.answer;
    } catch (error) {
        console.error('Error submitting query:', error);
        answerOutput.value = `Error: Could not connect to backend or query failed.\nDetails: ${error.message}`;
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

// Initialize
resetQueryState();
clearButton.disabled = true;
initializeModels();