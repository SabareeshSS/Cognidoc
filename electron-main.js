// main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs'); // For synchronous file operations like existsSync, copyFileSync
const fsPromises = require('fs').promises; // For asynchronous file operations
const os = require('os'); // For temporary directory

let mainWindow;
let pythonProcess = null;
const pendingRequests = new Map(); // Store pending request promises

function createWindow() {
    console.log(">>> Entering createWindow function..."); // <-- Add this
    mainWindow = new BrowserWindow({
        width: 1000, // Example width
        height: 800, // Example height
        icon: path.join(getResourcePath('src/assets'), 'inquiroai.png'), // or .png
        webPreferences: {
            preload: path.join(app.isPackaged ? __dirname : '.', 'src/js/preload.js'),
            contextIsolation: true, // Recommended for security
            nodeIntegration: false, // Keep false
        },
    });
    mainWindow.setMenu(null); // Remove the default menu bar
    console.log("Main window created."); // <-- Add log

    mainWindow.loadFile(path.join(app.isPackaged ? __dirname : '.', 'src/interface.html'));
    console.log("HTML file loaded."); // <-- Add log

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
} // <-- End of createWindow function

function startPythonBackend() {
    try {
        if (pythonProcess) {
            console.log('Python backend already running');
            return;
        }
        // ORIGINAL to use the default python executable on the system
        const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
        // Use the specified Python executable path
        // const pythonExecutable = 'D:/Python/env3.12/Scripts/python.exe';
        const scriptPath = path.join(getResourcePath('python_backend'), 'inquiroAI.py');
    
        // Ensure the script exists
        if (!fs.existsSync(scriptPath)) {
            const error = `Python script not found at: ${scriptPath}`;
            console.error(error);
            dialog.showErrorBox('Backend Error', `Python script not found:\n${scriptPath}\nPlease ensure the backend script exists.`);
            app.quit();
            throw new Error(error);
        }

        console.log(`Starting Python backend: ${pythonExecutable} ${scriptPath}`);
        pythonProcess = spawn(pythonExecutable, [scriptPath], {
            stdio: ['pipe', 'pipe', 'pipe']
        });

    // Consolidated Python process event handlers
    pythonProcess.stdout.on('data', (data) => {
        const lines = data.toString().trim().split('\n');
        lines.forEach(line => {
            if (!line.trim()) return;
            
            try {
                const response = JSON.parse(line);
                console.log('Python response:', response);
                
                const requestId = response.request_id;
                if (requestId && pendingRequests.has(requestId)) {
                    const { resolve } = pendingRequests.get(requestId);
                    console.log('Resolving pending request:', requestId, response);
                    resolve(response);
                    pendingRequests.delete(requestId);
                } else if (response.type === 'model_set') {
                    // Handle model change confirmation without request ID
                    console.log('Broadcasting model change:', response);
                    if (mainWindow) {
                        mainWindow.webContents.send('model-changed', response);
                    }
                } else if (mainWindow) {
                    // For other messages without request ID or status updates
                    console.log('Broadcasting message:', response);
                    mainWindow.webContents.send('python-message', response);
                }
            } catch (e) {
                console.error('Error parsing Python output:', e);
                console.log('Raw output:', line);
                // Broadcast parse error to renderer
                if (mainWindow) {
                    mainWindow.webContents.send('python-error', `Failed to parse Python output: ${e.message}`);
                }
            }
        });
    });

    // Handle Python process stderr
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', data.toString());
        }
    });

    // Handle Python process errors
    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', `Failed to start Python backend: ${err.message}`);
        }
        // Clean up any pending requests
        pendingRequests.forEach(({ reject }) => {
            reject(new Error('Python backend failed to start'));
        });
        pendingRequests.clear();
        dialog.showErrorBox('Backend Error', 
            `Failed to start Python backend:\n${err.message}\n` +
            'Please ensure Python is installed and in your PATH.');
    });

    // Handle Python process exit
    pythonProcess.on('close', (code) => {
        console.log(`Python backend exited with code ${code}`);
        
        // Reject all pending requests
        pendingRequests.forEach(({ reject }) => {
            reject(new Error(`Python backend process exited unexpectedly (code: ${code})`));
        });
        pendingRequests.clear();
        
        if (code !== 0 && mainWindow) {
            dialog.showErrorBox('Backend Error', 
                'The Python backend process has stopped unexpectedly.\n' +
                'Please check that Python is installed correctly and try restarting the application.');
        }
        
        pythonProcess = null;
    });

    // Debug logging for resource paths
    console.log('App is packaged:', app.isPackaged);
    console.log('__dirname:', __dirname);
    console.log('process.resourcesPath:', process.resourcesPath);

    } catch (error) { // Catch errors from the spawn process itself or initial setup
        console.error('Error in startPythonBackend:', error);
        dialog.showErrorBox('Backend Error', 
            `Failed to start Python backend:\n${error.message}\n` +
            'Please ensure Python is installed and the script exists.');
        throw error; // Re-throw if you want app startup to halt or be handled further up
    }
} // <-- End of startPythonBackend function

app.whenReady().then(() => {
    console.log(">>> App is ready. Calling startPythonBackend and createWindow..."); // <-- Add this
    checkModelsConfig();  // Add this line
    startPythonBackend(); // Call startPythonBackend here
    createWindow(); // Call createWindow here

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// Helper function to get correct path for resources
function getResourcePath(relativePath) {
    // In development, use path relative to __dirname
    if (app.isPackaged) {
        // In production, use path relative to process.resourcesPath
        return path.join(process.resourcesPath, relativePath);
    }
    return path.join(__dirname, relativePath);
}

// Helper function to check and ensure models.json directory exists
async function checkModelsConfig() {
    const documentsPath = app.getPath('documents');
    const configDir = path.join(documentsPath, 'InquiroAI');
    const modelsPath = path.join(configDir, 'models.json');

    // Create InquiroAI directory in Documents if it doesn't exist
    if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
    }

    // If models.json doesn't exist, show a notification to the user
    if (!fs.existsSync(modelsPath)) {
        if (mainWindow) {
            mainWindow.webContents.send('show-notification', {
                title: 'Configuration Required',
                message: `Please create a models.json file at:\n${modelsPath}\nThis file is required for configuring the available models.`
            });
        }
        console.log(`Models configuration file needed at: ${modelsPath}`);
    }
}

function sendToPython(data) {
    return new Promise((resolve, reject) => {
        if (!pythonProcess) {
            reject(new Error('Python backend not running'));
            return;
        }

        // Ensure we have a request ID
        const requestId = data.request_id || uuidv4();
        data.request_id = requestId;
        
        pendingRequests.set(requestId, { resolve, reject });

        try {
            console.log('Sending to Python:', data);
            pythonProcess.stdin.write(JSON.stringify(data) + '\n');
        } catch (error) {
            pendingRequests.delete(requestId);
            reject(error);
        }
    });
}

// IPC Handlers
ipcMain.handle('process-file', async (event, filePath) => {
    if (!pythonProcess) {
        throw new Error('Python backend process is not running.');
    }
    const requestId = uuidv4();
    const command = { type: 'process', file_path: filePath, request_id: requestId };

    return new Promise((resolve, reject) => {
        pendingRequests.set(requestId, { resolve, reject });
        try {
             pythonProcess.stdin.write(JSON.stringify(command) + '\n');
             console.log('Sent to Python:', command);
        } catch (e) {
             pendingRequests.delete(requestId);
             reject(e);
        }
    });
});

// Handle 'submit-query' request from renderer
ipcMain.handle('submit-query', async (event, queryText) => {
    if (!pythonProcess) {
        throw new Error('Python backend process is not running.');
    }
    
    const requestId = uuidv4();
    try {
        const response = await sendToPython({
            type: 'query',
            query_text: queryText,
            request_id: requestId
        });
        
        console.log('Query response:', response);
        
        // Only broadcast valid query responses
        if (response.type === 'query_result' || response.type === 'query_response') {
            if (mainWindow) {
                mainWindow.webContents.send('query-response', response);
            }
        }
        
        return response;
    } catch (e) {
        console.error('Error processing query:', e);
        return {
            type: 'query_result',
            success: false,
            message: e.message,
            answer: null,
            request_id: requestId
        };
    }
});

// Handle model listing
ipcMain.handle('get-models', async () => {
    const requestId = uuidv4();
    try {
        const response = await sendToPython({
            type: 'get_models',
            request_id: requestId
        });
        return response;
    } catch (error) {
        console.error('Error getting models:', error);
        return { type: 'error', message: error.message };
    }
});

// Handle model-related IPC events
ipcMain.handle('get-models-by-category', async (event, category) => {
    console.log(`Handling get-models-by-category request for: ${category}`);
    try {
        const modelsPath = path.join(app.getPath('documents'), 'InquiroAI', 'models.json');
        console.log(`Reading models from: ${modelsPath}`);
        
        if (!fs.existsSync(modelsPath)) {
            console.error('models.json not found');
            return { type: 'error', message: 'models.json not found' };
        }

        const modelsData = JSON.parse(fs.readFileSync(modelsPath, 'utf8'));
        const categoryModels = modelsData?.model_categories?.[category] || [];
        
        console.log(`Found ${categoryModels.length} models for category ${category}`);
        return { 
            type: 'models_list', 
            models: categoryModels,
            category: category
        };
    } catch (error) {
        console.error('Error getting models by category:', error);
        return { 
            type: 'error', 
            message: `Failed to get models for ${category}: ${error.message}` 
        };
    }
});

// Handle model selection changes
ipcMain.handle('set-embedding-model', async (event, modelName) => {
    return handleModelChange('embedding', modelName);
});

ipcMain.handle('set-image-embedding-model', async (event, modelName) => {
    return handleModelChange('image_embedding', modelName);
});

ipcMain.handle('set-querying-model', async (event, modelName) => {
    return handleModelChange('querying', modelName);
});

// Helper function to handle model changes
async function handleModelChange(type, modelName) {
    const requestId = uuidv4();
    console.log(`Handling model change: type=${type}, model=${modelName}, requestId=${requestId}`);
    
    try {
        const command = {
            type: `set_${type}_model`,
            model_name: modelName,
            request_id: requestId
        };
        console.log('Sending command to Python:', command);
        
        const result = await sendToPython(command);
        console.log('Received response from Python:', result);
        
        if (!result) {
            throw new Error('No response received from Python backend');
        }
        
        return result;
    } catch (error) {
        console.error(`Error setting ${type} model:`, error);
        return { 
            type: 'error', 
            message: `Failed to set ${type} model: ${error.message}`,
            request_id: requestId
        };
    }
}

// IPC handlers
ipcMain.handle('process-files', async (event, filePaths) => {
    console.log('Processing files:', filePaths);
    try {
        const command = {
            type: 'process_files',
            file_paths: filePaths
        };

        const response = await sendToPython(command);
        console.log('Process files response:', response);
        return response;
    } catch (error) {
        console.error('Error processing files:', error);
        throw error;
    }
});

// Handle 'transcribe-audio' request from renderer
ipcMain.handle('transcribe-audio', async (event, audioArrayBuffer) => {
    if (!pythonProcess) {
        return { type: 'error', message: 'Python backend process is not running.' };
    }

    const audioBuffer = Buffer.from(audioArrayBuffer);
    let tempFilePath = '';
    const requestId = uuidv4();

    try {
        const tempDir = path.join(app.getPath('userData'), 'temp_audio');
        await fsPromises.mkdir(tempDir, { recursive: true });
        // Using .webm as that's what MediaRecorder often defaults to.
        // Python's whisper should handle .webm if ffmpeg is installed.
        tempFilePath = path.join(tempDir, `recording-${Date.now()}.webm`);

        await fsPromises.writeFile(tempFilePath, audioBuffer);
        console.log('Audio saved to temporary file:', tempFilePath);

        const commandToPython = {
            type: 'transcribe_audio', // Matches the command in 
            audio_path: tempFilePath,
            request_id: requestId
        };

        const response = await sendToPython(commandToPython);
        console.log('Transcription response from Python:', response);
        return response; // Python backend should return { type: 'transcription_result', text: '...' } or error

    } catch (error) {
        console.error('Error handling transcribe-audio in main.js:', error);
        return { 
            type: 'error', 
            message: `Main process error during transcription: ${error.message}`,
            request_id: requestId 
        };
    } finally {
        if (tempFilePath) {
            try {
                await fsPromises.unlink(tempFilePath);
                console.log('Temporary audio file deleted:', tempFilePath);
            } catch (delError) {
                console.error('Error deleting temporary audio file:', delError);
            }
        }
    }
});

// Handle 'start-recording' request from renderer
ipcMain.handle('start-recording', async () => {
    const requestId = uuidv4();
    try {
        const response = await sendToPython({
            type: 'start_recording',
            request_id: requestId
        });
        return response;
    } catch (error) {
        console.error('Error starting recording:', error);
        return { 
            type: 'error', 
            message: `Failed to start recording: ${error.message}`,
            request_id: requestId
        };
    }
});

// Handle 'stop-recording' request from renderer
ipcMain.handle('stop-recording', async () => {
    const requestId = uuidv4();
    try {
        const response = await sendToPython({
            type: 'stop_recording',
            request_id: requestId
        });
        return response;
    } catch (error) {
        console.error('Error stopping recording:', error);
        return { 
            type: 'error', 
            message: `Failed to stop recording: ${error.message}`,
            request_id: requestId
        };
    }
});

// Handle 'process-voice-query' request from renderer
ipcMain.handle('process-voice-query', async (event, transcription) => {
    const requestId = uuidv4();
    try {
        const response = await sendToPython({
            type: 'voice_query',
            transcription: transcription,
            request_id: requestId
        });
        return response;
    } catch (error) {
        console.error('Error processing voice query:', error);
        return {
            type: 'error',
            message: `Failed to process voice query: ${error.message}`,
            request_id: requestId
        };
    }
});

app.on('window-all-closed', () => {
    console.log("All windows closed."); // <-- Add log
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    console.log("App will quit. Terminating Python process if running..."); // <-- Add log
    if (pythonProcess) {
        console.log('Terminating Python backend process...');
        pythonProcess.kill();
    }
});