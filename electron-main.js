// main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid'); // For request IDs
const fs = require('fs'); // <-- Add this

let mainWindow;
let pythonProcess = null;
const pendingRequests = new Map(); // Store pending request promises

function createWindow() {
    console.log(">>> Entering createWindow function..."); // <-- Add this
    mainWindow = new BrowserWindow({
        width: 1000, // Example width
        height: 800, // Example height
        webPreferences: {
            preload: path.join(__dirname, 'src/js/preload.js'),
            contextIsolation: true, // Recommended for security
            nodeIntegration: false, // Keep false
        },
    });
    console.log("Main window created."); // <-- Add log

    mainWindow.loadFile(path.join(__dirname, 'src/interface.html'));
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

        const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
        const scriptPath = path.join(__dirname, 'python_backend/cartamind1.0.py');
    
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

        // Handle Python process stdout
        pythonProcess.stdout.on('data', (data) => {
            try {
                const lines = data.toString().trim().split('\n');
                lines.forEach(line => {
                    if (!line.trim()) return;
                    
                    const response = JSON.parse(line);
                    console.log('Python response:', response);
                    
                    const requestId = response.request_id;
                    if (requestId && pendingRequests.has(requestId)) {
                        const { resolve } = pendingRequests.get(requestId);
                        pendingRequests.delete(requestId);
                        resolve(response);
                    }
                });
            } catch (e) {
                console.error('Error processing Python output:', e);
            }
        });

        // Handle Python process stderr
        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python stderr: ${data}`);
        });

        // Handle Python process errors
        pythonProcess.on('error', (err) => {
            console.error('Failed to start Python process:', err);
            dialog.showErrorBox('Backend Error', 
                `Failed to start Python backend:\n${err.message}\n` +
                'Please ensure Python is installed and in your PATH.');
        });

        // Handle Python process exit
        pythonProcess.on('close', (code) => {
            console.log(`Python backend exited with code ${code}`);
            pythonProcess = null;
        });

    } catch (error) {
        console.error('Error in startPythonBackend:', error);
        dialog.showErrorBox('Backend Error', 
            `Failed to start Python backend:\n${error.message}\n` +
            'Please ensure Python is installed and the script exists.');
        throw error;
    }

    // Handle Python process stdout
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

// Helper function to check and initialize models.json
async function checkModelsConfig() {
    const documentsPath = app.getPath('documents');
    const configDir = path.join(documentsPath, 'CartaMind');
    const modelsPath = path.join(configDir, 'models.json');

    if (!fs.existsSync(modelsPath)) {
        // Create CartaMind directory in Documents if it doesn't exist
        if (!fs.existsSync(configDir)) {
            fs.mkdirSync(configDir, { recursive: true });
        }

        // Copy default models.json if it doesn't exist in user's Documents
        const sourceModelsPath = path.join(__dirname, 'python_backend', 'models.json');
        if (fs.existsSync(sourceModelsPath)) {
            fs.copyFileSync(sourceModelsPath, modelsPath);
            if (mainWindow) {
                mainWindow.webContents.send('show-notification', {
                    title: 'Configuration Created',
                    message: `Models configuration file created at:\n${modelsPath}\nYou can customize the available models by editing this file.`
                });
            }
        }
    }
}

function sendToPython(data) {
    return new Promise((resolve, reject) => {
        if (!pythonProcess) {
            reject(new Error('Python backend not running'));
            return;
        }

        const requestId = data.request_id;
        pendingRequests.set(requestId, { resolve, reject });

        try {
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
    const command = { type: 'query', query_text: queryText, request_id: requestId };
    // Similar promise structure as 'process-file'
    return new Promise((resolve, reject) => {
        pendingRequests.set(requestId, { resolve, reject });
        try {
            pythonProcess.stdin.write(JSON.stringify(command) + '\n');
            console.log('Sent to Python:', command);
        } catch (e) {
            pendingRequests.delete(requestId);
            reject(e);
        }
         // Add a timeout?
    });
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
        const modelsPath = path.join(app.getPath('documents'), 'CartaMind', 'models.json');
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