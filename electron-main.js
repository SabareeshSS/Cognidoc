// main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid'); // For request IDs

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
    if (pythonProcess) {
        console.log('Python backend already running');
        return;
    }

    const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    const scriptPath = path.join(__dirname, 'python_backend/cognidoc1.0.py');
    
    // Ensure the script exists
    if (!require('fs').existsSync(scriptPath)) {
        console.error(`Python script not found at: ${scriptPath}`);
        dialog.showErrorBox('Backend Error', `Python script not found:\n${scriptPath}\nPlease ensure the backend script exists.`);
        app.quit();
        return;
    }

    console.log(`Starting Python backend: ${pythonExecutable} ${scriptPath}`);
    pythonProcess = spawn(pythonExecutable, [scriptPath], {
        // Ensure we can communicate with the process
        stdio: ['pipe', 'pipe', 'pipe']
    });

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
                    resolve(response);
                    pendingRequests.delete(requestId);
                } else if (mainWindow) {
                    // For messages without request ID or status updates
                    mainWindow.webContents.send('python-message', response);
                }
            } catch (e) {
                console.error('Error parsing Python output:', e);
                console.log('Raw output:', line);
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
    startPythonBackend(); // Call startPythonBackend here
    createWindow(); // Call createWindow here

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

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
    const command = { command: 'process', file_path: filePath, request_id: requestId };

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

// Handle 'submit-query' request from renderer
ipcMain.handle('submit-query', async (event, queryText) => {
    if (!pythonProcess) {
        throw new Error('Python backend process is not running.');
    }
    const requestId = uuidv4();
    const command = { command: 'query', query_text: queryText, request_id: requestId };
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

// Handle model listing
ipcMain.handle('get-models-by-category', async (event, category) => {
    try {
        // Read models.json directly
        const modelsPath = path.join(__dirname, 'python_backend', 'models.json');
        const modelsData = JSON.parse(require('fs').readFileSync(modelsPath, 'utf8'));
        const models = modelsData.model_categories[category] || [];
        return { type: 'models_list', models };
    } catch (error) {
        console.error('Error getting models by category:', error);
        return { type: 'error', message: error.message };
    }
});

// Handle embedding model setting
ipcMain.handle('set-embedding-model', async (event, modelName) => {
    if (!pythonProcess) {
        throw new Error('Python backend process is not running.');
    }
    const requestId = uuidv4();
    const command = { command: 'set_embedding_model', model_name: modelName, request_id: requestId };

    return new Promise((resolve, reject) => {
        pendingRequests.set(requestId, { resolve, reject });
        try {
            pythonProcess.stdin.write(JSON.stringify(command) + '\n');
            console.log('Sent set-embedding-model request to Python:', command);
        } catch (e) {
            pendingRequests.delete(requestId);
            reject(e);
        }
    });
});

// Handle set image embedding model request
ipcMain.handle('set-image-embedding-model', async (event, modelName) => {
    const requestId = uuidv4();
    try {
        const response = await sendToPython({
            command: 'set_image_embedding_model',
            model_name: modelName,
            request_id: requestId
        });
        return response;
    } catch (error) {
        console.error('Error setting image embedding model:', error);
        return {
            type: 'error',
            message: `Failed to set image embedding model: ${error.message}`,
            request_id: requestId
        };
    }
});

// Handle querying model setting
ipcMain.handle('set-querying-model', async (event, modelName) => {
    if (!pythonProcess) {
        throw new Error('Python backend process is not running.');
    }
    const requestId = uuidv4();
    const command = { command: 'set_model', model_name: modelName, request_id: requestId };

    return new Promise((resolve, reject) => {
        pendingRequests.set(requestId, { resolve, reject });
        try {
            pythonProcess.stdin.write(JSON.stringify(command) + '\n');
            console.log('Sent set-model request to Python:', command);
        } catch (e) {
            pendingRequests.delete(requestId);
            reject(e);
        }
    });
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