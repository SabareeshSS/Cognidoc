// c:\Sabareesh\Vector\deskapp\Rambo_deskapp\src\preload.js
const { contextBridge, ipcRenderer } = require('electron');

try {
    contextBridge.exposeInMainWorld('electronAPI', {
        // Ensure all functions needed by guiExample.js are listed here
        // openFileDialog: () => ipcRenderer.invoke('dialog:openFile'), // Add if needed by guiexample.js
        processFile: (filePath) => ipcRenderer.invoke('process-file', filePath),
        submitQuery: (queryText) => ipcRenderer.invoke('submit-query', queryText),
        getModels: () => ipcRenderer.invoke('get-models'),
        setEmbeddingModel: (modelName) => ipcRenderer.invoke('set-embedding-model', modelName),
        setImageEmbeddingModel: (modelName) => ipcRenderer.invoke('set-image-embedding-model', modelName),
        setQueryingModel: (modelName) => ipcRenderer.invoke('set-querying-model', modelName),
        getModelsByCategory: (category) => ipcRenderer.invoke('get-models-by-category', category),
        onPythonLog: (callback) => ipcRenderer.on('python-log', (_event, value) => callback(value)),
        onPythonError: (callback) => ipcRenderer.on('python-error', (_event, value) => callback(value)),
        onPythonStatusUpdate: (callback) => ipcRenderer.on('python-status-update', (_event, value) => callback(value)),
        onNotification: (callback) => ipcRenderer.on('show-notification', (_event, data) => callback(data)),
    });
    console.log('electronAPI exposed successfully via preload.'); // Add log
} catch (error) {
    console.error('Error exposing electronAPI via preload:', error); // Add error log
}
