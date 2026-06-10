const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),
  saveAudio: (filename) => ipcRenderer.invoke('save-audio', filename),
  minimizeWindow: () => ipcRenderer.invoke('window-minimize'),
  maximizeWindow: () => ipcRenderer.invoke('window-maximize'),
  closeWindow: () => ipcRenderer.invoke('window-close'),
  openOutputFolder: () => ipcRenderer.invoke('open-output-folder'),
  saveVideo: (filename) => ipcRenderer.invoke('save-video', filename),
  saveThumbnail: (filename) => ipcRenderer.invoke('save-thumbnail', filename),
  openVideoFolder: () => ipcRenderer.invoke('open-video-folder'),
  onBackendStatus: (callback) => ipcRenderer.on('backend-status', (event, data) => callback(data))
});
