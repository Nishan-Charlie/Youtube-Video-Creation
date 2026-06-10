const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

let mainWindow;
let pythonProcess;
const BACKEND_PORT = 5847;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;

function findPython() {
  const candidates = ['python', 'python3', 'py'];
  return candidates[0];
}

function findBackendExe() {
  // When packaged by electron-builder the bundled exe lives in the resources dir
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'backend_server.exe');
  }
  return null; // dev mode: use Python script instead
}

function startPythonBackend() {
  return new Promise((resolve, reject) => {
    const backendExe = findBackendExe();

    if (backendExe) {
      // --- Packaged mode: launch bundled backend_server.exe ---
      if (!fs.existsSync(backendExe)) {
        reject(new Error(`Bundled backend not found at: ${backendExe}`));
        return;
      }
      pythonProcess = spawn(backendExe, ['--port', String(BACKEND_PORT)], {
        windowsHide: true
      });
    } else {
      // --- Dev mode: launch server.py via Python ---
      const scriptPath = path.join(__dirname, 'backend', 'server.py');
      if (!fs.existsSync(scriptPath)) {
        reject(new Error('Backend server.py not found'));
        return;
      }
      const python = findPython();
      pythonProcess = spawn(python, [scriptPath, '--port', String(BACKEND_PORT)], {
        cwd: path.join(__dirname, 'backend'),
        env: { ...process.env },
        windowsHide: true
      });
    }

    pythonProcess.stdout.on('data', (data) => {
      console.log('[Python]', data.toString().trim());
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('[Python ERR]', data.toString().trim());
    });

    pythonProcess.on('error', (err) => {
      reject(err);
    });

    // Poll until backend is ready
    let attempts = 0;
    const maxAttempts = 60;
    const interval = setInterval(() => {
      attempts++;
      http.get(`${BACKEND_URL}/health`, (res) => {
        if (res.statusCode === 200) {
          clearInterval(interval);
          resolve();
        }
      }).on('error', () => {
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          reject(new Error('Backend did not start in time'));
        }
      });
    }, 1000);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 900,
    minHeight: 600,
    frame: false,
    backgroundColor: '#0f0f0f',
    icon: path.join(__dirname, 'src', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    titleBarStyle: 'hidden'
  });

  mainWindow.loadFile(path.join(__dirname, 'src', 'index.html'));

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
}

app.whenReady().then(async () => {
  createWindow();

  let backendResult = null;

  // Start backend in parallel with window load
  const backendPromise = startPythonBackend()
    .then(() => { backendResult = { status: 'ready', url: BACKEND_URL }; })
    .catch((err) => { backendResult = { status: 'error', message: err.message }; });

  // Once page loads, send the current status (or wait for it)
  mainWindow.webContents.once('did-finish-load', async () => {
    mainWindow.webContents.send('backend-status', { status: 'starting' });
    await backendPromise;
    mainWindow.webContents.send('backend-status', backendResult);
  });

  // Also send once backend resolves in case page already loaded
  backendPromise.then(() => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('backend-status', backendResult);
    }
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) pythonProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (pythonProcess) pythonProcess.kill();
});

// IPC handlers
ipcMain.handle('get-backend-url', () => BACKEND_URL);

ipcMain.handle('save-audio', async (event, filename) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Voiceover Audio',
    defaultPath: filename,
    filters: [{ name: 'WAV Audio', extensions: ['wav'] }]
  });

  if (!result.canceled) {
    const srcPath = path.join(__dirname, 'output', filename);
    fs.copyFileSync(srcPath, result.filePath);
    return result.filePath;
  }
  return null;
});

ipcMain.handle('window-minimize', () => mainWindow.minimize());
ipcMain.handle('window-maximize', () => {
  mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize();
});
ipcMain.handle('window-close', () => {
  if (pythonProcess) pythonProcess.kill();
  app.quit();
});

ipcMain.handle('open-output-folder', () => {
  shell.openPath(path.join(__dirname, 'output'));
});

ipcMain.handle('save-video', async (event, filename) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Video',
    defaultPath: filename,
    filters: [{ name: 'MP4 Video', extensions: ['mp4'] }]
  });
  if (!result.canceled) {
    const src = path.join(__dirname, 'output', 'videos', filename);
    fs.copyFileSync(src, result.filePath);
    return result.filePath;
  }
  return null;
});

ipcMain.handle('save-thumbnail', async (event, filename) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Thumbnail',
    defaultPath: filename,
    filters: [{ name: 'PNG Image', extensions: ['png'] }]
  });
  if (!result.canceled) {
    const src = path.join(__dirname, 'output', 'thumbnails', filename);
    fs.copyFileSync(src, result.filePath);
    return result.filePath;
  }
  return null;
});

ipcMain.handle('open-video-folder', () => {
  shell.openPath(path.join(__dirname, 'output', 'videos'));
});
