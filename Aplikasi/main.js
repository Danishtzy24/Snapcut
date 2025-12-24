const { app, BrowserWindow, ipcMain, dialog, Menu, globalShortcut } = require('electron');
const path = require('path');
const fs = require('fs');

let mainWindow = null;

app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('force_high_performance_gpu');

app.commandLine.appendSwitch('disable-background-timer-throttling');
app.commandLine.appendSwitch('disable-renderer-backgrounding');
app.commandLine.appendSwitch('disable-backgrounding-occluded-windows');

app.commandLine.appendSwitch('enable-features', 'CanvasOopRasterization');

function createWindow() {
    mainWindow = new BrowserWindow({
        title: 'SnapCut',
        width: 1200,
        height: 800,
        minWidth: 900,
        minHeight: 600,
        fullscreenable: true,
        icon: path.join(__dirname, 'Logo.ico'),
        backgroundColor: '#000000',
        show: false,
        frame: true,
        titleBarStyle: 'default',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false,
            backgroundThrottling: false,
            offscreen: false,

            webSecurity: !process.env.ELECTRON_DEV,
            allowRunningInsecureContent: !!process.env.ELECTRON_DEV
        }
    });

    Menu.setApplicationMenu(null);

    mainWindow.loadFile('index.html');

    if (!app.isPackaged) {
        mainWindow.webContents.openDevTools({ mode: 'detach' });
    }

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.whenReady().then(() => {
    app.setName('SnapCut');
    createWindow();

    globalShortcut.register('F11', () => {
        if (!mainWindow) return;
        mainWindow.setFullScreen(!mainWindow.isFullScreen());
    });

    globalShortcut.register('Esc', () => {
        if (!mainWindow) return;
        if (mainWindow.isFullScreen()) {
            mainWindow.setFullScreen(false);
        }
    });

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('will-quit', () => {
    globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

ipcMain.handle('select-image', async () => {
    try {
        const { filePaths, canceled } = await dialog.showOpenDialog(mainWindow, {
            title: 'Pilih Gambar',
            properties: ['openFile'],
            filters: [
                { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp'] }
            ]
        });

        if (canceled || !filePaths?.length) {
            return { success: false, cancelled: true };
        }

        const filePath = filePaths[0];
        const stat = fs.statSync(filePath);

        return {
            success: true,
            path: filePath,
            name: path.basename(filePath),
            size: stat.size
        };

    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.handle('save-image', async (event, base64, format) => {
    try {
        const { filePath, canceled } = await dialog.showSaveDialog(mainWindow, {
            title: 'Simpan Hasil',
            defaultPath: `nobg-result.${format}`,
            filters: [{ name: 'Images', extensions: [format] }]
        });

        if (canceled || !filePath) {
            return { success: false, cancelled: true };
        }

        fs.writeFileSync(filePath, Buffer.from(base64, 'base64'));
        return { success: true, path: filePath };

    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.handle('read-file', async (event, filePath) => {
    try {
        if (!fs.existsSync(filePath)) {
            throw new Error('File tidak ditemukan');
        }

        const buffer = fs.readFileSync(filePath);
        return { success: true, buffer, size: buffer.length };

    } catch (err) {
        return { success: false, error: err.message };
    }
});
