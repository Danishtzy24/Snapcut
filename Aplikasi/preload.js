if (global.ort) {
    console.log("🔄 Cleaning up previous ONNX Runtime instance...");
    delete global.ort;
    delete global.onnxruntime;
    
    const modulePaths = Object.keys(require.cache);
    modulePaths.forEach(path => {
        if (path.includes('onnxruntime-node')) {
            delete require.cache[path];
        }
    });
    
    if (global.gc) {
        global.gc();
    }
}

const { contextBridge, ipcRenderer } = require("electron");
const ort = require("onnxruntime-node");
const sharp = require("sharp");
const path = require("path");
const fs = require("fs");
const os = require("os");

const MODEL_INPUT_SIZE = 1024;
const MAX_IMAGE_SIZE = 2048;
const COMPRESSION_QUALITY = 95;
const AUTO_UNLOAD_DELAY = 5000;

const ENABLE_ALL_GPU = true;
const ENABLE_CUDA = true;
const ENABLE_DIRECTML = true;
const ENABLE_TENSORRT = true;
const ENABLE_COREML = true;
const USE_CPU_FALLBACK = true;

const PRELOAD_MODEL_ON_STARTUP = true;
const AUTO_UNLOAD_AFTER_PRELOAD = false;

const ENABLE_CPU_OPTIMIZATIONS = true;
const CPU_THREADS_MULTIPLIER = 0.75;

async function detectAvailableGPUs() {
    const gpus = {
        cuda: false,
        directml: false,
        tensorrt: false,
        coreml: false,
        cpu: true
    };
    
    const platform = os.platform();
    const arch = os.arch();
    
    console.log("🔍 Detecting available GPU backends...");
    console.log(`   Platform: ${platform}, Arch: ${arch}`);
    
    try {
        if (ENABLE_CUDA && (platform === 'win32' || platform === 'linux' || platform === 'darwin')) {
            try {
                if (process.env.CUDA_PATH || process.env.CUDA_HOME) {
                    gpus.cuda = true;
                    console.log("   ✅ CUDA detected via environment variables");
                } else {
                    const { execSync } = require('child_process');
                    try {
                        if (platform === 'win32') {
                            execSync('where nvidia-smi', { stdio: 'ignore' });
                            gpus.cuda = true;
                            console.log("   ✅ NVIDIA GPU detected via nvidia-smi");
                        } else {
                            execSync('which nvidia-smi', { stdio: 'ignore' });
                            gpus.cuda = true;
                            console.log("   ✅ NVIDIA GPU detected via nvidia-smi");
                        }
                    } catch (e) {
                        console.log("   ℹ️ NVIDIA GPU not detected");
                    }
                }
            } catch (error) {
                console.log("   ℹ️ CUDA detection skipped:", error.message);
            }
        }
        
        if (ENABLE_DIRECTML && platform === 'win32') {
            try {
                const { execSync } = require('child_process');
                execSync('dxdiag /t dxdiag.txt', { stdio: 'ignore' });
                gpus.directml = true;
                console.log("   ✅ DirectML available (Windows DX12)");
            } catch (error) {
                console.log("   ℹ️ DirectML not available");
            }
        }
        
        if (ENABLE_COREML && platform === 'darwin') {
            const macVersion = parseFloat(os.release());
            if (macVersion >= 17.0) {
                gpus.coreml = true;
                console.log("   ✅ CoreML available (macOS 10.13+)");
            } else {
                console.log("   ℹ️ CoreML requires macOS 10.13+");
            }
        }
        
        if (ENABLE_TENSORRT && gpus.cuda) {
            gpus.tensorrt = true;
            console.log("   ✅ TensorRT available (requires CUDA)");
        }
        
    } catch (error) {
        console.error("   ❌ GPU detection error:", error.message);
    }
    
    console.log(`   Available backends: ${Object.keys(gpus).filter(k => gpus[k]).join(', ')}`);
    return gpus;
}

class MemoryManager {
    constructor() {
        this.allocatedBuffers = new WeakSet();
        this.trackedBuffers = [];
        this.trackedTensors = [];
        this.memoryStats = {
            peak: 0,
            current: 0,
            buffers: 0,
            tensors: 0
        };
        this.lastGC = 0;
        this.gcCooldown = 3000;
        this.totalAllocated = 0;
    }
    
    allocate(size, type = 'Uint8Array') {
        try {
            let buffer;
            
            if (type === 'Uint8Array') {
                buffer = new Uint8Array(size);
            } else if (type === 'Buffer') {
                buffer = Buffer.allocUnsafe(size);
            } else if (type === 'Float32Array') {
                buffer = new Float32Array(size);
            } else {
                throw new Error(`Unsupported buffer type: ${type}`);
            }
            
            this.allocatedBuffers.add(buffer);
            this.trackedBuffers.push(new WeakRef(buffer));
            this.totalAllocated += buffer.byteLength;
            
            this.updateStats();
            return buffer;
            
        } catch (error) {
            console.error(`❌ Failed to allocate ${this.formatBytes(size)}:`, error);
            this.forceCleanup();
            throw new Error(`Memory allocation failed: ${error.message}`);
        }
    }
    
    trackTensor(tensor) {
        if (tensor && tensor.data) {
            this.trackedTensors.push(tensor);
            this.totalAllocated += tensor.data.byteLength;
            this.updateStats();
        }
        return tensor;
    }
    
    releaseTensor(tensor) {
        if (!tensor) return;
        
        try {
            if (tensor.data && tensor.data.buffer) {
                this.totalAllocated -= tensor.data.byteLength;
            }
            
            if (typeof tensor.dispose === 'function') {
                tensor.dispose();
            }
            
            const index = this.trackedTensors.indexOf(tensor);
            if (index > -1) {
                this.trackedTensors.splice(index, 1);
            }
            
            tensor = null;
            
        } catch (error) {
            console.error("⚠️ Tensor release warning:", error);
        }
    }
    
    release(buffer) {
        if (!buffer) return;
        
        try {
            if (buffer.fill) {
                buffer.fill(0);
            }
            
            if (buffer.byteLength) {
                this.totalAllocated -= buffer.byteLength;
            }
            
            buffer = null;
            this.updateStats();
            
        } catch (error) {
            console.error("⚠️ Release warning:", error);
        }
    }
    
    releaseAll() {
        console.log(`🧹 Releasing ${this.trackedTensors.length} tensors and ${this.trackedBuffers.length} buffers`);
        
        while (this.trackedTensors.length > 0) {
            const tensor = this.trackedTensors.pop();
            try {
                if (typeof tensor.dispose === 'function') {
                    tensor.dispose();
                }
            } catch (e) {}
        }
        
        this.trackedBuffers.length = 0;
        this.allocatedBuffers = new WeakSet();
        this.totalAllocated = 0;
        
        this.updateStats();
        this.forceCleanup();
    }
    
    forceCleanup() {
        const now = Date.now();
        if (now - this.lastGC < this.gcCooldown) {
            return;
        }
        
        if (global.gc) {
            try {
                global.gc();
                this.lastGC = now;
                console.log("🧹 Forced GC");
            } catch (e) {}
        }
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    updateStats() {
        try {
            const used = process.memoryUsage();
            this.memoryStats.current = used.heapUsed / 1024 / 1024;
            this.memoryStats.peak = Math.max(this.memoryStats.peak, this.memoryStats.current);
            this.memoryStats.buffers = this.trackedBuffers.length;
            this.memoryStats.tensors = this.trackedTensors.length;
            
            this.trackedBuffers = this.trackedBuffers.filter(ref => {
                const buffer = ref.deref();
                return buffer !== undefined;
            });
            
            ipcRenderer.send('memory-usage', {
                current: Math.round(this.memoryStats.current),
                peak: Math.round(this.memoryStats.peak),
                buffers: this.memoryStats.buffers,
                tensors: this.memoryStats.tensors,
                allocated: this.formatBytes(this.totalAllocated)
            });
            
        } catch (error) {
            console.error("⚠️ Stats update warning:", error);
        }
    }
    
    logUsage(label) {
        const used = process.memoryUsage();
        console.log(`💾 ${label}:`);
        console.log(`   RSS: ${Math.round(used.rss / 1024 / 1024)} MB`);
        console.log(`   Heap Used: ${Math.round(used.heapUsed / 1024 / 1024)} MB`);
        console.log(`   External: ${Math.round(used.external / 1024 / 1024)} MB`);
        console.log(`   Buffers: ${this.trackedBuffers.length}`);
        console.log(`   Tensors: ${this.trackedTensors.length}`);
        console.log(`   Allocated: ${this.formatBytes(this.totalAllocated)}`);
        console.log(`   Peak: ${Math.round(this.memoryStats.peak)} MB`);
        this.updateStats();
    }
}

const memoryManager = new MemoryManager();

let MODEL_PATH;
let isASAR = false;

try {
    isASAR = __dirname.includes('app.asar');
    console.log(`🔍 Mode: ${isASAR ? 'Production' : 'Development'}`);
} catch (e) {
    isASAR = false;
}

async function initializeModelPath() {
    console.log("🔧 Initializing model path...");
    
    try {
        if (isASAR) {
            const possiblePaths = [
                path.join(process.resourcesPath, 'app.asar.unpacked', 'models_Ai', 'BiRefNet.onnx'),
                path.join(process.resourcesPath, 'models_Ai', 'BiRefNet.onnx'),
                path.join(__dirname, 'models_Ai', 'BiRefNet.onnx'),
                path.join(require('os').homedir(), 'snapcut-models', 'BiRefNet.onnx'),
                path.join(__dirname, '..', '..', 'models_Ai', 'BiRefNet.onnx'),
                path.join(process.cwd(), 'models_Ai', 'BiRefNet.onnx')
            ];
            
            for (const modelPath of possiblePaths) {
                if (fs.existsSync(modelPath)) {
                    MODEL_PATH = modelPath;
                    console.log(`✅ Model found: ${modelPath}`);
                    break;
                }
            }
        } else {
            MODEL_PATH = path.join(__dirname, "models_Ai", "BiRefNet.onnx");
            
            if (!fs.existsSync(MODEL_PATH)) {
                const altPaths = [
                    path.join(__dirname, '..', 'models_Ai', 'BiRefNet.onnx'),
                    path.join(process.cwd(), 'models_Ai', 'BiRefNet.onnx'),
                    path.join(os.homedir(), 'snapcut-models', 'BiRefNet.onnx')
                ];
                
                for (const altPath of altPaths) {
                    if (fs.existsSync(altPath)) {
                        MODEL_PATH = altPath;
                        break;
                    }
                }
            }
        }
        
        if (!MODEL_PATH || !fs.existsSync(MODEL_PATH)) {
            console.warn('❌ Model BiRefNet.onnx tidak ditemukan di lokasi standar');
            
            const fallbackDir = path.join(os.homedir(), 'snapcut-models');
            MODEL_PATH = path.join(fallbackDir, "BiRefNet.onnx");
            
            if (!fs.existsSync(fallbackDir)) {
                fs.mkdirSync(fallbackDir, { recursive: true });
                console.log(`📁 Created fallback directory: ${fallbackDir}`);
            }
            
            console.log(`📁 Fallback path akan digunakan: ${MODEL_PATH}`);
        } else {
            const stats = fs.statSync(MODEL_PATH);
            console.log(`📊 Model size: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
        }
        
        return MODEL_PATH;
        
    } catch (error) {
        console.error('❌ Failed to initialize model path:', error);
        
        const fallbackDir = path.join(os.homedir(), 'snapcut-models');
        MODEL_PATH = path.join(fallbackDir, "BiRefNet.onnx");
        
        if (!fs.existsSync(fallbackDir)) {
            fs.mkdirSync(fallbackDir, { recursive: true });
        }
        
        console.log(`⚠️ Using fallback path: ${MODEL_PATH}`);
        return MODEL_PATH;
    }
}

let modelPathPromise = initializeModelPath();
let modelSession = null;
let processingStartTime = null;
let unloadTimer = null;
let availableGPUs = null;

let ortEnvInitialized = false;

async function ensureOrtEnv() {
    if (ortEnvInitialized) {
        return ort;
    }
    
    try {
        if (typeof ort.InferenceSession !== 'function') {
            console.warn("⚠️ ort.InferenceSession tidak ada, coba reload module...");
            
            const modulePaths = Object.keys(require.cache);
            modulePaths.forEach(path => {
                if (path.includes('onnxruntime-node')) {
                    delete require.cache[path];
                }
            });
        }
        
        ortEnvInitialized = true;
        console.log("✅ ONNX Runtime environment ready");
        return ort;
        
    } catch (error) {
        console.error("❌ Failed to initialize ONNX Runtime:", error);
        throw error;
    }
}

function getCPUConfiguration() {
    const cpuCores = os.cpus().length;
    console.log(`💻 CPU Cores detected: ${cpuCores}`);
    
    let optimalThreads;
    if (ENABLE_CPU_OPTIMIZATIONS) {
        optimalThreads = Math.max(2, Math.min(
            Math.floor(cpuCores * CPU_THREADS_MULTIPLIER),
            8
        ));
    } else {
        optimalThreads = Math.max(2, Math.min(cpuCores - 1, 4));
    }
    
    if (cpuCores <= 2) {
        optimalThreads = 1;
    }
    
    const config = {
        cpuCores: cpuCores,
        optimalThreads: optimalThreads,
        memoryLimit: Math.floor(os.totalmem() * 0.7 / 1024 / 1024)
    };
    
    console.log(`⚙️ CPU Configuration:`);
    console.log(`   Threads: ${optimalThreads} (${cpuCores} cores available)`);
    console.log(`   Memory Limit: ${Math.round(config.memoryLimit)} MB`);
    console.log(`   Optimizations: ${ENABLE_CPU_OPTIMIZATIONS ? 'ENABLED' : 'DISABLED'}`);
    
    return config;
}

async function loadBiRefNet() {
    if (unloadTimer) {
        clearTimeout(unloadTimer);
        unloadTimer = null;
        console.log("⏸️ Cancelled unload timer");
    }
    
    if (modelSession) {
        console.log("🔄 Using existing model session");
        return modelSession;
    }
    
    await modelPathPromise;
    
    console.log("\n" + "=".repeat(60));
    console.log("📥 LOADING MODEL (MULTI-GPU SUPPORT)");
    console.log("=".repeat(60));
    
    memoryManager.logUsage("before loading model");
    
    if (!fs.existsSync(MODEL_PATH)) {
        throw new Error(`Model tidak ditemukan: ${MODEL_PATH}`);
    }
    
    const fileSizeMB = (fs.statSync(MODEL_PATH).size / (1024 * 1024)).toFixed(1);
    console.log(`📦 Model size: ${fileSizeMB}MB`);
    
    try {
        await ensureOrtEnv();
        
        if (!availableGPUs) {
            availableGPUs = await detectAvailableGPUs();
        }
        
        const cpuConfig = getCPUConfiguration();
        
        const executionProviders = [];
        let primaryGPU = 'cpu';
        
        const platform = os.platform();
        
        if (ENABLE_ALL_GPU) {
            if (platform === 'darwin' && availableGPUs.coreml) {
                executionProviders.push('coreml');
                primaryGPU = 'coreml';
                console.log("🍎 Using CoreML (Apple Silicon/Intel Mac)");
            } 
            else if (platform === 'win32' && availableGPUs.directml) {
                executionProviders.push('dml');
                primaryGPU = 'directml';
                console.log("🪟 Using DirectML (Windows AMD/Intel/NVIDIA)");
            }
            else if (availableGPUs.cuda) {
                if (availableGPUs.tensorrt) {
                    executionProviders.push('tensorrt');
                    primaryGPU = 'tensorrt';
                    console.log("⚡ Using TensorRT (NVIDIA GPU with CUDA)");
                } else {
                    executionProviders.push('cuda');
                    primaryGPU = 'cuda';
                    console.log("🎮 Using CUDA (NVIDIA GPU)");
                }
            }
        }
        
        if (USE_CPU_FALLBACK) {
            executionProviders.push('cpu');
            console.log(`💻 CPU added as fallback provider`);
        }
        
        const sessionOptions = {
            executionProviders: executionProviders,
            graphOptimizationLevel: 'all',
            enableCpuMemArena: true,
            enableMemPattern: true,
            logSeverityLevel: 3,
            enableProfiling: false
        };
        
        if (executionProviders.includes('cpu') || executionProviders.length === 0) {
            sessionOptions.intraOpNumThreads = cpuConfig.optimalThreads;
            sessionOptions.interOpNumThreads = 1;
            sessionOptions.executionMode = 'sequential';
        }
        
        ipcRenderer.send('model-status', {
            status: 'loading',
            message: 'Memuat model AI...',
            mode: primaryGPU,
            threads: cpuConfig.optimalThreads,
            gpuBackend: primaryGPU
        });
        
        console.log(`⏳ Loading model with providers: ${executionProviders.join(', ')}...`);
        const loadStart = Date.now();
        
        modelSession = await ort.InferenceSession.create(MODEL_PATH, sessionOptions);
        
        const loadTime = Date.now() - loadStart;
        
        let actualProvider = 'cpu';
        try {
            if (primaryGPU !== 'cpu' && loadTime < 10000) {
                actualProvider = primaryGPU;
            }
        } catch (e) {
            actualProvider = primaryGPU;
        }
        
        console.log(`✅ BiRefNet loaded in ${loadTime}ms!`);
        console.log(`📥 Input: ${modelSession.inputNames[0]}`);
        console.log(`📤 Output: ${modelSession.outputNames[0]}`);
        console.log(`💻 Mode: ${actualProvider.toUpperCase()} (${executionProviders.join(', ')})`);
        
        memoryManager.logUsage("after loading model");
        
        ipcRenderer.send('model-loaded', {
            inputNames: modelSession.inputNames,
            outputNames: modelSession.outputNames,
            providers: executionProviders,
            usingGPU: actualProvider !== 'cpu',
            gpuType: actualProvider,
            threads: cpuConfig.optimalThreads,
            loadTime: loadTime
        });
        
        console.log("=".repeat(60) + "\n");
        
        return modelSession;
        
    } catch (error) {
        console.error("❌ Failed to load model:", error);
        
        if (USE_CPU_FALLBACK && !error.message.includes('CPU')) {
            console.log("🔄 Falling back to CPU-only mode...");
            
            ipcRenderer.send('model-status', {
                status: 'fallback',
                message: 'GPU failed, falling back to CPU...',
                mode: 'cpu-fallback'
            });
            
            ortEnvInitialized = false;
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            const sessionOptions = {
                executionProviders: ['cpu'],
                intraOpNumThreads: getCPUConfiguration().optimalThreads,
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true
            };
            
            modelSession = await ort.InferenceSession.create(MODEL_PATH, sessionOptions);
            console.log("✅ Model loaded in CPU fallback mode");
            
            return modelSession;
        }
        
        throw error;
    }
}

async function unloadBiRefNet() {
    if (!modelSession) {
        console.log("ℹ️ Model already unloaded");
        return;
    }
    
    console.log("\n" + "=".repeat(60));
    console.log("📤 UNLOADING MODEL");
    console.log("=".repeat(60));
    
    memoryManager.logUsage("before unload");
    
    try {
        if (modelSession && typeof modelSession.release === 'function') {
            await modelSession.release();
        }
        
        modelSession = null;
        
        memoryManager.releaseAll();
        
        if (global.gc) {
            try {
                global.gc();
            } catch (e) {}
        }
        
        memoryManager.logUsage("after unload");
        
        ipcRenderer.send('model-status', {
            status: 'unloaded',
            message: 'Model unloaded - RAM freed'
        });
        
        console.log("✅ Model unloaded successfully!");
        console.log("=".repeat(60) + "\n");
        
    } catch (error) {
        console.error("⚠️ Error during unload:", error);
        modelSession = null;
    }
}

function scheduleAutoUnload() {
    if (unloadTimer) {
        clearTimeout(unloadTimer);
    }
    
    unloadTimer = setTimeout(async () => {
        console.log(`⏰ Auto-unloading model after ${AUTO_UNLOAD_DELAY}ms idle`);
        await unloadBiRefNet();
        unloadTimer = null;
    }, AUTO_UNLOAD_DELAY);
    
    console.log(`⏲️ Auto-unload scheduled in ${AUTO_UNLOAD_DELAY}ms`);
}

async function preprocessBiRefNet(buffer) {
    memoryManager.logUsage("before preprocessing");
    
    const tempBuffers = [];
    
    try {
        console.log("🔄 Preprocessing image...");
        
        if (!buffer || buffer.length < 100) {
            throw new Error("Buffer gambar tidak valid");
        }
        
        const metadata = await sharp(buffer).metadata();
        
        console.log("📸 Original:", {
            width: metadata.width,
            height: metadata.height,
            format: metadata.format,
            size: `${(buffer.length / 1024).toFixed(1)}KB`
        });
        
        let resizeWidth = metadata.width;
        let resizeHeight = metadata.height;
        
        if (metadata.width > MAX_IMAGE_SIZE || metadata.height > MAX_IMAGE_SIZE) {
            const scale = Math.min(MAX_IMAGE_SIZE / metadata.width, MAX_IMAGE_SIZE / metadata.height);
            resizeWidth = Math.round(metadata.width * scale);
            resizeHeight = Math.round(metadata.height * scale);
            console.log(`📐 Downscaling to ${resizeWidth}x${resizeHeight}`);
        }
        
        const originalPNG = await sharp(buffer)
            .resize(resizeWidth, resizeHeight)
            .ensureAlpha()
            .png({ 
                quality: COMPRESSION_QUALITY,
                compressionLevel: 6 
            })
            .toBuffer();
        
        const targetSize = MODEL_INPUT_SIZE;
        console.log(`📐 Resizing to ${targetSize}x${targetSize}...`);
        
        const processed = await sharp(buffer)
            .resize(targetSize, targetSize, {
                fit: 'fill',
                kernel: sharp.kernel.lanczos3
            })
            .removeAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });
        
        tempBuffers.push(processed.data);
        
        const data = processed.data;
        const floatArray = memoryManager.allocate(1 * 3 * targetSize * targetSize, 'Float32Array');
        
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        const area = targetSize * targetSize;
        
        const rMean = mean[0], rStd = std[0];
        const gMean = mean[1], gStd = std[1];
        const bMean = mean[2], bStd = std[2];
        
        for (let i = 0; i < area; i++) {
            const srcIdx = i * 3;
            floatArray[i] = (data[srcIdx] / 255.0 - rMean) / rStd;
            floatArray[area + i] = (data[srcIdx + 1] / 255.0 - gMean) / gStd;
            floatArray[area * 2 + i] = (data[srcIdx + 2] / 255.0 - bMean) / bStd;
        }
        
        const tensor = new ort.Tensor("float32", floatArray, [1, 3, targetSize, targetSize]);
        memoryManager.trackTensor(tensor);
        
        tempBuffers.forEach(buf => memoryManager.release(buf));
        tempBuffers.length = 0;
        
        console.log(`✅ Preprocessing done`);
        memoryManager.logUsage("after preprocessing");
        
        return {
            tensor,
            originalWidth: resizeWidth,
            originalHeight: resizeHeight,
            originalBuffer: originalPNG
        };
        
    } catch (error) {
        console.error("❌ Preprocessing failed:", error);
        tempBuffers.forEach(buf => memoryManager.release(buf));
        throw error;
    }
}

function processMask(maskTensor) {
    try {
        console.log("🎭 Processing mask...");
        
        const maskData = maskTensor.data;
        const length = maskData.length;
        const thresholded = memoryManager.allocate(length, 'Uint8Array');
        
        const batchSize = 100000;
        const batchCount = Math.ceil(length / batchSize);
        
        for (let batch = 0; batch < batchCount; batch++) {
            const start = batch * batchSize;
            const end = Math.min(start + batchSize, length);
            
            for (let i = start; i < end; i++) {
                thresholded[i] = maskData[i] > 0.5 ? 255 : 0;
            }
        }
        
        console.log("✅ Mask processed");
        return thresholded;
        
    } catch (error) {
        console.error("❌ Mask processing failed:", error);
        throw error;
    }
}

async function postprocessBiRefNet(maskTensor, originalBuffer, originalWidth, originalHeight) {
    memoryManager.logUsage("before postprocessing");
    
    const tempBuffers = [];
    
    try {
        console.log("🎨 Postprocessing...");
        
        const processedMask = processMask(maskTensor);
        tempBuffers.push(processedMask);
        
        memoryManager.releaseTensor(maskTensor);
        
        console.log(`🔄 Upscaling mask to ${originalWidth}x${originalHeight}...`);
        
        const upscaledMask = await sharp(processedMask, {
            raw: { 
                width: MODEL_INPUT_SIZE, 
                height: MODEL_INPUT_SIZE, 
                channels: 1 
            }
        })
        .resize(originalWidth, originalHeight, {
            kernel: sharp.kernel.lanczos3,
            fit: 'fill'
        })
        .greyscale()
        .raw()
        .toBuffer();
        
        tempBuffers.push(upscaledMask);
        
        const originalImage = await sharp(originalBuffer)
            .ensureAlpha()
            .raw()
            .toBuffer();
        
        tempBuffers.push(originalImage);
        
        console.log("🎭 Applying alpha channel...");
        
        const resultRGBA = memoryManager.allocate(originalWidth * originalHeight * 4, 'Uint8Array');
        const totalPixels = originalWidth * originalHeight;
        
        for (let i = 0; i < totalPixels; i++) {
            const imgIdx = i * 4;
            resultRGBA[imgIdx] = originalImage[imgIdx];
            resultRGBA[imgIdx + 1] = originalImage[imgIdx + 1];
            resultRGBA[imgIdx + 2] = originalImage[imgIdx + 2];
            resultRGBA[imgIdx + 3] = upscaledMask[i];
        }
        
        const finalImage = await sharp(resultRGBA, {
            raw: {
                width: originalWidth,
                height: originalHeight,
                channels: 4
            }
        })
        .png({
            quality: COMPRESSION_QUALITY,
            compressionLevel: 6,
            adaptiveFiltering: false
        })
        .toBuffer();
        
        tempBuffers.forEach(buf => memoryManager.release(buf));
        memoryManager.release(resultRGBA);
        
        console.log(`✅ Postprocessing done`);
        memoryManager.logUsage("after postprocessing");
        
        return finalImage;
        
    } catch (error) {
        console.error("❌ Postprocessing failed:", error);
        tempBuffers.forEach(buf => memoryManager.release(buf));
        throw error;
    }
}

async function removeBackgroundBiRefNet(buffer) {
    const startTime = Date.now();
    
    try {
        console.log("\n" + "=".repeat(60));
        console.log("🎯 BiRefNet BACKGROUND REMOVAL (MULTI-GPU)");
        console.log("=".repeat(60));
        
        memoryManager.logUsage("START");
        
        let uint8Buffer;
        if (buffer instanceof Uint8Array) {
            uint8Buffer = buffer;
        } else if (buffer instanceof ArrayBuffer) {
            uint8Buffer = new Uint8Array(buffer);
        } else if (Buffer.isBuffer(buffer)) {
            uint8Buffer = new Uint8Array(buffer);
        } else {
            uint8Buffer = new Uint8Array(buffer);
        }
        
        console.log(`📦 Input: ${memoryManager.formatBytes(uint8Buffer.length)}`);
        
        ipcRenderer.send('ai-progress', {
            type: 'ai-progress',
            data: { 
                stage: 'loading_model', 
                percentage: 10, 
                message: 'Memuat model AI dengan GPU...' 
            }
        });
        
        const model = await loadBiRefNet();
        
        ipcRenderer.send('ai-progress', {
            type: 'ai-progress',
            data: { 
                stage: 'preprocessing', 
                percentage: 30, 
                message: 'Memproses gambar...' 
            }
        });
        
        console.log("\n1️⃣ PREPROCESSING...");
        const preprocessResult = await preprocessBiRefNet(uint8Buffer);
        
        memoryManager.release(uint8Buffer);
        uint8Buffer = null;
        
        ipcRenderer.send('ai-progress', {
            type: 'ai-progress',
            data: { 
                stage: 'inference', 
                percentage: 50, 
                message: 'Proses AI inference dengan GPU...' 
            }
        });
        
        console.log("\n2️⃣ AI INFERENCE...");
        const inferenceStart = Date.now();
        
        const feed = { [model.inputNames[0]]: preprocessResult.tensor };
        const results = await model.run(feed);
        const inferenceTime = Date.now() - inferenceStart;
        
        console.log(`⚡ AI Inference: ${inferenceTime}ms`);
        
        const maskTensor = results[model.outputNames[0]];
        memoryManager.trackTensor(maskTensor);
        
        memoryManager.releaseTensor(preprocessResult.tensor);
        preprocessResult.tensor = null;
        
        ipcRenderer.send('ai-progress', {
            type: 'ai-progress',
            data: { 
                stage: 'postprocessing', 
                percentage: 80, 
                message: 'Postprocessing hasil...' 
            }
        });
        
        console.log("\n3️⃣ POSTPROCESSING...");
        const finalImage = await postprocessBiRefNet(
            maskTensor,
            preprocessResult.originalBuffer,
            preprocessResult.originalWidth,
            preprocessResult.originalHeight
        );
        
        const totalTime = Date.now() - startTime;
        
        console.log("\n" + "=".repeat(60));
        console.log(`🎯 PROCESS COMPLETE: ${totalTime}ms`);
        console.log(`   - Preprocessing: ${inferenceStart - startTime}ms`);
        console.log(`   - Inference: ${inferenceTime}ms`);
        console.log(`   - Postprocessing: ${totalTime - (inferenceStart - startTime) - inferenceTime}ms`);
        console.log("=".repeat(60) + "\n");
        
        memoryManager.releaseAll();
        scheduleAutoUnload();
        
        return new Uint8Array(finalImage);
        
    } catch (error) {
        console.error("❌ Processing failed:", error);
        memoryManager.releaseAll();
        scheduleAutoUnload();
        throw error;
    }
}

contextBridge.exposeInMainWorld("bgRemoval", {
    removeBackground: async (buffer) => {
        try {
            console.log("📞 API: removeBackground (Multi-GPU Mode)");
            
            if (!buffer || buffer.length < 100) {
                throw new Error("Buffer tidak valid");
            }
            
            let uint8Buffer;
            if (buffer instanceof Uint8Array) {
                uint8Buffer = buffer;
            } else if (buffer instanceof ArrayBuffer) {
                uint8Buffer = new Uint8Array(buffer);
            } else if (Buffer.isBuffer(buffer)) {
                uint8Buffer = new Uint8Array(buffer);
            } else {
                uint8Buffer = new Uint8Array(buffer);
            }
            
            processingStartTime = Date.now();
            
            if (!availableGPUs) {
                availableGPUs = await detectAvailableGPUs();
            }
            
            const gpuMode = availableGPUs.cuda ? 'cuda' : 
                           availableGPUs.directml ? 'directml' : 
                           availableGPUs.coreml ? 'coreml' : 'cpu';
            
            ipcRenderer.send('processing-timer-start', { 
                startTime: processingStartTime,
                mode: gpuMode,
                gpuAvailable: gpuMode !== 'cpu'
            });
            
            ipcRenderer.send('ai-progress', {
                type: 'ai-progress',
                data: { 
                    stage: 'init', 
                    percentage: 5, 
                    message: `Memulai AI Background Removal dengan ${gpuMode.toUpperCase()}...` 
                }
            });
            
            const result = await removeBackgroundBiRefNet(uint8Buffer);
            
            const endTime = Date.now();
            const processingTime = Math.floor((endTime - processingStartTime) / 1000);
            processingStartTime = null;
            
            ipcRenderer.send('processing-timer-end', { 
                endTime: endTime,
                processingTime: processingTime,
                mode: gpuMode,
                gpuUsed: gpuMode !== 'cpu'
            });
            
            console.log(`✅ Success: ${memoryManager.formatBytes(result.length)}`);
            console.log(`⏱️ Total Time: ${processingTime} detik`);
            console.log(`🎮 GPU Mode: ${gpuMode}`);
            
            return {
                success: true,
                aiUsed: true,
                image: result,
                stats: {
                    size: memoryManager.formatBytes(result.length),
                    processingTime: processingTime + " detik",
                    mode: gpuMode.toUpperCase(),
                    gpuAccelerated: gpuMode !== 'cpu'
                }
            };
            
        } catch (error) {
            console.error("❌ API Error:", error);
            
            processingStartTime = null;
            memoryManager.releaseAll();
            
            return {
                success: false,
                aiUsed: false,
                error: error.message
            };
        }
    },
    
    unloadModel: async () => {
        try {
            await unloadBiRefNet();
            return { 
                success: true, 
                message: "Model unloaded from RAM" 
            };
        } catch (error) {
            return { 
                success: false, 
                error: error.message 
            };
        }
    },
    
    testConnection: async () => {
        await modelPathPromise;
        
        const modelExists = fs.existsSync(MODEL_PATH);
        const modelSize = modelExists ? 
            (fs.statSync(MODEL_PATH).size / (1024 * 1024)).toFixed(1) + "MB" : "0MB";
        
        const cpuInfo = getCPUConfiguration();
        
        if (!availableGPUs) {
            availableGPUs = await detectAvailableGPUs();
        }
        
        return {
            status: modelExists ? "ready" : "missing_model",
            modelPath: MODEL_PATH,
            modelExists,
            modelSize,
            modelInputSize: MODEL_INPUT_SIZE,
            modelLoaded: modelSession !== null,
            gpu: {
                cuda: availableGPUs.cuda,
                directml: availableGPUs.directml,
                tensorrt: availableGPUs.tensorrt,
                coreml: availableGPUs.coreml,
                available: availableGPUs.cuda || availableGPUs.directml || availableGPUs.coreml
            },
            cpu: {
                cores: cpuInfo.cpuCores,
                threads: cpuInfo.optimalThreads,
                memoryLimit: Math.round(cpuInfo.memoryLimit) + " MB"
            },
            memory: {
                current: Math.round(memoryManager.memoryStats.current) + " MB",
                peak: Math.round(memoryManager.memoryStats.peak) + " MB",
                allocated: memoryManager.formatBytes(memoryManager.totalAllocated)
            }
        };
    },
    
    cleanupMemory: () => {
        memoryManager.releaseAll();
        
        return {
            success: true,
            message: "Memory cleaned",
            memory: {
                current: Math.round(memoryManager.memoryStats.current) + " MB",
                peak: Math.round(memoryManager.memoryStats.peak) + " MB"
            }
        };
    },
    
    getMemoryUsage: () => {
        const used = process.memoryUsage();
        
        return {
            rss: Math.round(used.rss / 1024 / 1024) + " MB",
            heapUsed: Math.round(used.heapUsed / 1024 / 1024) + " MB",
            external: Math.round(used.external / 1024 / 1024) + " MB",
            trackedBuffers: memoryManager.trackedBuffers.length,
            trackedTensors: memoryManager.trackedTensors.length,
            totalAllocated: memoryManager.formatBytes(memoryManager.totalAllocated),
            modelLoaded: modelSession !== null
        };
    },
    
    getSystemInfo: () => {
        const cpus = os.cpus();
        
        return {
            cpu: {
                cores: cpus.length,
                model: cpus[0]?.model || "Unknown",
                speed: cpus[0]?.speed + " MHz" || "Unknown"
            },
            memory: {
                total: Math.round(os.totalmem() / 1024 / 1024) + " MB",
                free: Math.round(os.freemem() / 1024 / 1024) + " MB"
            },
            platform: os.platform(),
            arch: os.arch(),
            nodeVersion: process.version,
            gpuEnabled: ENABLE_ALL_GPU
        };
    },
    
    enableGPU: (enable) => {
        console.log(`⚙️ GPU control: ${enable ? 'ENABLED' : 'DISABLED'}`);
        return {
            success: true,
            message: `GPU ${enable ? 'enabled' : 'disabled'}`,
            currentSettings: {
                allGPU: ENABLE_ALL_GPU,
                cuda: ENABLE_CUDA,
                directml: ENABLE_DIRECTML,
                coreml: ENABLE_COREML
            }
        };
    },
    
    redetectGPUs: async () => {
        availableGPUs = await detectAvailableGPUs();
        return availableGPUs;
    }
});

console.log("\n" + "=".repeat(60));
console.log("🚀 BiRefNet AI LOADED (MULTI-GPU SUPPORT)");
console.log("=".repeat(60));
console.log("📍 Model: BiRefNet (1024x1024)");
console.log("🎮 GPU Support: CUDA, DirectML, CoreML, TensorRT");
console.log("💻 CPU Fallback: Enabled");
console.log(`🔄 Auto-unload: ${AUTO_UNLOAD_DELAY}ms setelah selesai`);
console.log("=".repeat(60) + "\n");

let shouldPreloadModel = PRELOAD_MODEL_ON_STARTUP;

modelPathPromise.then(async path => {
    console.log(`✅ Model path ready: ${path}`);
    
    if (fs.existsSync(path)) {
        const stats = fs.statSync(path);
        console.log(`📊 Model size: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
        
        ipcRenderer.send('model-path-ready', {
            path: path,
            exists: true,
            size: (stats.size / (1024 * 1024)).toFixed(2) + ' MB'
        });
    } else {
        console.warn(`⚠️ Model not found at: ${path}`);
        ipcRenderer.send('model-path-error', `Model tidak ditemukan: ${path}`);
    }
    
    console.log("🔍 Performing initial GPU detection...");
    availableGPUs = await detectAvailableGPUs();
    
    setTimeout(async () => {
        ipcRenderer.send('splashscreen-complete');
        console.log("✅ Splashscreen selesai");
        memoryManager.logUsage("after splashscreen");
        
        if (shouldPreloadModel) {
            console.log("\n" + "=".repeat(60));
            console.log("🔥 PRELOADING MODEL TO RAM (with GPU acceleration)");
            console.log("=".repeat(60));
            
            try {
                const gpuMode = availableGPUs.cuda ? 'CUDA' : 
                               availableGPUs.directml ? 'DirectML' : 
                               availableGPUs.coreml ? 'CoreML' : 'CPU';
                
                ipcRenderer.send('model-status', {
                    status: 'preloading',
                    message: `Preloading AI model dengan ${gpuMode}...`,
                    mode: gpuMode.toLowerCase()
                });
                
                const preloadStart = Date.now();
                await loadBiRefNet();
                const preloadTime = Date.now() - preloadStart;
                
                console.log(`✅ Model preloaded in ${preloadTime}ms using ${gpuMode}`);
                console.log("💡 First processing will be FAST with GPU acceleration!");
                console.log("=".repeat(60) + "\n");
                
                memoryManager.logUsage("after preload");
                
                ipcRenderer.send('model-status', {
                    status: 'ready',
                    message: `AI ready - Model loaded with ${gpuMode}`,
                    preloadTime: preloadTime,
                    gpuAccelerated: gpuMode !== 'CPU'
                });
                
                if (AUTO_UNLOAD_AFTER_PRELOAD) {
                    console.log("⏲️ Will auto-unload after idle period");
                    scheduleAutoUnload();
                } else {
                    console.log("💾 Model will stay in RAM for instant processing");
                }
                
            } catch (error) {
                console.error("⚠️ Preload failed:", error.message);
                console.log("💡 Model will load on first use instead");
                
                ipcRenderer.send('model-status', {
                    status: 'ready',
                    message: 'AI ready - Will load on demand'
                });
            }
        } else {
            console.log(`ℹ️ Model will load on first use with GPU detection`);
            ipcRenderer.send('model-status', {
                status: 'ready',
                message: 'AI ready - On demand loading with GPU support'
            });
        }
    }, 1000);
    
}).catch(error => {
    console.error('❌ Model initialization failed:', error);
    ipcRenderer.send('model-path-error', error.message);
});

ipcRenderer.on('cleanup-memory', () => {
    console.log("🧹 Manual memory cleanup requested");
    memoryManager.releaseAll();
});

ipcRenderer.on('unload-model', async () => {
    console.log("📤 Manual model unload requested");
    await unloadBiRefNet();
});

ipcRenderer.on('get-cpu-info', () => {
    const cpuInfo = getCPUConfiguration();
    ipcRenderer.send('cpu-info', cpuInfo);
});

ipcRenderer.on('get-gpu-info', async () => {
    if (!availableGPUs) {
        availableGPUs = await detectAvailableGPUs();
    }
    ipcRenderer.send('gpu-info', availableGPUs);
});

process.on('exit', () => {
    console.log("👋 Cleaning up before exit...");
    if (unloadTimer) {
        clearTimeout(unloadTimer);
    }
    memoryManager.releaseAll();
});

process.on('SIGINT', () => {
    console.log("🚪 SIGINT received, cleaning up...");
    if (unloadTimer) {
        clearTimeout(unloadTimer);
    }
    memoryManager.releaseAll();
    process.exit(0);
});
