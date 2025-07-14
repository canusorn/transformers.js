import './style.css';

import { AutoModel, AutoProcessor, env, RawImage } from '@xenova/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Proxy the WASM backend to prevent the UI from freezing
env.backends.onnx.wasm.proxy = true;

// Constants
const EXAMPLE_URL = 'https://images.pexels.com/photos/5965592/pexels-photo-5965592.jpeg?auto=compress&cs=tinysrgb&w=1024';

// Reference the elements that we will need
const status = document.getElementById('status');
const queueInfo = document.getElementById('queue-info');
const fileUpload = document.getElementById('upload');
const originalImagesContainer = document.getElementById('original-images');
const processedImagesContainer = document.getElementById('processed-images');
const example = document.getElementById('example');
const clearQueueBtn = document.getElementById('clear-queue');
const downloadAllBtn = document.getElementById('download-all');

// Queue management
let imageQueue = [];
let processedImages = [];
let isProcessing = false;
let currentProcessingIndex = 0;

// Load model and processor
status.textContent = 'Loading model...';

const model = await AutoModel.from_pretrained('briaai/RMBG-1.4', {
    // Do not require config.json to be present in the repository
    config: { model_type: 'custom' },
});

const processor = await AutoProcessor.from_pretrained('briaai/RMBG-1.4', {
    // Do not require config.json to be present in the repository
    config: {
        do_normalize: true,
        do_pad: false,
        do_rescale: true,
        do_resize: true,
        image_mean: [0.5, 0.5, 0.5],
        feature_extractor_type: "ImageFeatureExtractor",
        image_std: [1, 1, 1],
        resample: 2,
        rescale_factor: 0.00392156862745098,
        size: { width: 1024, height: 1024 },
    }
});

status.textContent = 'Ready';
updateQueueInfo();

// Initialize upload section visibility
const uploadSection = document.getElementById('upload-section');
uploadSection.style.display = 'block';

// Event listeners
example.addEventListener('click', (e) => {
    e.preventDefault();
    addToQueue({ url: EXAMPLE_URL, name: 'Example Image', type: 'example' });
});

fileUpload.addEventListener('change', function (e) {
    const files = Array.from(e.target.files);
    if (files.length === 0) {
        return;
    }

    files.forEach(file => {
        const reader = new FileReader();
        reader.onload = e2 => {
            addToQueue({ 
                url: e2.target.result, 
                name: file.name, 
                type: 'file',
                size: file.size 
            });
        };
        reader.readAsDataURL(file);
    });
    
    // Clear the input
    e.target.value = '';
});

clearQueueBtn.addEventListener('click', () => {
    if (!isProcessing) {
        clearQueue();
    }
});

downloadAllBtn.addEventListener('click', () => {
    downloadAllProcessed();
});

// Queue management functions
function addToQueue(imageData) {
    const queueItem = {
        id: `img_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
        ...imageData,
        status: 'pending',
        addedAt: new Date()
    };
    
    imageQueue.push(queueItem);
    
    // Add to original images UI immediately
    addOriginalImageToResults(queueItem);
    
    updateQueueInfo();
    
    if (!isProcessing) {
        processQueue();
    }
}

function clearQueue() {
    imageQueue = [];
    processedImages = [];
    originalImagesContainer.innerHTML = '';
    processedImagesContainer.innerHTML = '';
    updateQueueInfo();
    status.textContent = 'Queue cleared';
    
    // Show upload section when queue is cleared
    const uploadSection = document.getElementById('upload-section');
    uploadSection.style.display = 'block';
}

function updateQueueInfo() {
    const pending = imageQueue.filter(item => item.status === 'pending').length;
    const processing = imageQueue.filter(item => item.status === 'processing').length;
    const completed = processedImages.length;
    
    queueInfo.textContent = `Queue: ${pending} pending, ${processing} processing, ${completed} completed`;
    
    clearQueueBtn.disabled = isProcessing;
    downloadAllBtn.disabled = completed === 0;
    
    // Show/hide upload section based on queue status
    const uploadSection = document.getElementById('upload-section');
    if (imageQueue.length === 0) {
        uploadSection.style.display = 'block';
    } else {
        uploadSection.style.display = 'none';
    }
}

async function processQueue() {
    if (isProcessing || imageQueue.length === 0) {
        return;
    }
    
    isProcessing = true;
    updateQueueInfo();
    
    while (imageQueue.length > 0) {
        const item = imageQueue.find(item => item.status === 'pending');
        if (!item) break;
        
        item.status = 'processing';
        currentProcessingIndex++;
        
        // Update original image status in UI
        updateOriginalImageStatus(item.id, 'processing');
        
        updateQueueInfo();
        
        try {
            const result = await processImage(item);
            
            // Remove from queue and add to processed
            imageQueue = imageQueue.filter(qItem => qItem.id !== item.id);
            processedImages.push(result);
            
            // Add to processed images UI
            addProcessedImageToResults(result);
            
        } catch (error) {
            console.error('Error processing image:', error);
            item.status = 'error';
            item.error = error.message;
            // Update the existing original image with error state
            updateOriginalImageStatus(item.id, 'error');
        }
        
        updateQueueInfo();
    }
    
    isProcessing = false;
    status.textContent = 'All images processed!';
    updateQueueInfo();
}

// Process a single image
async function processImage(item) {
    status.textContent = `Processing ${item.name}... (${currentProcessingIndex})`;
    
    // Read image
    const image = await RawImage.fromURL(item.url);
    
    // Preprocess image
    const { pixel_values } = await processor(image);
    
    // Predict alpha matte
    const { output } = await model({ input: pixel_values });
    
    // Resize mask back to original size
    const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);
    
    // Create new canvas
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext('2d');
    
    // Draw original image output to canvas
    ctx.drawImage(image.toCanvas(), 0, 0);
    
    // Update alpha channel
    const pixelData = ctx.getImageData(0, 0, image.width, image.height);
    for (let i = 0; i < mask.data.length; ++i) {
        pixelData.data[4 * i + 3] = mask.data[i];
    }
    ctx.putImageData(pixelData, 0, 0);
    
    return {
        ...item,
        originalImage: image,
        processedCanvas: canvas,
        processedAt: new Date(),
        dimensions: { width: image.width, height: image.height }
    };
}

// Add original image to original images section
function addOriginalImageToResults(item) {
    const resultDiv = document.createElement('div');
    resultDiv.className = 'image-result original-image';
    resultDiv.setAttribute('data-id', item.id);
    
    const statusClass = item.status === 'error' ? 'error' : item.status;
    const statusText = item.status === 'error' ? `❌ Error: ${item.error}` : 
                      item.status === 'processing' ? '⏳ Processing...' : 
                      item.status === 'pending' ? '⏸️ Pending' : '✅ Ready';
    
    resultDiv.innerHTML = `
        <img src="${item.url}" alt="${item.name}" />
        <div class="image-info">
            <div><strong>${item.name}</strong></div>
            <div class="status ${statusClass}">${statusText}</div>
            <div>Added: ${item.addedAt.toLocaleTimeString()}</div>
            ${item.size ? `<div>Size: ${(item.size / 1024).toFixed(1)} KB</div>` : ''}
        </div>
        <div class="image-actions">
            <button onclick="removeOriginalImage('${item.id}')">Remove</button>
        </div>
    `;
    
    originalImagesContainer.appendChild(resultDiv);
}

// Add processed image to processed images section
function addProcessedImageToResults(result) {
    const resultDiv = document.createElement('div');
    resultDiv.className = 'image-result processed-image';
    resultDiv.setAttribute('data-id', result.id);
    resultDiv.innerHTML = `
        <canvas></canvas>
        <div class="image-info">
            <div><strong>${result.name}</strong></div>
            <div>${result.dimensions.width} × ${result.dimensions.height}</div>
            <div>✅ Processed: ${result.processedAt.toLocaleTimeString()}</div>
        </div>
        <div class="image-actions">
            <button onclick="downloadImage('${result.id}')">Download</button>
            <button onclick="removeProcessedImage('${result.id}')">Remove</button>
        </div>
    `;
    
    const canvas = resultDiv.querySelector('canvas');
    canvas.width = result.processedCanvas.width;
    canvas.height = result.processedCanvas.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(result.processedCanvas, 0, 0);
    
    processedImagesContainer.appendChild(resultDiv);
}

// Download functions
function downloadImage(imageId) {
    console.log('Download requested for ID:', imageId);
    console.log('Available processed images:', processedImages.map(img => ({ id: img.id, name: img.name })));
    
    const result = processedImages.find(img => img.id === imageId);
    if (!result) {
        console.error('Image not found for download:', imageId);
        alert('Error: Image not found for download');
        return;
    }
    
    try {
        const link = document.createElement('a');
        const fileName = result.name.includes('.') ? 
            result.name.substring(0, result.name.lastIndexOf('.')) : 
            result.name;
        link.download = `${fileName}_no_bg.png`;
        link.href = result.processedCanvas.toDataURL('image/png');
        
        // Add the link to the document temporarily
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log('Download initiated for:', result.name);
    } catch (error) {
        console.error('Download error:', error);
        alert('Error downloading image: ' + error.message);
    }
}

function downloadAllProcessed() {
    if (processedImages.length === 0) {
        alert('No processed images to download');
        return;
    }
    
    console.log('Downloading all processed images:', processedImages.length);
    processedImages.forEach((result, index) => {
        // Stagger downloads to avoid browser blocking
        setTimeout(() => downloadImage(result.id), index * 200);
    });
}

function removeOriginalImage(imageId) {
    // Remove from queue if still pending
    imageQueue = imageQueue.filter(img => img.id !== imageId);
    updateQueueInfo();
    
    // Remove from UI
    const element = originalImagesContainer.querySelector(`[data-id="${imageId}"]`);
    if (element) {
        element.remove();
    }
}

function removeProcessedImage(imageId) {
    processedImages = processedImages.filter(img => img.id !== imageId);
    updateQueueInfo();
    
    // Remove from UI
    const element = processedImagesContainer.querySelector(`[data-id="${imageId}"]`);
    if (element) {
        element.remove();
    }
}

// Update original image status in UI
function updateOriginalImageStatus(imageId, status) {
    const element = originalImagesContainer.querySelector(`[data-id="${imageId}"]`);
    if (element) {
        const statusElement = element.querySelector('.status');
        if (statusElement) {
            statusElement.className = `status ${status}`;
            let statusText;
            if (status === 'error') {
                const item = imageQueue.find(img => img.id === imageId);
                statusText = item && item.error ? `❌ ${item.error}` : '❌ Processing Error';
            } else {
                statusText = status === 'processing' ? '⏳ Processing...' : 
                            status === 'pending' ? '⏸️ Pending' : '✅ Ready';
            }
            statusElement.textContent = statusText;
        }
    }
}

// Make functions global for onclick handlers
window.downloadImage = downloadImage;
window.removeOriginalImage = removeOriginalImage;
window.removeProcessedImage = removeProcessedImage;
