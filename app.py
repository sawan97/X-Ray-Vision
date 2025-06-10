from flask import Flask, request, jsonify, send_file, render_template
import os
import pydicom
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import logging
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ResNet-9 Model
class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=14):
        super().__init__()
        
        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, class_idx):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()
        
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.mul(activations, weights).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().detach().cpu().numpy()

# Load model and thresholds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = ResNet9(num_classes=14).to(DEVICE)
    model.load_state_dict(torch.load("best_resnet9.pth", map_location=DEVICE))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Initialize Grad-CAM
grad_cam = GradCAM(model, model.res2[1])

# Disease labels
disease_labels = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
    'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
]

# Optimal thresholds
optimal_thresholds = np.array([0.85, 0.9, 0.65, 0.9, 0.55, 0.65, 0.65, 0.65, 0.75, 0.7, 0.7, 0.8, 0.75, 0.7])

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    try:
        return preprocess(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise

def is_medical_image(image):
    try:
        img_array = np.array(image.convert('RGB'))
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        color_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(b - r))
        is_grayscale = color_diff < 10
        gray = np.array(image.convert('L'))
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        is_high_contrast = entropy > 4
        mean_intensity = np.mean(gray)
        is_valid_intensity = 50 < mean_intensity < 200
        logger.info(f"Image validation - Grayscale: {is_grayscale}, Entropy: {entropy:.2f}, Mean Intensity: {mean_intensity:.2f}")
        return is_grayscale and is_high_contrast and is_valid_intensity
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Received upload request")
    if 'file' not in request.files:
        logger.error("No file provided in upload request")
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected in upload request")
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    is_dicom = filename.lower().endswith('.dcm')
    num_slices = 1
    initial_image = None
    img = None

    if is_dicom:
        try:
            dicom = pydicom.dcmread(file_path)
            if len(dicom.pixel_array.shape) == 3:
                num_slices = dicom.pixel_array.shape[0]
                img_array = dicom.pixel_array[0]
            else:
                img_array = dicom.pixel_array
            img = Image.fromarray(img_array).convert('L')
            initial_image_path = os.path.join(UPLOAD_FOLDER, f"{filename}_0.jpg")
            img.save(initial_image_path)
            initial_image = f"/get_slice/{filename}/0"
            logger.info(f"DICOM processed: {num_slices} slices")
        except Exception as e:
            logger.error(f"Invalid DICOM file: {str(e)}")
            return jsonify({'error': f'Invalid DICOM file: {str(e)}'}), 400
    else:
        try:
            img = Image.open(file_path).convert('L')
            initial_image_path = file_path
            initial_image = f"/get_slice/{filename}/0"
            logger.info("Non-DICOM image processed")
        except Exception as e:
            logger.error(f"Invalid image file: {str(e)}")
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    return jsonify({
        'initial_image': initial_image,
        'num_slices': num_slices,
        'is_dicom': is_dicom,
        'filename': filename
    })

@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    logger.info(f"Prediction request for: {filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404

    is_dicom = filename.lower().endswith('.dcm')
    img = None

    if is_dicom:
        try:
            dicom = pydicom.dcmread(file_path)
            img_array = dicom.pixel_array[0] if len(dicom.pixel_array.shape) == 3 else dicom.pixel_array
            img = Image.fromarray(img_array).convert('L')
            logger.info("DICOM image loaded for prediction")
        except Exception as e:
            logger.error(f"Invalid DICOM file: {str(e)}")
            return jsonify({'error': f'Invalid DICOM file: {str(e)}'}), 400
    else:
        try:
            img = Image.open(file_path).convert('L')
            logger.info("Non-DICOM image loaded for prediction")
        except Exception as e:
            logger.error(f"Invalid image file: {str(e)}")
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    if not is_medical_image(img):
        logger.info("Non-medical image detected, returning no diseases")
        return jsonify({
            'diseases': [],
            'message': 'Non-medical image detected. No diseases found.'
        })

    try:
        img_tensor = preprocess_image(img)
        with torch.no_grad():
            predictions = model(img_tensor).sigmoid().cpu().numpy()[0]
        logger.info("Model inference completed")
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

    detected_diseases = [
        {"disease": disease_labels[i], "probability": float(predictions[i])}
        for i in range(len(predictions)) if predictions[i] > optimal_thresholds[i]
    ]
    detected_diseases = sorted(detected_diseases, key=lambda x: x['probability'], reverse=True)[:3]
    logger.info(f"Detected diseases: {detected_diseases}")

    if detected_diseases and all(d['probability'] < 0.75 for d in detected_diseases):
        logger.info("Low confidence predictions, treating as no disease")
        detected_diseases = []

    response = {
        'diseases': detected_diseases
    }
    if detected_diseases:
        response['top_disease'] = detected_diseases[0]['disease']
    return jsonify(response)

@app.route('/get_slice/<filename>/<int:slice_index>')
def get_slice(filename, slice_index):
    logger.info(f"Request for slice: {filename}/{slice_index}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404

    if filename.lower().endswith('.dcm'):
        try:
            dicom = pydicom.dcmread(file_path)
            if len(dicom.pixel_array.shape) == 3 and slice_index < dicom.pixel_array.shape[0]:
                img_array = dicom.pixel_array[slice_index]
            else:
                img_array = dicom.pixel_array
            img = Image.fromarray(img_array).convert('L')
            logger.info(f"DICOM slice {slice_index} retrieved")
        except Exception as e:
            logger.error(f"Error reading DICOM: {str(e)}")
            return jsonify({'error': f'Error reading DICOM: {str(e)}'}), 400
    else:
        try:
            img = Image.open(file_path).convert('L')
            logger.info("Non-DICOM image retrieved")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return jsonify({'error': f'Error reading image: {str(e)}'}), 400

    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/explain/<filename>/<int:slice_index>')
def explain(filename, slice_index):
    logger.info(f"Explain request for: {filename}/{slice_index}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404

    if filename.lower().endswith('.dcm'):
        try:
            dicom = pydicom.dcmread(file_path)
            if len(dicom.pixel_array.shape) == 3 and slice_index < dicom.pixel_array.shape[0]:
                img_array = dicom.pixel_array[slice_index]
            else:
                img_array = dicom.pixel_array
            img = Image.fromarray(img_array).convert('L')
            logger.info(f"DICOM slice {slice_index} retrieved for Grad-CAM")
        except Exception as e:
            logger.error(f"Error reading DICOM: {str(e)}")
            return jsonify({'error': f'Error reading DICOM: {str(e)}'}), 400
    else:
        try:
            img = Image.open(file_path).convert('L')
            logger.info("Non-DICOM image retrieved for Grad-CAM")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return jsonify({'error': f'Error reading image: {str(e)}'}), 400

    try:
        img_tensor = preprocess_image(img)
        
        # Run prediction to get detected diseases
        with torch.no_grad():
            predictions = model(img_tensor).sigmoid().cpu().numpy()[0]
        detected_diseases = [
            {"disease": disease_labels[i], "probability": float(predictions[i]), "index": i}
            for i in range(len(predictions)) if predictions[i] > optimal_thresholds[i]
        ]
        detected_diseases = sorted(detected_diseases, key=lambda x: x['probability'], reverse=True)[:3]
        
        if not detected_diseases:
            logger.info("No diseases detected for heatmap")
            return jsonify({'error': 'No diseases detected for heatmap'}), 400

        # Generate combined Grad-CAM for detected diseases
        combined_cam = None
        total_weight = 0
        for disease in detected_diseases:
            class_idx = disease['index']
            cam = grad_cam.generate(img_tensor, class_idx)
            weight = disease['probability']
            if combined_cam is None:
                combined_cam = np.zeros_like(cam)
            combined_cam += weight * cam
            total_weight += weight
        
        if total_weight > 0:
            combined_cam /= total_weight
        
        # Resize CAM to original image size
        original_size = img.size
        cam = cv2.resize(combined_cam, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        # Convert original image to RGB for overlay
        img_np = np.array(img.convert('RGB'))
        heatmap = plt.cm.jet(cam)[:, :, :3] * 255
        heatmap = heatmap.astype(np.uint8)
        
        # Overlay heatmap on original image
        alpha = 0.4
        overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
        
        # Convert back to PIL Image
        result = Image.fromarray(overlay)
        
        # Save to BytesIO
        img_io = io.BytesIO()
        result.save(img_io, format='JPEG')
        img_io.seek(0)
        logger.info("Combined Grad-CAM heatmap generated and sent")
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {str(e)}")
        return jsonify({'error': f'Grad-CAM generation failed: {str(e)}'}), 500

@app.route('/')
def index():
    logger.info("Serving index.html")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Failed to render index.html: {str(e)}")
        return jsonify({'error': f'Failed to load index.html: {str(e)}'}), 404

if __name__ == '__main__':
    app.run(debug=True)