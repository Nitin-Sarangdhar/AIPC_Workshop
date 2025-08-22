import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
import cv2
import numpy as np
import openvino as ov
import json
import urllib.request
import sys
import os
import subprocess
import time

# --- (Rest of your script remains the same) ---
# Download ImageNet labels, load ResNet-50 model, preprocess function, etc.
# ...

# Define image preprocessing (same as ImageNet training)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image {image_path}")
            sys.exit(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)

# --- MODIFIED SECTION ---

def convert_and_save_openvino_model(model, output_path="resnet50_openvino", onnx_path="resnet50.onnx"):
    """
    Converts PyTorch model to OpenVINO IR if the files do not already exist.
    """
    ir_model_xml = os.path.join(output_path, "resnet50.xml")
    ir_model_bin = os.path.join(output_path, "resnet50.bin")

    # Check if ONNX and OpenVINO files already exist
    if os.path.exists(ir_model_xml) and os.path.exists(ir_model_bin):
        print("OpenVINO IR files already exist. Skipping conversion.")
        return output_path

    print("OpenVINO IR files not found. Starting conversion...")

    try:
        # Export model to ONNX
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
        print("Model successfully exported to ONNX.")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        sys.exit(1)

    # Convert ONNX to OpenVINO IR using command-line mo tool
    try:
        os.makedirs(output_path, exist_ok=True)
        mo_cmd = [
            "mo",  # Model Optimizer command
            "--input_model", onnx_path,
            "--output_dir", output_path,
            "--input_shape", "[1,3,224,224]"
        ]
        result = subprocess.run(mo_cmd, check=True, capture_output=True, text=True)
        print(f"OpenVINO model conversion successful: {result.stdout}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting to OpenVINO IR: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running Model Optimizer: {e}")
        sys.exit(1)

# Run inference with OpenVINO on specified device
def run_openvino_inference(image, ir_path="resnet50_openvino", device="AUTO"):
    try:
        core = ov.Core()
        available_devices = core.available_devices
        if device not in available_devices:
            print(f"Warning: {device} not available, falling back to CPU")
            device = "CPU"
        model_xml = os.path.join(ir_path, "resnet50.xml")
        model_bin = os.path.join(ir_path, "resnet50.bin")
        if not os.path.exists(model_xml) or not os.path.exists(model_bin):
            print(f"Error: OpenVINO model files {model_xml} or {model_bin} not found. Please run the conversion first.")
            sys.exit(1)
        model = core.read_model(model=model_xml)
        compiled_model = core.compile_model(model, device)
        output_layer = compiled_model.output(0)
        
        # Run inference
        result = compiled_model([image])[output_layer]
        return result
    except Exception as e:
        print(f"Error running OpenVINO inference on {device}: {e}")
        sys.exit(1)

# Main function to classify image
def classify_image(image_path):
    # Load PyTorch model (if not already loaded) and labels
    try:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
    except Exception as e:
        print(f"Error loading ResNet-50 model: {e}")
        sys.exit(1)
    
    # Download ImageNet labels (1000 classes)
    try:
        LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(LABELS_URL) as url:
            labels = json.loads(url.read().decode())
    except Exception as e:
        print(f"Error downloading ImageNet labels: {e}")
        sys.exit(1)

    # Convert model to OpenVINO format and save to disk
    ir_path = convert_and_save_openvino_model(model)

    # Load and preprocess image
    img_tensor = load_image(image_path)
    img_np = img_tensor.numpy()  # Convert to NumPy for OpenVINO
    
    # Run inference with timing
    start_time = time.time()
    predictions = run_openvino_inference(img_np, ir_path, device="NPU")
    inference_time = time.time() - start_time
    
    # Apply softmax to normalize predictions to probabilities
    predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    
    # Get top prediction
    pred_idx = np.argmax(predictions, axis=1)[0]
    pred_label = labels[pred_idx]
    confidence = predictions[0][pred_idx]
    
    # Check if prediction is a dog (ImageNet dog classes: indices 151 to 268)
    is_dog = 151 <= pred_idx <= 268
    
    # Print results
    print(f"Prediction: {pred_label}, Confidence: {confidence:.4f}")
    print(f"Is Dog: {is_dog}")
    print(f"Inference Time: {inference_time:.4f} seconds ({inference_time*1000:.2f} ms)")
    
    # Display image with prediction (for teaching)
    img = cv2.imread(image_path)
    cv2.putText(img, f"{pred_label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with an image
if __name__ == "__main__":
    image_path = "dog.jpg"  # Replace with path to your dog image
    classify_image(image_path)
