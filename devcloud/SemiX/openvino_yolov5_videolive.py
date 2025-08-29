import cv2
import numpy as np
import openvino.runtime as ov
import os
import sys
import time

# Load and preprocess image/frame for inference
def preprocess_for_inference(frame):
    """Preprocesses a single image frame for OpenVINO model inference."""
    try:
        # Convert to RGB, resize, transpose, normalize for model input
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_chw = img_resized.transpose(2, 0, 1)  # HWC to CHW
        img_normalized = img_chw.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
        return img_batch
    except Exception as e:
        print(f"Error preprocessing frame for inference: {e}")
        return None

# Run inference with OpenVINO
def run_openvino_inference(compiled_model, image_input):
    """
    Runs inference on the preprocessed image using an already compiled OpenVINO model.
    """
    try:
        output_layer = compiled_model.output(0)
        start_time = time.time()
        result = compiled_model([image_input])[output_layer]
        end_time = time.time()
        time_taken = end_time - start_time
        return result, time_taken
    except Exception as e:
        print(f"Error during OpenVINO inference: {e}")
        return None, 0.0

def postprocess_yolo_output(output, conf_threshold=0.5, iou_threshold=0.5):
    """Post-processes YOLOv5 model output to get final bounding boxes, scores, and class IDs."""
    if output.ndim == 2:
        output = np.expand_dims(output, axis=0)
    elif output.ndim != 3 or output.shape[0] != 1:
        print(f"Warning: Unexpected output shape {output.shape}. Attempting to proceed, but results might be incorrect.")
        if output.ndim == 3 and output.shape[0] > 1:
            print("Processing first batch item only.")
            output = output[0:1, :, :]
        elif output.ndim > 3:
            output = np.squeeze(output)
            if output.ndim == 2:
                output = np.expand_dims(output, axis=0)
            else:
                print("Cannot reformat output for post-processing. Returning empty.")
                return np.array([]), np.array([]), np.array([])

    boxes = output[0, :, :4]
    
    if output.shape[2] >= 8: # Assuming 3 classes
        obj_conf = output[0, :, 4]
        class_scores = output[0, :, 5:8]
        scores = obj_conf * np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
    elif output.shape[2] == 6: # Common for 1 class + objectness
        obj_conf = output[0, :, 4]
        class_scores = output[0, :, 5]
        scores = obj_conf * class_scores
        class_ids = np.zeros(output.shape[1], dtype=np.int32)
        print("Warning: Model output seems to have only one class score. Assuming single class detection.")
    else:
        print(f"Error: Model output has {output.shape[2]} columns. Expected at least 6 (for 1 class) or 8 (for 3 classes). Cannot parse.")
        return np.array([]), np.array([]), np.array([])

    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert boxes from [x_center, y_center, width, height] to [x1, y1, width, height] for NMSBoxes
    x_centers = boxes[:, 0]
    y_centers = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]

    x1 = x_centers - widths / 2
    y1 = y_centers - heights / 2
    
    boxes_for_nms = np.stack([x1, y1, widths, heights], axis=1)

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
    
    # Convert final boxes to (x1, y1, x2, y2) for easier display/interpretation
    final_boxes_xyxy = np.copy(boxes)
    final_boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    final_boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    final_boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    final_boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    return final_boxes_xyxy, scores, class_ids

def draw_detections_on_frame(frame, detections, labels):
    """Draws bounding boxes and labels on a single image frame."""
    # Since we are showing live camera feed, we should resize to a standard size for consistency
    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    
    # Scale boxes to original frame size (since we resized for inference)
    scale_x = frame_width / 640.0
    scale_y = frame_height / 640.0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    for label, score, box in detections:
        x1_scaled = int(box[0] * scale_x)
        y1_scaled = int(box[1] * scale_y)
        x2_scaled = int(box[2] * scale_x)
        y2_scaled = int(box[3] * scale_y)
        
        # Determine color based on class
        color = (0, 255, 0)  # Green for "With Helmet"
        if label == "Without Helmet":
            color = (0, 0, 255)  # Red for "Without Helmet"
        elif label == "licence":
            color = (255, 0, 0)  # Blue for "licence"
        
        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        
        text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        text_bg_y1 = max(0, y1_scaled - text_height - 10)
        cv2.rectangle(frame, (x1_scaled, text_bg_y1), (x1_scaled + text_width, y1_scaled), color, -1)
        
        cv2.putText(frame, text, (x1_scaled, y1_scaled - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return frame

def run_camera_detection(model_xml_path, device="AUTO"):
    """
    Main function to run live detection on the PC camera feed.
    """
    
    labels = ["With Helmet", "Without Helmet", "licence"]
    
    core = ov.Core()
    available_devices = core.available_devices
    print(f"Available devices for OpenVINO: {available_devices}")
    
    if device == "AUTO":
        if "NPU" in available_devices:
            device_to_use = "NPU"
        elif "GPU" in available_devices:
            device_to_use = "GPU"
        else:
            print("Warning: Neither NPU nor GPU available. Defaulting to CPU.")
            device_to_use = "CPU"
    else:
        if device not in available_devices:
            print(f"Error: Specified device '{device}' not available. Available: {available_devices}")
            sys.exit(1)
        device_to_use = device

    device_name = core.get_property(device_to_use, "FULL_DEVICE_NAME")
    print(f"Running inference on device: {device_to_use} ({device_name})")
    
    normalized_model_xml_path = os.path.normpath(model_xml_path)
    if not os.path.exists(normalized_model_xml_path):
        print(f"Error: OpenVINO model XML file not found at {normalized_model_xml_path}")
        sys.exit(1)
    
    model = core.read_model(model=normalized_model_xml_path)
    compiled_model = core.compile_model(model, device_to_use)
    
    # Start video capture from PC camera (index 0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera. Please check your camera connection.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # Preprocess frame and run inference
        img_np_for_inference = preprocess_for_inference(frame)
        if img_np_for_inference is not None:
            output, time_taken = run_openvino_inference(compiled_model, img_np_for_inference)
            
            # Post-process and get detections
            boxes, scores, class_ids = postprocess_yolo_output(output)
            
            detections = []
            if len(boxes) > 0:
                for i in range(len(class_ids)):
                    detections.append((labels[class_ids[i]], scores[i], boxes[i]))
            
            # Draw detections and display
            annotated_frame = draw_detections_on_frame(frame, detections, labels)
            
            # Add timing information to the frame
            inference_text = f"Inference: {time_taken:.4f}s"
            cv2.putText(annotated_frame, inference_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('PC Camera Detection', annotated_frame)
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")


# Test with a specific image and pre-converted model
if __name__ == "__main__":
    # --- Configuration ---
    openvino_model_dir = "yolov5/openvino_model"
    openvino_xml_path = os.path.join(openvino_model_dir, "best_model.xml")
    device_to_use = "AUTO"  # Change to "CPU", "GPU", "NPU" or "AUTO"

    # --- Run on PC Camera ---
    print("\n\n--- Starting live detection on PC camera ---")
    run_camera_detection(openvino_xml_path, device=device_to_use)
