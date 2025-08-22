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
    image_display = cv2.resize(frame, (640, 640))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    for label, score, box in detections:
        x1, y1, x2, y2 = box
        
        # Determine color based on class
        color = (0, 255, 0)  # Green for "With Helmet"
        if label == "Without Helmet":
            color = (0, 0, 255)  # Red for "Without Helmet"
        elif label == "licence":
            color = (255, 0, 0)  # Blue for "licence"
        
        cv2.rectangle(image_display, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        text_bg_y1 = max(0, y1 - text_height - 10)
        cv2.rectangle(image_display, (x1, text_bg_y1), (x1 + text_width, y1), color, -1)
        
        cv2.putText(image_display, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return image_display

def run_detection_on_input(input_path, model_xml_path, device="AUTO", output_dir="detected_media"):
    """
    Main function to handle both images and videos.
    """
    
    # Check if input is a video or image based on file extension
    file_extension = os.path.splitext(input_path)[1].lower()
    is_video = file_extension in ['.mp4', '.avi', '.mov', '.mkv']
    
    os.makedirs(output_dir, exist_ok=True)
    
    labels = ["With Helmet", "Without Helmet", "licence"]
    
    # A single object to handle OpenVINO model compilation
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
    
    # Handle image input
    if not is_video:
        print(f"Processing image from: {input_path}")
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Could not load image {input_path}")
            return
        
        img_np_for_inference = preprocess_for_inference(frame)
        if img_np_for_inference is not None:
            # Run inference and time it
            output, time_taken = run_openvino_inference(compiled_model, img_np_for_inference)
            
            # Post-process and get detections
            boxes, scores, class_ids = postprocess_yolo_output(output)
            
            detections = []
            if len(boxes) > 0:
                for i in range(len(class_ids)):
                    box_int = [int(coord) for coord in boxes[i]]
                    detections.append((labels[class_ids[i]], scores[i], box_int))
            
            # Draw detections and save image
            annotated_frame = draw_detections_on_frame(frame, detections, labels)
            
            base_filename = os.path.basename(input_path)
            name, ext = os.path.splitext(base_filename)
            output_filename = f"{name}_detected{ext}"
            output_filepath = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_filepath, annotated_frame)
            print(f"Output image saved to: {output_filepath}")

            print("\n--- Detections ---")
            if detections:
                for label, score, box in detections:
                    print(f"Class: {label}, Confidence: {score:.4f}, Box: {box}")
                
                has_helmet = any(label == "With Helmet" and score > 0.5 for label, score, _ in detections)
                if has_helmet:
                    print(f"\nHelmet Violation Status: No Helmet Violation Detected (Found 'With Helmet' class with high confidence).")
                else:
                    has_no_helmet_violation = any(label == "Without Helmet" and score > 0.5 for label, score, _ in detections)
                    if has_no_helmet_violation:
                        print(f"\nHelmet Violation Status: VIOLATION DETECTED (Found 'Without Helmet' class with high confidence).")
                    else:
                        print(f"\nHelmet Violation Status: Indeterminate (No 'With Helmet' or 'Without Helmet' class found with high confidence).")
            else:
                print("No objects detected above confidence threshold.")
                print("\nHelmet Violation Status: Indeterminate (No objects found).")
            
            print(f"\nTime taken to process single image frame: {time_taken:.4f} seconds")


    # Handle video input
    else:
        print(f"Processing video from: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
        
        # Get video properties for output
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        base_filename = os.path.basename(input_path)
        name, ext = os.path.splitext(base_filename)
        output_filename = f"{name}_detected.mp4"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4 files
        out = cv2.VideoWriter(output_filepath, fourcc, fps, (640, 640))
        
        frame_count = 0
        total_time_taken = 0
        
        # Calculate expected time per frame for real-time comparison
        if fps > 0:
            time_per_frame = 1.0 / fps
        else:
            time_per_frame = 0 # Default to 0 if FPS is not available

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}...")
            
            # Preprocess frame and run inference
            img_np_for_inference = preprocess_for_inference(frame)
            
            if img_np_for_inference is not None:
                # Run inference and time it
                output, time_taken = run_openvino_inference(compiled_model, img_np_for_inference)
                total_time_taken += time_taken
                
                # Check if the system is keeping up with real-time speed
                if time_per_frame > 0 and time_taken > time_per_frame:
                    print(f"Warning: System is NOT keeping up with real-time video speed on frame {frame_count}. Inference took {time_taken:.4f}s, expected {time_per_frame:.4f}s.")
                
                # Post-process and get detections
                boxes, scores, class_ids = postprocess_yolo_output(output)
                
                detections = []
                if len(boxes) > 0:
                    for i in range(len(class_ids)):
                        box_int = [int(coord) for coord in boxes[i]]
                        detections.append((labels[class_ids[i]], scores[i], box_int))
                
                # Draw detections and write to output video
                annotated_frame = draw_detections_on_frame(frame, detections, labels)
                
                # Add timing information to the frame
                inference_text = f"Inference: {time_taken:.4f}s"
                cv2.putText(annotated_frame, inference_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                out.write(annotated_frame)
            
            # Display the resulting frame (optional, for real-time visualization)
            cv2.imshow('Helmet Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output video saved to: {output_filepath}")
        
        # Print average time taken
        if frame_count > 0:
            avg_time_per_frame = total_time_taken / frame_count
            print(f"Total frames processed: {frame_count}")
            print(f"Total time for inference: {total_time_taken:.4f} seconds")
            print(f"Average time to detect one frame: {avg_time_per_frame:.4f} seconds")
            if avg_time_per_frame > 0:
                print(f"Estimated Frames Per Second (FPS): {1/avg_time_per_frame:.2f}")

# Test with a specific image and pre-converted model
if __name__ == "__main__":
    # --- Configuration ---
    image_to_process = r"C:\Users\devcloud\SemiX\yolo\images\BikesHelmets85_png.rf.e7f00d719d549719bfbdb12d6b373433.jpg"
    # To process a video, change this path to your video file.
    video_to_process = r"C:\Users\devcloud\SemiX\yolo\videos\traffic_video.mp4"
    
    openvino_model_dir = r"C:\Users\devcloud\SemiX\yolov5\openvino_model"
    openvino_xml_path = os.path.join(openvino_model_dir, "best_model.xml")
    
    device_to_use = "AUTO" 
    output_directory = "detected_media"
    
    # --- Run on image ---
    print("--- Running detection on a single image ---")
    if os.path.exists(image_to_process):
        run_detection_on_input(image_to_process, openvino_xml_path, device=device_to_use, output_dir=output_directory)
    else:
        print(f"Image file not found: {image_to_process}. Skipping image detection.")

    # --- Run on video ---
    print("\n\n--- Running detection on a video file ---")
    if os.path.exists(video_to_process):
        run_detection_on_input(video_to_process, openvino_xml_path, device=device_to_use, output_dir=output_directory)
    else:
        print(f"Video file not found: {video_to_process}. Skipping video detection.")
        print("Please provide a valid path to a video file to test this functionality.")

