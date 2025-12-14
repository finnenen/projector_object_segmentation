from ultralytics import YOLO, SAM, YOLOWorld
import cv2
import numpy as np
import ast

import os

def segment_and_mask(image_path, output_dir, classes_to_segment=None):
    """
    Segments objects in an image and creates individual mask files for each object.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output mask images.
        classes_to_segment (list): List of class names.
        
    Returns:
        list: List of dicts [{'path': 'filename.png', 'label': 'class_name'}]
    """
    # 1. Load YOLO-World for Detection
    print("Loading YOLO-World model...")
    detector = YOLOWorld('yolov8s-world.pt')
    
    # 2. Define custom classes
    if classes_to_segment:
        print(f"Setting custom classes: {classes_to_segment}")
        detector.set_classes(classes_to_segment)
    else:
        print("No classes specified. Detecting all COCO classes by default.")

    # 3. Run Detection
    print(f"Detecting objects in {image_path}...")
    detection_results = detector.predict(image_path)
    det_result = detection_results[0]
    
    # Get bounding boxes
    boxes = det_result.boxes.xyxy
    
    if len(boxes) == 0:
        print("No objects found matching the criteria.")
        return []

    print(f"Found {len(boxes)} objects. Generating masks with SAM...")

    # 4. Load SAM
    segmenter = SAM('mobile_sam.pt') 
    
    # 5. Segment
    sam_results = segmenter(image_path, bboxes=boxes, verbose=False)
    sam_result = sam_results[0]

    # Get original image dimensions
    h, w = sam_result.orig_shape
    
    generated_masks = []
    
    if sam_result.masks is not None:
        # Get class names map and indices
        names = det_result.names
        class_indices = det_result.boxes.cls.cpu().numpy().astype(int)
        
        # Get masks as bitmaps
        masks_data = sam_result.masks.data.cpu().numpy()
        
        # Get original boxes to calculate centers for UI scaling
        # boxes is a tensor or numpy array of [x1, y1, x2, y2]
        boxes_data = boxes.cpu().numpy()

        for i, mask_array in enumerate(masks_data):
            # Resize if necessary to match original image
            if mask_array.shape != (h, w):
                mask_array = cv2.resize(mask_array, (w, h))
            
            # Create RGBA image for this single mask
            single_mask_img = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Fill the mask area with white (255, 255, 255, 255)
            single_mask_img[mask_array > 0] = [255, 255, 255, 255]
            
            class_idx = class_indices[i]
            label = names[class_idx]
            
            # Get bbox for this object
            box = boxes_data[i] # [x1, y1, x2, y2]
            bbox = {
                'x': float(box[0]),
                'y': float(box[1]),
                'w': float(box[2] - box[0]),
                'h': float(box[3] - box[1])
            }

            # Sanitize label for filename
            safe_label = "".join([c for c in label if c.isalnum() or c in (' ', '-', '_')]).strip()
            filename = f"mask_{i}_{safe_label}.png"
            full_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(full_path, single_mask_img)
            generated_masks.append({'path': filename, 'label': label, 'bbox': bbox})
            
        print(f"Generated {len(generated_masks)} individual masks.")
        return generated_masks
    else:
        print("SAM could not generate masks for the detected boxes.")
        return []

def apply_mask_to_image(image_path, mask_path, output_path):
    """
    Applies the mask to the original image, making the background transparent.
    """
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        print("Error loading image or mask")
        return False

    # Ensure mask is same size as image
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Create RGBA image (add alpha channel)
    b, g, r = cv2.split(img)
    # The mask is white (255) for objects, black (0) for background.
    # We use the mask directly as the alpha channel.
    rgba = [b, g, r, mask]
    dst = cv2.merge(rgba, 4)
    
    cv2.imwrite(output_path, dst)
    print(f"Cropped image saved to {output_path}")
    return True

if __name__ == "__main__":
    # Ask user for target classes
    print("Bitte geben Sie die Zielklassen ein (z.B. ['couch', 'laptop'] oder einfach: couch, laptop)")
    user_input = input("Target classes: ")
    
    target_classes = []
    if user_input.strip():
        try:
            # Try to parse as a python list (e.g. ["a", "b"])
            parsed = ast.literal_eval(user_input)
            if isinstance(parsed, list):
                target_classes = parsed
            elif isinstance(parsed, str):
                target_classes = [parsed]
            elif isinstance(parsed, tuple):
                target_classes = list(parsed)
            else:
                # Fallback
                target_classes = [str(parsed)]
        except (ValueError, SyntaxError):
            # Fallback: split by comma
            target_classes = [c.strip() for c in user_input.split(',') if c.strip()]
    
    if not target_classes:
        print("Keine Klassen angegeben. Es werden alle COCO-Klassen erkannt (Standard).")
        target_classes = None
    else:
        print(f"Segmentiere folgende Klassen: {target_classes}")

    segment_and_mask(
        image_path='Beispiel-Bild.jpg', 
        output_path='mask_output.jpg', 
        classes_to_segment=target_classes
    )
