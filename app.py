from flask import Flask, render_template, request, jsonify
import os
from segment_objects import segment_and_mask, apply_mask_to_image

app = Flask(__name__)

# Configuration
IMAGE_FOLDER = 'static'
DEFAULT_IMAGE = 'Beispiel-Bild.jpg'
MASK_IMAGE = 'mask_output.png' # PNG for transparency support if needed later
CROP_IMAGE = 'cropped_output.png'

@app.route('/')
def index():
    return render_template('index.html', original_image=DEFAULT_IMAGE)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
        
    if file:
        # Save as a standard name or keep original? 
        # Let's keep it simple and overwrite a "current" image or use a unique name.
        # For this session, let's just save it as 'current_image.jpg' to avoid clutter, 
        # or better, just save it and return the name.
        filename = 'current_image.jpg'
        file.save(os.path.join(IMAGE_FOLDER, filename))
        return jsonify({'success': True, 'filename': filename})

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    classes_input = data.get('classes', '')
    filename = data.get('filename', DEFAULT_IMAGE)
    
    # Parse classes
    target_classes = [c.strip() for c in classes_input.split(',') if c.strip()]
    if not target_classes:
        target_classes = None # Detect all
    
    input_path = os.path.join(IMAGE_FOLDER, filename)
    
    print(f"Processing {filename} with classes: {target_classes}")
    
    # segment_and_mask now returns a list of dicts [{'path': '...', 'label': '...'}]
    masks = segment_and_mask(input_path, IMAGE_FOLDER, target_classes)
    
    if masks:
        import time
        ts = str(time.time())
        # Add full URL path
        for m in masks:
            m['url'] = m['path'] + '?t=' + ts
            
        return jsonify({'success': True, 'masks': masks})
    else:
        return jsonify({'success': False, 'error': 'No objects found or error in processing'})

@app.route('/crop', methods=['POST'])
def crop():
    input_path = os.path.join(IMAGE_FOLDER, ORIGINAL_IMAGE)
    mask_path = os.path.join(IMAGE_FOLDER, MASK_IMAGE)
    output_path = os.path.join(IMAGE_FOLDER, CROP_IMAGE)
    
    if not os.path.exists(mask_path):
        return jsonify({'success': False, 'error': 'Mask not found. Generate mask first.'})
        
    success = apply_mask_to_image(input_path, mask_path, output_path)
    
    if success:
        return jsonify({'success': True, 'crop_url': CROP_IMAGE + '?t=' + str(os.path.getmtime(output_path))})
    else:
        return jsonify({'success': False, 'error': 'Error creating cropped image'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
