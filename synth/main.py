from flask import Flask, request, jsonify
import os
from synth import Synth
import shutil
from floorplan_synth import LoadFloorplanSynth
from vectorization import WallAbstract
from PIL import Image, ImageDraw
import numpy as np
import time
import base64
import subprocess
from pathlib import Path
from gunicorn.app.base import BaseApplication
from gunicorn.workers.sync import SyncWorker

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'synth_output'
NORMALIZED_FOLDER = 'synth_normalization'
MODEL_DIR = 'trained_model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SYNTHESIS_WAIT_TIME = 2  # seconds to wait after synthesis
VISUALIZATION_FOLDER = 'visualization_output'
DEEPLAYOUT_PATH = '../visualization/Deeplayout/build/Deeplayout'
TEXTURE_SOURCE = '../visualization/Deeplayout/texture'
TEXTURE_DEST = '../visualization/Deeplayout/build/texture'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(NORMALIZED_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_directory_exists(directory):
    """Ensure directory exists and is empty"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def wait_for_file(file_path, timeout=30):
    """Wait for a file to exist and be accessible"""
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for file: {file_path}")
        time.sleep(0.5)
    
    # Additional wait to ensure file is fully written
    time.sleep(0.5)
    return True

def setup_visualization_directories():
    """Setup visualization and texture directories"""
    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    
    # Create and populate texture directory
    texture_dest = Path(TEXTURE_DEST).resolve()
    texture_source = Path(TEXTURE_SOURCE).resolve()
    
    if texture_source.exists():
        # Ensure texture directory exists in build
        texture_dest.mkdir(parents=True, exist_ok=True)
        
        # Copy all texture files
        for texture_file in texture_source.glob('*.jpg'):
            shutil.copy2(
                str(texture_file), 
                str(texture_dest / texture_file.name)
            )
        print(f"Copied textures to: {texture_dest}")
    else:
        print(f"Warning: Texture source directory not found: {texture_source}")

setup_visualization_directories()

@app.route('/generate_floorplan', methods=['POST'])
def generate_floorplan():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save uploaded file
        filename = file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        print(f"Saved file at {input_path}")

        with open(input_path, 'rb') as f:
            print(base64.b64encode(f.read()).decode('utf-8'))

        # Step 1: Synthesis
        print("Starting synthesis phase...")
        ensure_directory_exists(OUTPUT_FOLDER)
        
        synthesizer = Synth(
            model_dir=MODEL_DIR,
            living_epoch=150, from_multi_GPU_living=False,
            continue_epoch=70, from_multi_GPU_continue=False,
            location_epoch=32, from_multi_GPU_location=False,
            wall_epoch=50, from_multi_GPU_wall=False
        )

        synthesizer.floorplan = LoadFloorplanSynth(input_path)
        synthesizer.map = synthesizer.floorplan.input_map
        synthesizer.map.flags.writeable = True

        synthesizer.add_living()
        synthesizer.continue_adding = True
        while synthesizer.continue_adding:
            synthesizer.add_room()
            synthesizer.should_continue()
        synthesizer.add_wall()

        # Save synthesis result
        synth_output_path = synthesizer.save_synth_result(OUTPUT_FOLDER, filename)
        print(f"Synthesis complete. Output saved at: {synth_output_path}")

        # Wait for synthesis to complete and files to be ready
        print(f"Waiting {SYNTHESIS_WAIT_TIME} seconds for synthesis to stabilize...")
        time.sleep(SYNTHESIS_WAIT_TIME)

        # Verify synthesis output exists
        synthesis_file = os.path.join(OUTPUT_FOLDER, filename)
        if not wait_for_file(synthesis_file):
            raise Exception("Synthesis output file not created")

        # Step 2: Normalization
        print("Starting normalization phase...")
        ensure_directory_exists(NORMALIZED_FOLDER)
        
        floorplans = os.listdir(OUTPUT_FOLDER)
        
        for floorplan_name in floorplans:
            print(f"{OUTPUT_FOLDER}/{floorplan_name}")
            input_floorplan_path = os.path.join(OUTPUT_FOLDER, floorplan_name)
            output_floorplan_path = os.path.join(NORMALIZED_FOLDER, floorplan_name)
            
            # Verify input file exists and is readable
            if not os.path.isfile(input_floorplan_path):
                raise FileNotFoundError(f"Synthesis output not found: {input_floorplan_path}")
            
            floorplan = Image.open(input_floorplan_path)
            input_map = np.asarray(floorplan, dtype=np.uint8)

            output_map = np.zeros(input_map.shape, dtype=np.uint8)
            output_map[:,:,0] = input_map[:,:,0]
            output_map[:,:,2] = input_map[:,:,2]
            output_map[:,:,3] = input_map[:,:,3]
            
            print(input_map)
            abstracter = WallAbstract(input_map) 
            abstracter.exterior_boundary_abstract()
            abstracter.interior_wall_abstract() 

            for wall in abstracter.interior_walls:
                for h in range(wall[0], wall[1]+1):
                    for w in range(wall[2], wall[3]+1):
                        output_map[h, w, 1] = 127
                        
            output = Image.fromarray(np.uint8(output_map))
            output.save(output_floorplan_path)
            print(f"Normalization complete for: {floorplan_name}")

        # Convert the normalized image to base64 for direct embedding in response
        normalized_image_path = os.path.join(NORMALIZED_FOLDER, filename)
        with open(normalized_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Step 3: Visualization
        print("Starting visualization phase...")
        visualization_output = os.path.join(VISUALIZATION_FOLDER, filename)
        
        try:
            # Ensure Deeplayout executable exists
            deeplayout_exec = Path(DEEPLAYOUT_PATH).resolve()
            if not deeplayout_exec.exists():
                raise FileNotFoundError(f"Deeplayout executable not found at: {deeplayout_exec}")

            # Run visualization from the build directory to access textures
            normalized_path = os.path.join(NORMALIZED_FOLDER, filename)
            visualization_output = os.path.join(VISUALIZATION_FOLDER, filename)
            
            result = subprocess.run([
                str(deeplayout_exec),
                '--headless',
                '--text',  # Add text flag
                '-i', str(Path(normalized_path).resolve()),
                '-o', str(Path(visualization_output).resolve())
            ], 
            capture_output=True, 
            text=True,
            cwd=str(deeplayout_exec.parent))  # Set working directory to build folder

            print(f"Running command: {' '.join(result.args)}")  # Debug the exact command
            print(f"Working directory: {result.args[-1]}")
            
            if result.returncode != 0:
                print(f"Visualization stderr: {result.stderr}")
                print(f"Visualization stdout: {result.stdout}")
                raise Exception(f"Visualization failed with return code {result.returncode}")

            print(f"Visualization complete. Output saved at: {visualization_output}")

            # Add visualization image to response
            with open(visualization_output, "rb") as viz_file:
                viz_encoded = base64.b64encode(viz_file.read()).decode('utf-8')

            # Modify the return statement to include visualization
            return jsonify({
                'message': 'Floorplan generated, normalized, and visualized successfully',
                'status': {
                    'synthesis': 'complete',
                    'normalization': 'complete',
                    'visualization': 'complete'
                },
                'paths': {
                    'synthesis': os.path.join(OUTPUT_FOLDER, filename),
                    'normalized': os.path.join(NORMALIZED_FOLDER, filename),
                    'visualization': visualization_output
                },
                'images': {
                    'normalized': encoded_string,
                    'visualization': viz_encoded
                }
            }), 200

        except Exception as viz_error:
            print(f"Visualization error: {str(viz_error)}")
            # Still return success for synthesis and normalization
            return jsonify({
                'message': 'Floorplan generated and normalized successfully (visualization failed)',
                'status': {
                    'synthesis': 'complete',
                    'normalization': 'complete',
                    'visualization': 'failed'
                },
                'paths': {
                    'synthesis': os.path.join(OUTPUT_FOLDER, filename),
                    'normalized': os.path.join(NORMALIZED_FOLDER, filename)
                },
                'images': {
                    'normalized': encoded_string
                },
                'visualization_error': str(viz_error)
            }), 200

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

class GunicornApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        # Set configuration
        for key, value in {
            'bind': f"0.0.0.0:{os.environ.get('PORT', '10000')}",
            'workers': 4,
            'timeout': 900,  # 15 minutes
            'worker_class': 'sync',
            'threads': 1
        }.items():
            self.cfg.set(key, value)

    def load(self):
        return self.application

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    if os.environ.get('GUNICORN_RUN', '0') == '1':

        GunicornApplication(app).run()
    else:
        # Development server
        print(f"Starting development server on port {port}")
        app.run(debug=False, host='0.0.0.0', port=port)