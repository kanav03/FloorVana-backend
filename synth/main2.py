# import time

# import numpy as np
# from flask import Flask, request, jsonify
# import os
# from synth import Synth
# import shutil
# from floorplan_synth import LoadFloorplanSynth
# from vectorization import WallAbstract
# from PIL import Image


# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# OUTPUT_FOLDER = 'synth_output'
# MODEL_DIR = 'trained_model'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # Create necessary directories
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/generate_floorplan', methods=['POST'])
# def generate_floorplan():
#     # Check if image file is present in request
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400

#     try:
#         # Save uploaded file
#         filename = file.filename
#         input_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(input_path)

#         # Prepare output directory
#         if os.path.exists(OUTPUT_FOLDER):
#             shutil.rmtree(OUTPUT_FOLDER)
#         os.makedirs(OUTPUT_FOLDER)

#         # Initialize synthesizer
#         synthesizer = Synth(
#             model_dir=MODEL_DIR,
#             living_epoch=150, from_multi_GPU_living=False,
#             continue_epoch=70, from_multi_GPU_continue=False,
#             location_epoch=32, from_multi_GPU_location=False,
#             wall_epoch=50, from_multi_GPU_wall=False
#         )

#         # Process the image
#         synthesizer.floorplan = LoadFloorplanSynth(input_path)
#         synthesizer.map = synthesizer.floorplan.input_map
#         synthesizer.map.flags.writeable = True

#         # Generate floorplan
#         synthesizer.add_living()
#         synthesizer.continue_adding = True
#         while synthesizer.continue_adding:
#             synthesizer.add_room()
#             synthesizer.should_continue()
#         synthesizer.add_wall()

#         # Save result
#         output_path = os.path.join(OUTPUT_FOLDER, filename)
#         print(output_path)
#         # synthesizer.save_synth_result(OUTPUT_FOLDER, filename)

#         output_dir = 'synth_normalization'
#         # if os.path.exists(output_dir):
#         #     shutil.rmtree(output_dir)
#         # os.mkdir(output_dir)

#         test_number = 0
#         start_time = time.perf_counter()
#         temp_time = start_time

#         # output_path = 'synth_normalization'

#         test_number = test_number + 1
#         floorplan = Image.open(f'{filename}')
#         input_map = np.asarray(floorplan, dtype=np.uint8)

#         output_map = np.zeros(input_map.shape, dtype=np.uint8)
#         output_map[:,:,0] = input_map[:,:,0]
#         output_map[:,:,2] = input_map[:,:,2]
#         output_map[:,:,3] = input_map[:,:,3]
#         abstracter = WallAbstract(input_map) 
#         abstracter.exterior_boundary_abstract()
#         abstracter.interior_wall_abstract() 

#         for wall in abstracter.interior_walls:
#             for h in range(wall[0], wall[1]+1):
#                 for w in range(wall[2], wall[3]+1):
#                     output_map[h, w, 1] = 127
#         output = Image.fromarray(np.uint8(output_map))
#         output.save(f'{output_dir}/{filename}')   

#         end_time = time.perf_counter()
#         print(f'{filename}: {(end_time-temp_time):.2f}s')
#         temp_time = end_time  
            
#         end_time = time.perf_counter()
#         cost_time = end_time-start_time
#         print(f'Total test time: {cost_time:.2f}s')
#         print(f'Total test number: {test_number}')
#         print(f'Average time: {(cost_time/test_number):.2f}s')    

    
        

#         return jsonify({
#             'message': 'Floorplan generated successfully',
#             'output_path': f'synth_normalization/{filename}'
#         }), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)