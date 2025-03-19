# from PIL import Image
# import numpy as np
# import os
# import shutil
# import argparse
# import sys

# def process_input_image(input_path, output_path):
#     """
#     Process an iOS image to make it compatible with the floorplan system.
    
#     Args:
#         input_path: Path to input image
#         output_path: Path where the processed image should be saved
    
#     Returns:
#         True if successful, False otherwise
#     """
#     try:
#         print(f"Processing image: {input_path}")
#         with Image.open(input_path) as img:
#             print(f"Original format: {img.format}, Size: {img.size}, Mode: {img.mode}")
            
#             # Convert to numpy
#             img_array = np.array(img, dtype=np.uint8)
#             print(f"Original array shape: {img_array.shape}, dtype: {img_array.dtype}")
            
#             # Convert to 4-channel RGBA format if needed
#             if len(img_array.shape) == 2:  # Grayscale image
#                 # Convert to 4-channel format with appropriate defaults
#                 new_img_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
#                 new_img_array[:,:,0] = img_array  # Use grayscale as boundary channel
#                 new_img_array[:,:,3] = (img_array > 0).astype(np.uint8) * 255  # Set non-zero pixels as inside
#                 img_array = new_img_array
#             elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB image
#                 # Convert RGB to 4-channel format
#                 new_img_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
#                 new_img_array[:,:,0] = img_array[:,:,0]  # Use red channel as boundary
#                 new_img_array[:,:,3] = (img_array[:,:,0] > 0).astype(np.uint8) * 255  # Set non-zero pixels as inside
#                 img_array = new_img_array
            
#             # Resize to expected dimensions (256x256)
#             if img_array.shape[0] != 256 or img_array.shape[1] != 256:
#                 # Create a temporary PIL image for resizing
#                 pil_img = Image.fromarray(img_array)
#                 pil_img = pil_img.resize((256, 256), Image.LANCZOS)
#                 img_array = np.array(pil_img, dtype=np.uint8)
            
#             # Save the processed image
#             output_img = Image.fromarray(img_array)
#             output_img.save(output_path)
            
#             print(f"Processed array shape: {img_array.shape}, dtype: {img_array.dtype}")
#             print(f"Saved processed image to: {output_path}")
            
#             return True
#     except Exception as e:
#         print(f"Error processing image: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def main():
#     parser = argparse.ArgumentParser(description='Process iOS images for floorplan system')
#     parser.add_argument('-i', '--input', required=True, help='Input image path')
#     parser.add_argument('-o', '--output', help='Output image path (default: processed_<input>)')
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.input):
#         print(f"Error: Input file does not exist: {args.input}")
#         return 1
    
#     output_path = args.output
#     if not output_path:
#         input_dir, input_filename = os.path.split(args.input)
#         output_path = os.path.join(input_dir, f"processed_{input_filename}")
    
#     if process_input_image(args.input, output_path):
#         print("Image processing completed successfully")
#         return 0
#     else:
#         print("Image processing failed")
#         return 1

# if __name__ == "__main__":
#     sys.exit(main())