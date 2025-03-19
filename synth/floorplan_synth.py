from PIL import Image
import numpy as np
import torch as t
import utils

num_category = utils.num_category

class LoadFloorplanSynth():
    """
    Loading a floorplan for synth
    """ 
    def __init__(self, floorplan_path, mask_size=9):
        "Read data from Image"
        with Image.open(floorplan_path) as temp:
            # Convert to numpy array and make it writable
            floorplan = np.array(temp, dtype=np.uint8).copy()
            
        self.input_map = floorplan.copy()
        # Ensure arrays are writable before converting to tensor
        boundary_array = floorplan[:,:,0].copy()
        inside_array = floorplan[:,:,3].copy()
        
        self.boundary = t.from_numpy(boundary_array)
        self.inside = t.from_numpy(inside_array)
        self.data_size = self.inside.shape[0]
        self.mask_size = mask_size
        
        "inside_mask"
        self.inside_mask = t.zeros((self.data_size, self.data_size))
        self.inside_mask[self.inside != 0] = 1.0    
        
        "boundary_mask" 
        self.boundary_mask = t.zeros((self.data_size, self.data_size))      
        self.boundary_mask[self.boundary == 127] = 1.0 
        self.boundary_mask[self.boundary == 255] = 0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

        "front_door_mask"
        self.front_door_mask = t.zeros((self.data_size, self.data_size))
        self.front_door_mask[self.boundary == 255] = 1.0

        "category_mask"
        self.category_mask = t.zeros((utils.num_category, self.data_size, self.data_size))

        "room_node"  
        self.room_node = []

        "existing_category"  
        self.existing_category = t.zeros(utils.num_category)
 
    # [Rest of the methods remain exactly the same]
    def get_composite_living(self, num_extra_channels=0):
        composite = t.zeros((num_extra_channels+3, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask     
        return composite

    def get_composite_continue(self, num_extra_channels=0):
        composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask 
        composite[3] = self.category_mask.sum(0)
        for i in range(utils.num_category):
            composite[i+4] = self.category_mask[i]
        return composite

    def get_composite_location(self, num_extra_channels=0):
        composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask  
        composite[3] = self.category_mask.sum(0)
        for i in range(utils.num_category):
            composite[i+4] = self.category_mask[i]
        return composite

    def get_composite_wall(self, num_extra_channels=0):
        composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask
        composite[3] = self.category_mask.sum(0)
        for i in range(utils.num_category):
            composite[i+4] = self.category_mask[i]
        return composite

    def add_room(self, node):
        index = utils.label2index(node['category']) 
        h, w = node['centroid']
        min_h = max(h - self.mask_size, 0)
        max_h = min(h + self.mask_size, self.data_size - 1)
        min_w = max(w - self.mask_size, 0)
        max_w = min(w + self.mask_size, self.data_size - 1)
        self.category_mask[index, min_h:max_h+1, min_w:max_w+1] = 1.0
        self.room_node.append(node)
        self.existing_category[index] += 1
# from PIL import Image
# import numpy as np
# import torch as t
# import utils

# num_category = utils.num_category

# class LoadFloorplanSynth():
#     """
#     Loading a floorplan for synth
#     """ 
#     def __init__(self, floorplan_path, mask_size=9):
#         "Read data from Image"
#         try:
#             with Image.open(floorplan_path) as temp:
#                 # Check if image is in RGB or RGBA format
#                 floorplan = np.array(temp, dtype=np.uint8).copy()
                
#                 # Preprocess image if needed
#                 if len(floorplan.shape) == 2:  # Grayscale image
#                     # Convert to 4-channel format with appropriate defaults
#                     new_floorplan = np.zeros((floorplan.shape[0], floorplan.shape[1], 4), dtype=np.uint8)
#                     new_floorplan[:,:,0] = floorplan  # Use grayscale as boundary channel
#                     new_floorplan[:,:,3] = (floorplan > 0).astype(np.uint8) * 255  # Set non-zero pixels as inside
#                     floorplan = new_floorplan
#                 elif len(floorplan.shape) == 3 and floorplan.shape[2] == 3:  # RGB image
#                     # Convert RGB to 4-channel format
#                     new_floorplan = np.zeros((floorplan.shape[0], floorplan.shape[1], 4), dtype=np.uint8)
#                     new_floorplan[:,:,0] = floorplan[:,:,0]  # Use red channel as boundary
#                     new_floorplan[:,:,3] = (floorplan[:,:,0] > 0).astype(np.uint8) * 255  # Set non-zero pixels as inside
#                     floorplan = new_floorplan
                
#                 # Resize to expected dimensions if needed (256x256)
#                 if floorplan.shape[0] != 256 or floorplan.shape[1] != 256:
#                     # Create a resized image with proper dimensions
#                     pil_img = Image.fromarray(floorplan)
#                     pil_img = pil_img.resize((256, 256), Image.LANCZOS)
#                     floorplan = np.array(pil_img, dtype=np.uint8)
                
#                 self.input_map = floorplan.copy()
#                 # Ensure arrays are writable before converting to tensor
#                 boundary_array = floorplan[:,:,0].copy()
#                 inside_array = floorplan[:,:,3].copy()
                
#                 self.boundary = t.from_numpy(boundary_array)
#                 self.inside = t.from_numpy(inside_array)
#                 self.data_size = self.inside.shape[0]
#                 self.mask_size = mask_size
                
#                 "inside_mask"
#                 self.inside_mask = t.zeros((self.data_size, self.data_size))
#                 self.inside_mask[self.inside != 0] = 1.0    
                
#                 "boundary_mask" 
#                 self.boundary_mask = t.zeros((self.data_size, self.data_size))      
#                 self.boundary_mask[self.boundary == 127] = 1.0 
#                 self.boundary_mask[self.boundary == 255] = 0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

#                 "front_door_mask"
#                 self.front_door_mask = t.zeros((self.data_size, self.data_size))
#                 self.front_door_mask[self.boundary == 255] = 1.0

#                 "category_mask"
#                 self.category_mask = t.zeros((utils.num_category, self.data_size, self.data_size))

#                 "room_node"  
#                 self.room_node = []

#                 "existing_category"  
#                 self.existing_category = t.zeros(utils.num_category)
                
#         except Exception as e:
#             print(f"Error loading floorplan: {e}")
#             raise ValueError(f"Failed to process the input image. Please check the image format: {e}")
 
#     # [Rest of the methods remain exactly the same]
#     def get_composite_living(self, num_extra_channels=0):
#         composite = t.zeros((num_extra_channels+3, self.data_size, self.data_size))
#         composite[0] = self.inside_mask
#         composite[1] = self.boundary_mask
#         composite[2] = self.front_door_mask     
#         return composite

#     def get_composite_continue(self, num_extra_channels=0):
#         composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
#         composite[0] = self.inside_mask
#         composite[1] = self.boundary_mask
#         composite[2] = self.front_door_mask 
#         composite[3] = self.category_mask.sum(0)
#         for i in range(utils.num_category):
#             composite[i+4] = self.category_mask[i]
#         return composite

#     def get_composite_location(self, num_extra_channels=0):
#         composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
#         composite[0] = self.inside_mask
#         composite[1] = self.boundary_mask
#         composite[2] = self.front_door_mask  
#         composite[3] = self.category_mask.sum(0)
#         for i in range(utils.num_category):
#             composite[i+4] = self.category_mask[i]
#         return composite

#     def get_composite_wall(self, num_extra_channels=0):
#         composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
#         composite[0] = self.inside_mask
#         composite[1] = self.boundary_mask
#         composite[2] = self.front_door_mask
#         composite[3] = self.category_mask.sum(0)
#         for i in range(utils.num_category):
#             composite[i+4] = self.category_mask[i]
#         return composite

#     def add_room(self, node):
#         index = utils.label2index(node['category']) 
#         h, w = node['centroid']
#         min_h = max(h - self.mask_size, 0)
#         max_h = min(h + self.mask_size, self.data_size - 1)
#         min_w = max(w - self.mask_size, 0)
#         max_w = min(w + self.mask_size, self.data_size - 1)
#         self.category_mask[index, min_h:max_h+1, min_w:max_w+1] = 1.0
#         self.room_node.append(node)
#         self.existing_category[index] += 1