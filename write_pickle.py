from PIL import Image
import numpy as np
import shutil
import pickle
import utils
import os

def write2pickle(train_dir, pkl_dir):
    # Filter out non-image files and ensure only valid image formats are included
    train_data_path = [os.path.join(train_dir, path) for path in os.listdir(train_dir) if path.endswith(('.png', '.jpg', '.jpeg'))]
    print(f'Number of dataset: {len(train_data_path)}')

    for path in train_data_path:
        try:
            # Open the image file
            with Image.open(path) as temp:
                image_array = np.asarray(temp, dtype=np.uint8)

            # Extract different layers from the image array
            boundary_mask = image_array[:,:,0]
            category_mask = image_array[:,:,1]
            index_mask = image_array[:,:,2]
            inside_mask = image_array[:,:,3]
            shape_array = image_array.shape
            index_category = []
            room_node = []

            # Masks for interior walls and doors
            interiorWall_mask = np.zeros(category_mask.shape, dtype=np.uint8)
            interiorWall_mask[category_mask == 16] = 1        
            interiordoor_mask = np.zeros(category_mask.shape, dtype=np.uint8)
            interiordoor_mask[category_mask == 17] = 1

            # Loop through the image to extract room nodes
            for h in range(shape_array[0]):  
                for w in range(shape_array[1]):
                    index = index_mask[h, w]
                    category = category_mask[h, w]
                    if index > 0 and category <= 12:
                        if len(index_category):
                            flag = True
                            for i in index_category:
                                if i[0] == index:
                                    flag = False
                            if flag:
                                index_category.append((index, category))
                        else:
                            index_category.append((index, category))

            # Create nodes based on room categories and centroids
            for (index, category) in index_category:
                node = {}
                node['category'] = int(category)
                mask = np.zeros(index_mask.shape, dtype=np.uint8)
                mask[index_mask == index] = 1
                node['centroid'] = utils.compute_centroid(mask)
                room_node.append(node)

            # Save the result into a pickle file
            pkl_path = path.replace(train_dir, pkl_dir)
            pkl_path = pkl_path.replace('png', 'pkl')
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump([inside_mask, boundary_mask, interiorWall_mask, interiordoor_mask, room_node], 
                            pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        except (Image.UnidentifiedImageError, OSError) as e:
            # If the file is not a valid image, skip it
            print(f"Skipped file {path} due to an error: {e}")

if __name__=='__main__':
    print("*******************************************")
    
    # Define directories for training and validation datasets
    train_dataset_dir = f"dataset/train"
    val_dataset_dir = f"dataset/val"
    
    # Define directories where pickle files will be stored
    train_pickle_dir = f"pickle/train"
    val_pickle_dir = f"pickle/val"

    # Create directories for pickle files if they don't exist
    if os.path.exists(train_pickle_dir):
        shutil.rmtree(train_pickle_dir)
    os.mkdir(train_pickle_dir)
    
    if os.path.exists(val_pickle_dir):
        shutil.rmtree(val_pickle_dir)
    os.mkdir(val_pickle_dir)

    # Write the training and validation datasets into pickle files
    write2pickle(train_dataset_dir, train_pickle_dir)
    write2pickle(val_dataset_dir, val_pickle_dir)
