from pathlib import Path
from . import TYPES
import numpy as np
import re
import warnings
import tifffile
from tqdm import tqdm
import random
from time import perf_counter
import logging

# Define the channel order explicitly
CHANNEL_ORDER = {'R': 0, 'G': 1, 'B': 2}

def get_files(path):
    files_grabbed = []
    for type in TYPES:
        files_grabbed.extend(path.glob(type))
    return files_grabbed

def combine_to_RGB(folder_path, num_images=np.inf, shuffle_images=False, return_identifiers=False, normalize=False):
    
    folder_path = Path(folder_path)
    files = get_files(folder_path)
    unique_file_identifiers = np.unique([f.stem[:-1] for f in files])
    if shuffle_images:
        random.shuffle(unique_file_identifiers)

    rgb_images = []
    unique_file_identifiers = unique_file_identifiers if np.isinf(num_images) else unique_file_identifiers[:num_images]
    
    for file_identifier in tqdm(unique_file_identifiers):
        matching_files = [f for f in files if re.findall(f"{file_identifier}.{{{1}}}\.", str(f))]
        
        if len(matching_files) != 3:
            warnings.warn(f"Skipping {file_identifier}, there was an error in {file_identifier} did not find R, G, B files but found: {matching_files}", UserWarning)
            continue
        
        channels = []
        for rgb_file in sorted(matching_files)[::-1]:  # Sorting ensures R, G, B order if filenames are properly named
            image = tifffile.imread(rgb_file).astype(float)
            if normalize:
                image = np.clip(image /np.percentile(image.flatten(), 99.99), 0, 1)
            channels.append(image)

        rgb_image = np.stack(channels, axis=-1)  # Reverse the channels list to get RGB order
        rgb_images.append(rgb_image)

        if len(rgb_images) > num_images:
            break

    if return_identifiers:
        return rgb_images, unique_file_identifiers
    
    return rgb_images  # Convert the list of images to a NumPy array1

def get_random_crop(image, crop_size=(128, 128)):
    
    if image.ndim == 3 and len(crop_size) == 2:
        assert not np.any([image.shape[i] < crop_size[i] for i in range(image.ndim-1)]), "Not able to cropy since the cropy size id too big for the image"
        
    elif image.ndim == 2 and len(crop_size) == 2:
        assert not np.any([image.shape[i] < crop_size[i] for i in range(image.ndim)]), "Not able to cropy since the cropy size id too big for the image"
        
    else:
        raise ValueError("weird combination of image and crop sizes")
    
    max_y_idx = image.shape[0] - crop_size[0]
    max_x_idx = image.shape[1] - crop_size[1]
    
    x_idx = np.random.choice(range(max_x_idx), 1)[0]
    y_idx = np.random.choice(range(max_y_idx), 1)[0]
    
    return image[y_idx:y_idx+crop_size[0], x_idx:x_idx+crop_size[1]]
    
    
        
    