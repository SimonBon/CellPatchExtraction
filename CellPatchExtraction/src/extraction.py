from typing import Union, Dict, Tuple, Any, List
from scipy import ndimage
import numpy as np
import tifffile
from cellpose import models
import os
from . import AVAIL_MODELS, CELLPOSE_PATH, TYPES
import torch

def remove_masks(image, min_size=None, max_size=None):
    """
    Remove masks from the image that are larger than the given max_size.

    :param image: numpy array representing the image with segmentation masks.
    :param max_size: threshold for mask size.
    :return: numpy array with large masks removed.
    """
    if min_size is None:
        min_size = 0
    
    if max_size is None:
        max_size = np.inf
    
    unique_colors = np.unique(image)
    
    for color in unique_colors:
        # Create a mask for the current color
        mask = image == color

        # Calculate the size of the current mask
        mask_size = np.sum(mask)
        # If the mask is larger than the threshold, remove it
        if mask_size > max_size or mask_size < min_size:
            image[mask] = 0  # Assuming you want to set it to black

    return image
    
    

def segment_image(image: Union[str, np.ndarray], 
                  model: Union[str, models.Cellpose, models.CellposeModel], 
                  cellpose_kwargs: Dict[str, Union[int, float]] = {"diameter": 50},
                  nuclear_channel=2,
                  device=None,
                  max_size=None,
                  min_size=None,
                  do_3D=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment an image using Cellpose.
    
    Parameters:
        image: str or np.ndarray - Path to the image or a numpy array representing the image.
        model: str or CellposeModel - Cellpose model name, path or pre-loaded model object.
        cellpose_kwargs: dict - Additional keyword arguments for Cellpose model.
        
    Returns:
        Tuple containing masks and the original image.
    """
    image = tifffile.imread(image) if isinstance(image, str) else image
    if not isinstance(image, np.ndarray):
        raise TypeError("Invalid type for 'image'. Must be either str or np.ndarray.")
    
    if isinstance(model, str):
        if os.path.exists(model):
            model = models.CellposeModel(pretrained_model=model, gpu=torch.cuda.is_available(), device=device)
        elif model in AVAIL_MODELS:
            model = models.CellposeModel(pretrained_model=f"{CELLPOSE_PATH}/{model}", gpu=torch.cuda.is_available(), device=device)
        else:
            raise ValueError(f"Invalid model type or path. Must be in {AVAIL_MODELS} or a valid file path.")
    elif not isinstance(model, (models.Cellpose, models.CellposeModel)):
        raise TypeError("Invalid type for 'model'. Must be either str or cellpose.models.CellposeModel.")
    
    for k, v in cellpose_kwargs.items():
        setattr(model, k, v)
    
    if do_3D:
        masks, _, _ = model.eval(image, do_3D=do_3D, **cellpose_kwargs)
    elif image.ndim > 2:
        masks, _, _ = model.eval(image[..., nuclear_channel], **cellpose_kwargs)
    elif image.ndim == 2:
        masks, _, _ = model.eval(image, **cellpose_kwargs)
    else:
        raise ValueError(f"Invalid shape for image to segment. Dimension {image.shape} not implemented")
    
    if max_size is not None or min_size is not None:
        masks = remove_masks(masks, min_size=min_size, max_size=max_size)
    
    return masks, image
    

def get_coordinates(x: slice, y: slice, image_size: Tuple[int, int], half_size: int) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
    """
    Get the coordinates and padding information for an object in an image.
    
    Parameters:
        x: slice - X-axis slice object.
        y: slice - Y-axis slice object.
        image_size: tuple - Shape of the image as (height, width).
        half_size: int - Half the size of the object.
        
    Returns:
        Tuple containing center coordinates, max and min coordinates for x and y,
        and padding information (left, right, top, bottom).
    """
    
    # Calculate the object center
    center_x = (x.start + x.stop) // 2
    center_y = (y.start + y.stop) // 2
    
    # Calculate the object dimensions constrained by image size
    x_min = max(0, center_x - half_size)
    x_max = min(image_size[1] - 1, center_x + half_size)
    
    y_min = max(0, center_y - half_size)
    y_max = min(image_size[0] - 1, center_y + half_size)
    
    # Calculate padding if the object is near the edge of the image
    pad_left  = abs(min(0, center_x - half_size))
    pad_right = abs(min(0, image_size[1] - 1 - (center_x + half_size)))
    
    pad_top    = abs(min(0, center_y - half_size))
    pad_bottom = abs(min(0, image_size[0] - 1 - (center_y + half_size)))
    
    return center_x, center_y, x_max, x_min, y_max, y_min, pad_left, pad_right, pad_top, pad_bottom


def extract_and_pad_objects(mask: np.ndarray, 
                            image: np.ndarray, 
                            patch_size: int, 
                            exclude_edges: bool = True, 
                            use_surrounding: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extracts and pads objects based on a mask and original image.
    
    Parameters:
        mask: np.ndarray - Labeled mask image.
        image: np.ndarray - Original image.
        patch_size: int - Size of the square patch.
        exclude_edges: bool - Whether to exclude patches touching the edge.
        use_surrounding: bool - Whether to include surrounding in the patch.
        
    Returns:
        A tuple containing lists of image patches, cell patches, surrounding patches, background patches, and coordinates.
    """
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("Mask and Image do not match shapes")
    
    if patch_size % 2 != 0:
        raise ValueError("Patch size must be even")
    
    half_size = patch_size // 2
    image_shape = mask.shape
    
    image_patches = []
    mask_patches = []
    surrounding_patches = []
    background_patches = []
    coords = []
    
    objects = ndimage.find_objects(mask)
    for i, obj in enumerate(objects):
        if obj is None:
            continue

        label = i + 1
        y_slice, x_slice = obj
        
        center_x, center_y, x_max, x_min, y_max, y_min, pad_left, pad_right, pad_top, pad_bottom = get_coordinates(x_slice, y_slice, image_shape, half_size)
        
        if exclude_edges and sum([pad_left, pad_right, pad_top, pad_bottom]) > 0:
            continue

        def pad_patch(patch: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
            return np.pad(patch, padding, mode='constant', constant_values=0)
        
        mask_patch = pad_patch(mask[y_min:y_max, x_min:x_max], ((pad_top, pad_bottom), (pad_left, pad_right)))
        if image.ndim == 3:
            image_patch = pad_patch(image[y_min:y_max, x_min:x_max], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        elif image.ndim == 2:
            image_patch = pad_patch(image[y_min:y_max, x_min:x_max], ((pad_top, pad_bottom), (pad_left, pad_right)))
        else:
            raise ValueError(f"Image of shape {image.shape} not implemented")

        
        cell_mask = (mask_patch == label).astype(int)
        surrounding_mask = np.logical_and(mask_patch != label, mask_patch != 0).astype(int)
        background_mask = (mask_patch == 0).astype(int)
        
        if not use_surrounding:
            image_patch[cell_mask != 1] = 0

        image_patches.append(image_patch)
        mask_patches.append(cell_mask)
        surrounding_patches.append(surrounding_mask)
        background_patches.append(background_mask)
        coords.append((center_x, center_y))
        
    return image_patches, mask_patches, surrounding_patches, background_patches, coords


def extract_patches(image: Union[str, np.ndarray], 
                    model: Union[str, models.Cellpose, models.CellposeModel], 
                    patch_size: int = 50, 
                    cellpose_kwargs: Dict[str, Union[int, float]] = {"diameter": 50},
                    max_size=None,
                    min_size=None,
                    do_3D=False,
                    return_all: bool = False,
                    nuclear_channel=2,
                    return_segmentation=False,
                    device=None,
                    exclude_edges=True,
                    use_surrounding=False) -> Any:
    """
    Extract single nucleus patches from an image using Cellpose model and custom patch extraction logic.
    
    Parameters:
        image: Union[str, np.ndarray] - Either a path to an image file or a numpy array representing the image.
        model: Union[str, models.Cellpose, models.CellposeModel] - Either a path to a pre-trained model, or a Cellpose model instance.
        patch_size: int - Size of the extracted square patch.
        cellpose_kwargs: Dict[str, Union[int, float]] - Additional keyword arguments to be passed to the Cellpose model for segmentation.
        return_all: bool - Whether to return all types of patches and coordinates or just image_patches.
        
    Returns:
        If return_all is True, a tuple containing lists of image_patches, mask_patches, surrounding_patches, background_patches, and coords.
        If return_all is False, a list of numpy arrays, each representing an extracted nucleus patch.
    """
    
    image = image/image.max()
    
    # Segment the image to get nucleus masks and the original image
    masks, original_image = segment_image(image, model, cellpose_kwargs=cellpose_kwargs, nuclear_channel=nuclear_channel, device=device, max_size=max_size, min_size=min_size, do_3D=False)
    
    # Extract and pad objects (i.e., nucleus patches) based on the masks and original image
    image_patches, mask_patches, surrounding_patches, background_patches, coords = extract_and_pad_objects(masks, original_image, patch_size, exclude_edges=exclude_edges, use_surrounding=use_surrounding)
    
    if return_all:
        ret_val = {"image_patches": image_patches, "mask_patches": mask_patches, "surrounding_mask_patches": surrounding_patches, "background_mask_patches": background_patches, "coordinates": coords, "segmentation": masks}
    else:
        ret_val = image_patches
        
    if return_segmentation:
        
        return ret_val, masks
    
    return ret_val