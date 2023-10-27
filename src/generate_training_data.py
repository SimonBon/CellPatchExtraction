from cellpose.models import CellposeModel
from src.utils import get_files, extract_and_pad_objects, segment_image
import argparse
import pathlib
import os
import torch
from cellpose import utils
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


cellpose_path = f"{os.path.dirname(__file__)}/.cellpose_model"
AVAIL_MODELS = os.listdir(cellpose_path)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, required=True)
    args.add_argument("--diameter", type=float, default=45)
    args.add_argument("--model_type", type=str, default="CP_BM")
    args.add_argument("--use_surrounding", type=bool, default=False, action=argparse.BooleanOptionalAction)
    args.add_argument("--min_size", type=int, default=2000)
    args.add_argument("--patch_sz", type=int, default=128)
    args.add_argument("--sample_type", default="None", help="please identify if the sample id 'positive' or 'negative'")
    args = args.parse_args()
    
    assert args.model_type in AVAIL_MODELS
    cellpose_model = CellposeModel(pretrained_model=f"{cellpose_path}/{args.model_type}", gpu=True if torch.cuda.is_available() else False)
    cellpose_model.diameter = args.diameter
    cellpose_model.min_size = args.min_size
    
    path = pathlib.Path(args.path)
    
    files = get_files(path)
    
    A_image_patches, A_cell_patches, A_surrounding_patches, A_background_patches, image_coordinates, images, masks = [], [], [], [], [], [], []
    for num, file in enumerate(files): 
        
        mask, image = segment_image(file, cellpose_model)
        image_patches, cell_patches, surrounding_patches, background_patches, coords = extract_and_pad_objects(mask, image, args.patch_sz, exclude_edges=True, use_surrounding=args.use_surrounding)
        A_image_patches.extend(image_patches)
        A_cell_patches.extend(cell_patches)
        A_surrounding_patches.extend(surrounding_patches)
        A_background_patches.extend(background_patches)
        image_coordinates.extend([[num, coord[0], coord[1]] for coord in coords])
        images.append(image)
        masks.append(mask)
        
    
    with h5py.File(f"{args.path}/SingleCellPatches.h5", "w") as fout:
        
        fout.create_dataset("sample_type", data=args.sample_type)
        fout.create_dataset("image_patches", data=A_image_patches)
        fout.create_dataset("images", data=images)
        fout.create_dataset("masks", data=masks)
        fout.create_dataset("image_coordinates", data=np.array(image_coordinates))
        
        if args.use_surrounding:
            fout.create_dataset("cell_patches", data=A_cell_patches)
            fout.create_dataset("surrounding_patches", data=A_surrounding_patches)
            fout.create_dataset("background_patches", data=A_background_patches)
            
