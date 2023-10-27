# CellPatchExtraction

![Example Image of Patches](./__documentation/cellpatchextraction.png) 

## Overview

This repository contains code for extracting single-cell patches from microscopic images. The primary objective is to facilitate the analysis of cellular structures and their properties.

## Features

- Extracts single-cell patches from microscopic images
- Utilizes advanced image processing techniques
- Offers an example notebook for quick implementation

## Requirements

- Python 3.9
- OpenCV
- NumPy
- PyTorch
- Cellpose ([https://github.com/MouseLand/cellpose](https://github.com/MouseLand/cellpose))
- Scipy
- tifffile
- matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/SimonBon/CellPatchExtraction.git
```

Install the required packages:

Either install the packages in an existing enironment or create a new one using:


```bash
cd CellPatchExtraction

conda create -n CellPatches python=3.9
conda activate Cellpatches

pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the example notebook `Example.ipynb` to get started.

### Advanced Usage

For more control, you can directly use the `extraction.py` script located in the `src` directory.

```python
from CellPatchExtraction import extraction
from plotutils import gridPlot #used for visualization

image_path = "path_to_TIFF_image" # or already loaded image as np.ndarray
model = "path_to_model" # or CellposeModel or one of "CP_TU" or "CP_BM"
diameter = 50 # set mean size of nuclei
min_size = 400 # set minimum size of nuclei, everything below will be discarded
patch_size = 32 # define size of patches
nuclear_channel = 38 # if image has more than 3 channels, define which channel should be used for segmentation

patches = extract_patches(image, model, cellpose_kwargs={"diameter": diameter, "min_size": min_size}, patch_size=32, nuclear_channel=38)

gridPlot(patches)
```

## Contributing

Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
