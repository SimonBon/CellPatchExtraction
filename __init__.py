import os 

TYPES = ('*.tiff', '*.TIF', '*.TIFF', '*.tif')
CELLPOSE_PATH = f"{os.path.dirname(__file__)}/.cellpose_model"
AVAIL_MODELS = os.listdir(CELLPOSE_PATH)