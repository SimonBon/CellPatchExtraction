from pathlib import Path

TYPES = ('*.tiff', '*.TIF', '*.TIFF', '*.tif')
CELLPOSE_PATH = Path(__file__).resolve().parent.parent / "cellpose_model"
AVAIL_MODELS = [p.name for p in Path(CELLPOSE_PATH).glob("*")]