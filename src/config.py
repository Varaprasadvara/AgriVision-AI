import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(os.environ.get("AGRIVISION_DATA_ROOT", PROJECT_DIR)).expanduser()
ASSETS_DIR = PROJECT_DIR / "assets"
DATA_DIR = BASE_DIR

CROP_DATASET_PATH = Path(
    os.environ.get(
        "AGRIVISION_CROP_CSV",
        BASE_DIR / "archive" / "Crop_recommendationV2.csv",
    )
)
HYPER_SPECTRAL_DIR = Path(
    os.environ.get(
        "AGRIVISION_HYPERSPECTRAL_DIR",
        BASE_DIR / "beyond-visible-spectrum-ai-for-agriculture-P1",
    )
)
PEST_DATASET_DIR = Path(
    os.environ.get(
        "AGRIVISION_PEST_DIR",
        BASE_DIR / "pest" / "classification",
    )
)
PEST_CLASSES_PATH = Path(
    os.environ.get(
        "AGRIVISION_PEST_CLASSES",
        BASE_DIR / "pest" / "classes.txt",
    )
)

RANDOM_STATE = 42
IMAGE_SIZE = (64, 64)
MAX_PEST_IMAGES = 500