from pathlib import Path

from dotenv import load_dotenv
from loguru import logger as _loguru_logger
from tqdm import tqdm

logger = _loguru_logger


load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


try:
    logger.remove(0)
except ValueError:
    # Handler 0 may not exist if this module is being reloaded
    pass
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
