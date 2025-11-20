import logging
from pathlib import Path

try:
    from loguru import logger as _loguru_logger
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class _FallbackLogger:
        """Minimal logger compatible with loguru calls used in this project."""

        def __init__(self) -> None:
            self._logger = logging.getLogger("ireranker")
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        def info(self, msg: str, *args, **kwargs):
            self._logger.info(msg, *args, **kwargs)

        def warning(self, msg: str, *args, **kwargs):
            self._logger.warning(msg, *args, **kwargs)

        def error(self, msg: str, *args, **kwargs):
            self._logger.error(msg, *args, **kwargs)

        def debug(self, msg: str, *args, **kwargs):
            self._logger.debug(msg, *args, **kwargs)

        def success(self, msg: str, *args, **kwargs):
            self._logger.info(msg, *args, **kwargs)

        def remove(self, *args, **kwargs):
            return None

        def add(self, *args, **kwargs):
            return None

    logger = _FallbackLogger()
else:
    logger = _loguru_logger

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    logger.warning("python-dotenv not installed; skipping environment file loading")

    def load_dotenv(*args, **kwargs):  # type: ignore[redefine-outer-name]
        return False


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

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
