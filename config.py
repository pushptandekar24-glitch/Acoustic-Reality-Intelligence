"""
Configuration settings and constants for Acoustic Reality Intelligence.
"""
import os
import tempfile
from pathlib import Path

# Ensure Matplotlib uses a writable temp cache directory
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "ari_matplotlib")

# Project Identity
APP_NAME = "Acoustic Reality Intelligence"
APP_SUBTITLE = "An AI-Powered Environmental Audio Analysis and Synthetic Sound Assessment System"
APP_VERSION = "2.0.0"

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "acoustic_model.joblib"
METADATA_FILE = MODELS_DIR / "acoustic_model_metadata.json"
ASSETS_DIR = BASE_DIR / "existing_assets"
SAMPLE_AUDIO_DIR = BASE_DIR / "sample_audio"
SAMPLE_MANIFEST_FILE = SAMPLE_AUDIO_DIR / "sample_manifest.json"

# Fallback root paths if models dir is missing
ROOT_MODEL_FILE = BASE_DIR / "acoustic_model.joblib"
ROOT_METADATA_FILE = BASE_DIR / "acoustic_model_metadata.json"

# Audio Processing Standards (from original training procedure)
TARGET_SAMPLE_RATE = 16000
IS_MONO = True
N_MFCC = 13
FEATURE_DIMENSION = 26  # 13 means + 13 standard deviations
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]

# Model Verification
EXPECTED_MODEL_SHA256 = "b6d965c31ceba6a6be2483af2e378dead36f59a44ea97f5c3b3cf6f5a8160f91"

# Verified Class Mapping (Classes: 0 = Real, 1 = AI-generated)
VERIFIED_CLASS_NAMES = {
    0: "Real",
    1: "AI-generated",
}

VERIFIED_HEADLINES = {
    0: "Your audio appears to be REAL",
    1: "Your audio appears to be AI-GENERATED",
}

# Theme Colors (Dark Futuristic Palette)
COLOR_PAGE_BG = "#080c16"
COLOR_PANEL_BG = "#101725"
COLOR_BORDER = "#263248"
COLOR_TEXT_PRIMARY = "#f3f6ff"
COLOR_TEXT_MUTED = "#9aa9c1"
COLOR_CYAN = "#31d2ff"
COLOR_BLUE = "#3878ff"
COLOR_PURPLE = "#9561ff"
COLOR_REAL_GREEN = "#2bd49a"
COLOR_AI_MAGENTA = "#ff6d8b"

