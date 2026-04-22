from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
MODELS_DIR = PROJ_ROOT / "models"

# Define column types for the pipeline
NUM_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CAT_FEATURES = ["Sex", "Embarked", "Pclass"]
TARGET = "Survived"
