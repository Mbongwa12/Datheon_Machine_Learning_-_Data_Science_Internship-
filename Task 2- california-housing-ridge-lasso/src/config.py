"""Shared configuration: paths, constants, feature lists."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/"
    "master/datasets/housing/housing.csv"
)
LOCAL_CSV = DATA_DIR / "housing.csv"

TARGET = "median_house_value"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 8 original numeric columns + the 3 engineered ratios (see features.py)
NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
