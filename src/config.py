"""
Configuration Module

Central configuration for the NYC Taxi Trip Duration prediction project.
"""

from pathlib import Path
import yaml

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLED_DATA_DIR = DATA_DIR / "sampled"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLED_DATA_DIR, 
                 MODELS_DIR, VISUALIZATIONS_DIR, DOCS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data Files
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
TRAIN_PROCESSED = PROCESSED_DATA_DIR / "train_processed.parquet"
TEST_PROCESSED = PROCESSED_DATA_DIR / "test_processed.parquet"
TRAIN_SAMPLE = SAMPLED_DATA_DIR / "train_sample_10k.csv"

# Model Files
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# Data Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Big Data Configuration
CHUNK_SIZE = 100000  # Number of rows per chunk
DASK_BLOCKSIZE = '25MB'  # Dask partition size
N_JOBS = -1  # Use all available CPU cores
SAMPLE_SIZE = 10000  # Sample size for quick iteration (0.67% of full data)
SAMPLE_FRACTION = 0.01  # 1% sample for very quick tests

# Feature Engineering
DISTANCE_UNIT = 'km'  # 'km' or 'miles'
TIME_ZONE = 'America/New_York'

# NYC Geographic Constants
NYC_CENTER = {'lat': 40.7580, 'lon': -73.9855}  # Times Square
JFK_AIRPORT = {'lat': 40.6413, 'lon': -73.7781}
LAGUARDIA_AIRPORT = {'lat': 40.7769, 'lon': -73.8740}
NEWARK_AIRPORT = {'lat': 40.6895, 'lon': -74.1745}

# Airports with radius in km
AIRPORTS = {
    'JFK': {'lat': 40.6413, 'lon': -73.7781, 'radius_km': 3.0},
    'LGA': {'lat': 40.7769, 'lon': -73.8740, 'radius_km': 2.5},
    'EWR': {'lat': 40.6895, 'lon': -74.1745, 'radius_km': 3.0}
}

# Data Quality Filters
MIN_TRIP_DURATION = 60  # seconds (1 minute)
MAX_TRIP_DURATION = 10800  # seconds (3 hours)
MIN_DISTANCE = 0.1  # km
MAX_DISTANCE = 100  # km
MIN_SPEED = 1  # km/h (for validation)
MAX_SPEED = 100  # km/h (for validation)

# Feature Engineering Settings
TEMPORAL_FEATURES = True
GEOSPATIAL_FEATURES = True
INTERACTION_FEATURES = True
AGGREGATED_FEATURES = True
POLYNOMIAL_DEGREE = 2

# Rush Hours (NYC typical rush hours)
MORNING_RUSH = (7, 9)  # 7am-9am
EVENING_RUSH = (17, 19)  # 5pm-7pm

# Model Configuration
MODELS_TO_TRAIN = ['ridge', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
DEFAULT_MODEL = 'lightgbm'

# Model Hyperparameters (baseline)
MODEL_PARAMS = {
    'ridge': {
        'alpha': 10.0
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'n_jobs': N_JOBS,
        'random_state': RANDOM_STATE
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        'n_jobs': N_JOBS
    },
    'lightgbm': {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': N_JOBS
    }
}

# Optuna Tuning Configuration
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour
OPTUNA_N_JOBS = 2  # Parallel trials

# Metrics
PRIMARY_METRIC = 'rmse'
METRICS = ['rmse', 'mae', 'r2', 'mape']

# Visualization Settings
FIGURE_SIZE = (12, 6)
FIGURE_DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'viridis'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Column Names (for reference)
ORIGINAL_COLUMNS = [
    'id',
    'vendor_id',
    'pickup_datetime',
    'dropoff_datetime',
    'passenger_count',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'store_and_fwd_flag',
    'trip_duration'  # target
]

TARGET_COLUMN = 'trip_duration'
ID_COLUMN = 'id'
DATETIME_COLUMNS = ['pickup_datetime', 'dropoff_datetime']

# Features to drop before modeling
COLUMNS_TO_DROP = ['id', 'pickup_datetime', 'dropoff_datetime', 'dropoff_datetime']


def load_config(config_file: str = None) -> dict:
    """
    Load configuration from YAML file (if exists).
    
    Args:
        config_file: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_config(config: dict, config_file: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_file: Output path
    """
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
