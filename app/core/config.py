# app/core/config.py

# Import the 'os' module for operating system dependent functionality,
# especially for constructing file and directory paths in a platform-independent way.
import os
import logging # Import logging to use its level constants like logging.INFO

# app/core/config.py
import os
import logging
from datetime import timedelta # For token expiration

from dotenv import load_dotenv

# --- Load Environment Variables from .env file ---
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_file_dir, "..", ".."))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    # print(f"INFO: Loaded environment variables from: {dotenv_path}") # Good for debug
else:
    # print(f"INFO: .env file not found at {dotenv_path}. Using defaults or system env vars.") # Good for debug
    pass

# --- Path Configurations ---
# These paths help locate important directories and files for the application.

# PROJECT_ROOT: Defines the absolute path to the root directory of your project.
# This is calculated by navigating two levels up from the current file's directory (app/core/).
# Assumes this file (config.py) is located at `product_categorizer_project/app/core/config.py`.
# Adjust if your project structure or the location of this file changes.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# MODEL_STORE_DIR: Absolute path to the directory where trained machine learning
# model artifacts (like .joblib files) are stored.
# It's constructed by joining the PROJECT_ROOT with the 'models_store' folder name.
MODEL_STORE_DIR = os.path.join(PROJECT_ROOT, "models_store")

# Specific paths to the model artifact files.
# These are used by `model_management.py` to load the ML components.
VECTORIZER_PATH = os.path.join(MODEL_STORE_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_STORE_DIR, "svc_model.joblib")
CLASSES_PATH = os.path.join(MODEL_STORE_DIR, "model_classes.joblib") # Stores the list of category names

# TRAINING_DATA_DIR: Absolute path to the directory containing the raw training data (e.g., Excel files).
# This path is used by the `train.py` script and could be used by a future retraining API.
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# --- ML Model and Prediction Configurations ---

# TOP_N_PREDICTIONS: Specifies the number of top predictions (category and score)
# to return by the prediction API.
TOP_N_PREDICTIONS = 5


# --- Logging Configurations ---
# These settings control how logging messages are formatted and displayed.

# LOGGING_LEVEL: Sets the minimum severity level for log messages to be processed.
# Common levels: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
# Using "INFO" means DEBUG messages will be ignored.
# The actual logging level constant (e.g., logging.INFO) is retrieved in main.py.
LOGGING_LEVEL = "INFO" # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

# LOGGING_FORMAT: Defines the format string for log messages.
# - %(asctime)s: Time the log record was created.
# - %(levelname)s: Textual representation of the log level (e.g., "INFO", "WARNING").
# - %(name)s: Name of the logger used to log the call (usually the module name).
# - %(module)s: Module name part of the pathname of the source file.
# - %(message)s: The logged message.
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S' # Date format for the asctime part of the log.


# --- NLTK Data Configurations ---

# NLTK_PACKAGES: A dictionary defining the NLTK data packages required by the application.
# Keys are the short names used with `nltk.download()`.
# Values are the resource locators used with `nltk.data.find()` for verification.
# These are used by `app.core.nltk_utils.download_nltk_essential_data()`.
NLTK_PACKAGES = {
    'wordnet': 'corpora/wordnet',                      # For lemmatization (WordNetLemmatizer)
    'omw-1.4': 'corpora/omw-1.4',                      # Open Multilingual WordNet, often used with wordnet
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger', # For Part-of-Speech (POS) tagging (nltk.pos_tag)
    'stopwords': 'corpora/stopwords',                  # For lists of common stopwords
    'punkt': 'tokenizers/punkt'                        # For sentence and word tokenization (nltk.word_tokenize)
}


# --- Training Script Default Parameters ---
# These default values can be used by the `train.py` script if specific arguments
# are not provided, or by a future retraining API to standardize training runs.

# DEFAULT_MAX_FEATURES: Default maximum number of features (terms) for TfidfVectorizer.
DEFAULT_MAX_FEATURES = 15000

# DEFAULT_C_PARAM: Default regularization parameter 'C' for LinearSVC.
# 'C' is inversely proportional to the strength of regularization; smaller C means stronger regularization.
DEFAULT_C_PARAM = 1.0

# DEFAULT_MODEL_DIR_IN_TRAIN_SCRIPT: Default directory where train.py saves its artifacts.
# This should align with MODEL_STORE_DIR if the API and train.py use the same models.
# Note: train.py uses an argument `--model-dir`. If that argument is relative,
# this config might define how that relative path is interpreted or provide a default.
# For consistency, ensure train.py saves to the location from which model_management loads.
# If train.py is run from PROJECT_ROOT and uses "models_store" as --model-dir, it aligns.
DEFAULT_TRAIN_OUTPUT_MODEL_DIR = "models_store" # This aligns with MODEL_STORE_DIR's folder name

# DEFAULT_INPUT_FOLDER_IN_TRAIN_SCRIPT: Default input data folder for train.py.
# This aligns with TRAINING_DATA_DIR's folder name.
DEFAULT_TRAIN_INPUT_DATA_DIR = "data"
 
# --- Database Configuration ---
# For SQLite, the path is relative to where the app runs.
# We'll place it in the project root for simplicity during development.
# Example: "sqlite:///./api_clients.db"
# For PostgreSQL: "postgresql://user:password@host:port/database"
# For MySQL: "mysql+mysqlconnector://user:password@host:port/database"
DATABASE_FILENAME = "api_clients.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(PROJECT_ROOT, DATABASE_FILENAME)}"
# SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(PROJECT_ROOT, DATABASE_FILENAME)}") # Example using env var

# --- JWT / Authentication Configuration ---
# This SECRET_KEY should be very strong and kept secret in production (e.g., from environment variable).
# To generate a good secret key, you can use: openssl rand -hex 32
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-default-secret-key-SHOULD-NOT-BE-USED-IN-PROD")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256") # Fine to have a default algo.
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")) # Fine to have default expiry.

# --- API Client Configuration (Example - for initial client creation script) ---
# These are NOT for storing active client secrets directly in config for the app to read for auth.
# The app will read hashed secrets from the database.
# This might be used by a script to create an initial client.
# EXAMPLE_INITIAL_CLIENT_ID = "testclient"
# EXAMPLE_INITIAL_CLIENT_SECRET = "testsecret" # This would be hashed by the creation script