#!/usr/bin/env python
# coding: utf-8

# train.py (Modified to use the integrated training pipeline)

# Standard library imports
import argparse  # For parsing command-line arguments
import logging   # For logging script activity and progress
import os        # For operating system dependent functionality (like path manipulation)
import sys       # For system-specific parameters and functions (like modifying sys.path)

# --- Adjust Python Path to find the 'app' Package ---
# This section is crucial if you run this script directly from the project root
# and it needs to import modules from a sub-package like 'app'.
# It adds the project's root directory (where this script is located)
# to Python's search path for modules.
# This allows imports like `from app.ml.training_pipeline import ...` to work.
try:
    # Determine the absolute path to the directory containing this script (train.py).
    # This is assumed to be the project root.
    PROJECT_ROOT_FOR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to sys.path if it's not already there.
    # sys.path is a list of strings that specifies the search path for modules.
    if PROJECT_ROOT_FOR_SCRIPT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FOR_SCRIPT) # Insert at the beginning for priority
        print(f"Info: Added '{PROJECT_ROOT_FOR_SCRIPT}' to sys.path for module resolution.")

    # Now, attempt to import the necessary modules from the 'app' package.
    from app.ml.training_pipeline import run_complete_training_pipeline
    from app.core import config  # To get default values for paths and training parameters
    from app.core.nltk_utils import download_nltk_essential_data # For NLTK data setup

except ImportError as e:
    # If imports fail, print an informative error message and exit.
    # This usually indicates issues with the script's location, PYTHONPATH, or missing modules.
    print(f"Error: Could not import necessary application modules from the 'app' package.")
    print(f"Please ensure 'train.py' is located in the project root directory and that the 'app' package is accessible.")
    print(f"Import error details: {e}")
    sys.exit(1) # Exit the script with a non-zero status code to indicate failure


# --- Configure Logging for this Standalone Script ---
# This sets up logging specifically for when this script is run.
# It's independent of the FastAPI application's logging when the API triggers training.
# - level: Minimum severity of messages to log (e.g., INFO, DEBUG).
# - format: Defines the structure of log messages.
# - datefmt: Defines the format for the timestamp in log messages.
logging.basicConfig(
    level=logging.INFO, # You can change this to logging.DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - SCRIPT - %(message)s', # Added "SCRIPT" to distinguish
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Create a logger instance for this script.
logger = logging.getLogger("standalone_train_script")


# --- Main Execution Block ---
# This code runs only when the script is executed directly (not when imported as a module).
if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    # Initialize ArgumentParser to handle command-line arguments.
    parser = argparse.ArgumentParser(
        description="Standalone script to trigger the ML model training pipeline. "
                    "This script calls the integrated training logic from app.ml.training_pipeline."
    )
    
    # Define command-line arguments.
    # Default values are sourced from `app.core.config` for consistency.
    parser.add_argument(
        "--input-folder",
        type=str,
        default=config.TRAINING_DATA_DIR, # Default path for training data
        help=(
            f"Path to the folder containing input Excel data files for training. "
            f"Default: {config.TRAINING_DATA_DIR}"
        )
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=config.MODEL_STORE_DIR, # Default path for saving model artifacts
        help=(
            f"Directory where the trained model, vectorizer, and class labels will be saved. "
            f"Default: {config.MODEL_STORE_DIR}"
        )
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=config.DEFAULT_MAX_FEATURES, # Default for TfidfVectorizer
        help=(
            f"Maximum number of features (terms) for the TfidfVectorizer. "
            f"Default: {config.DEFAULT_MAX_FEATURES}"
        )
    )
    parser.add_argument(
        "--C",
        type=float,
        default=config.DEFAULT_C_PARAM, # Default C for LinearSVC
        help=(
            f"Regularization parameter 'C' for the LinearSVC model. "
            f"Default: {config.DEFAULT_C_PARAM}"
        )
    )
    # Example of adding more configurable parameters if needed:
    # parser.add_argument(
    #     "--text-column", type=str, default="item_name",
    #     help="Name of the column in the Excel files containing the text to be processed."
    # )
    # parser.add_argument(
    #     "--target-column", type=str, default="Fixed Category Code",
    #     help="Name of the column in the Excel files containing the target category labels."
    # )

    # Parse the arguments provided from the command line.
    args = parser.parse_args()

    logger.info("Standalone training script initiated with parsed arguments.")
    logger.info(f"Input Data Folder: {args.input_folder}")
    logger.info(f"Model Output Directory: {args.model_dir}")
    logger.info(f"Max TF-IDF Features: {args.max_features}")
    logger.info(f"LinearSVC C Parameter: {args.C}")
    # If you added more args:
    # logger.info(f"Text Column: {args.text_column}")
    # logger.info(f"Target Column: {args.target_column}")


    # --- Ensure NLTK Data is Available (for standalone script execution) ---
    # The FastAPI application handles NLTK data on its startup via the lifespan manager.
    # However, when this script is run directly, it also needs to ensure NLTK data is present
    # because the underlying `app.ml.preprocessing` functions (called by the pipeline) depend on it.
    logger.info("Checking/Downloading NLTK data packages required for preprocessing...")
    # Call the same utility function used by the FastAPI app for consistency.
    # The list of required packages is sourced from `config.NLTK_PACKAGES`.
    download_nltk_essential_data(config.NLTK_PACKAGES)
    logger.info("NLTK data check/download process completed for script execution.")


    # --- Prepare Training Parameters for the Pipeline ---
    # Create a dictionary of training parameters to pass to the integrated pipeline function.
    # This allows the pipeline to be flexible and accept various configurations.
    pipeline_training_params = {
        'max_tfidf_features': args.max_features,
        'svc_c_param': args.C
        # If you added text_column and target_column to argparse:
        # 'text_column': args.text_column,
        # 'target_column': args.target_column,
        # Add any other parameters the `run_complete_training_pipeline` might accept in its `training_params` dict.
    }
    logger.debug(f"Training parameters for pipeline: {pipeline_training_params}")

    # --- Execute the Centralized Training Pipeline ---
    # Call the main orchestrator function from `app.ml.training_pipeline`.
    # This function now contains all the core logic for loading data, preprocessing,
    # training, evaluating, and saving artifacts.
    logger.info("Invoking the integrated model training pipeline...")
    training_successful = run_complete_training_pipeline(
        data_input_folder=args.input_folder,      # Pass the input data folder path
        model_output_folder=args.model_dir,     # Pass the model output directory path
        training_params=pipeline_training_params  # Pass the dictionary of training parameters
    )

    # --- Report Final Status ---
    if training_successful:
        logger.info("Standalone training script completed successfully. Model artifacts should be updated in the specified directory.")
    else:
        logger.error("Standalone training script encountered errors during the pipeline execution. Please check the logs above for details.")
        sys.exit(1) # Exit with a non-zero status code to indicate failure to calling processes/scripts.

    logger.info("Standalone training script finished.")