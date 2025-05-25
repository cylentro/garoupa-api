# app/services/training_service.py

# Import 'logging' for recording the progress and outcome of the training process.
import logging

# Import configuration settings, especially default paths and training parameters.
from app.core import config
# Import the model management function to reload artifacts after successful training.
from app.core.model_management import load_artifacts as reload_all_model_artifacts
# Import the main orchestrator function for the training pipeline.
from app.ml.training_pipeline import run_complete_training_pipeline # <-- NEW IMPORT

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

def execute_model_retraining_pipeline_integrated():
    """
    Executes the model retraining pipeline by directly calling the integrated
    training functions within the application.

    This function is designed to be run as a background task. It handles:
    - Calling the `run_complete_training_pipeline` from `app.ml.training_pipeline`.
    - If the pipeline completes successfully, it triggers the reloading of the
      newly trained model artifacts into the application.
    """
    logger.info("Starting integrated model retraining pipeline execution...")

    try:
        # --- Define parameters for the training pipeline ---
        # These will be passed to the run_complete_training_pipeline function.
        # They primarily come from our central configuration.
        data_input_folder_path = config.TRAINING_DATA_DIR
        model_output_folder_path = config.MODEL_STORE_DIR
        
        # Optional: If you want to pass specific training parameters that might
        # override defaults within the pipeline, you can construct a dictionary here.
        # For now, we'll let `run_complete_training_pipeline` use its internal defaults
        # which also source from `config.py`.
        custom_training_params = {
            'max_tfidf_features': config.DEFAULT_MAX_FEATURES,
            'svc_c_param': config.DEFAULT_C_PARAM,
            # Add other parameters like 'text_column', 'target_column' if you want to
            # make them configurable from here or a future API request.
            # 'text_column': 'product_description', # Example
            # 'target_column': 'category_id'      # Example
        }
        # If custom_training_params is empty or None, run_complete_training_pipeline will use its defaults.

        logger.info(f"Data input folder: {data_input_folder_path}")
        logger.info(f"Model output folder: {model_output_folder_path}")
        logger.info(f"Custom training parameters (if any): {custom_training_params if custom_training_params else 'Using defaults'}")

        # --- Execute the integrated training pipeline ---
        # This function now directly calls our Python-based training logic.
        training_successful = run_complete_training_pipeline(
            data_input_folder=data_input_folder_path,
            model_output_folder=model_output_folder_path,
            training_params=custom_training_params # Pass the params dict
        )
        
        # --- Check Training Result and Reload Models ---
        if training_successful:
            logger.info("Integrated training pipeline completed successfully.")
            logger.info("Attempting to reload model artifacts with newly trained models...")
            
            # Call the function from model_management to reload all artifacts.
            if reload_all_model_artifacts():
                logger.info("Model artifacts reloaded successfully after retraining.")
            else:
                logger.error("Failed to reload model artifacts after successful training. "
                             "The application might still be using the old models.")
        else:
            logger.error(
                "Integrated training pipeline failed. "
                "New models were NOT loaded. Check logs from 'app.ml.training_pipeline' for details."
            )

    except Exception as e:
        # Catch any unexpected errors during the retraining trigger process.
        logger.error(f"An unexpected error occurred in the training service layer: {e}", exc_info=True)

    logger.info("Integrated model retraining pipeline execution finished.")