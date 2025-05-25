# app/core/model_management.py

# Import 'joblib' for loading scikit-learn models and other Python objects efficiently from disk.
# Joblib is particularly good for objects that contain large NumPy arrays.
import joblib
# Import 'os' for interacting with the operating system, primarily for path checking.
import os
# Import 'logging' for recording events, errors, and information during the model loading process.
import logging
# Import typing hints for better code readability and static analysis.
from typing import Dict, Any, Optional, List
# Import 'numpy' as scikit-learn models often return class lists as numpy arrays.
import numpy as np

# Import the paths to model artifact files from the central configuration module.
# This avoids hardcoding paths directly in this file.
from app.core.config import VECTORIZER_PATH, MODEL_PATH, CLASSES_PATH

# Get a logger instance specific to this module.
# This helps in identifying the source of log messages.
logger = logging.getLogger(__name__)

# Internal, module-level dictionary to store the loaded ML artifacts.
# It's prefixed with an underscore (`_loaded_artifacts`) to indicate it's intended
# for internal use within this module, accessed via getter functions.
# This acts as a cache for the loaded models.
_loaded_artifacts: Dict[str, Optional[Any]] = {
    "vectorizer": None,  # Will hold the loaded TF-IDF vectorizer
    "model": None,       # Will hold the loaded classification model (e.g., LinearSVC)
    "model_classes": None # Will hold the list of category names the model was trained on
}

def load_artifacts() -> bool:
    """
    Loads or reloads all ML model artifacts (TF-IDF vectorizer, classification model, and class names)
    from the file paths specified in `app.core.config`.

    This function updates the internal `_loaded_artifacts` dictionary.
    It's designed to be called at application startup and potentially after a model retraining.

    Returns:
        bool: True if all critical artifacts (vectorizer and model) were loaded successfully,
              False otherwise. Note: If model_classes fails to load from file but can be
              retrieved from the model object, it's still considered a partial success for classes.
    """
    # Use the global keyword to modify the module-level dictionary `_loaded_artifacts`.
    global _loaded_artifacts
    
    # Assume success initially, and set to False if any critical loading step fails.
    all_critical_artifacts_loaded = True
    
    try:
        # --- Load TF-IDF Vectorizer ---
        logger.info(f"Attempting to load TF-IDF vectorizer from: {VECTORIZER_PATH}")
        if not os.path.exists(VECTORIZER_PATH):
            logger.error(f"Vectorizer file not found at: {VECTORIZER_PATH}")
            _loaded_artifacts["vectorizer"] = None
            all_critical_artifacts_loaded = False
        else:
            _loaded_artifacts["vectorizer"] = joblib.load(VECTORIZER_PATH)
            logger.info("TF-IDF vectorizer loaded successfully.")

        # --- Load Classification Model ---
        logger.info(f"Attempting to load classification model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at: {MODEL_PATH}")
            _loaded_artifacts["model"] = None
            all_critical_artifacts_loaded = False
        else:
            _loaded_artifacts["model"] = joblib.load(MODEL_PATH)
            logger.info("Classification model loaded successfully.")

        # --- Load Model Classes (Category Names) ---
        # This is important for mapping model output indices back to human-readable category names.
        if all_critical_artifacts_loaded and _loaded_artifacts["model"] is not None: # Only proceed if model loaded
            if os.path.exists(CLASSES_PATH):
                logger.info(f"Attempting to load model classes list from: {CLASSES_PATH}")
                _loaded_artifacts["model_classes"] = joblib.load(CLASSES_PATH)
                logger.info("Model classes list loaded successfully from file.")
            elif hasattr(_loaded_artifacts["model"], 'classes_'):
                # Fallback: If the .joblib file for classes isn't found,
                # try to get the class labels directly from the loaded scikit-learn model object.
                # Many scikit-learn classifiers store class labels in a `classes_` attribute after fitting.
                logger.warning(
                    f"Model classes file ({CLASSES_PATH}) not found. "
                    "Attempting to retrieve classes from the loaded model's 'classes_' attribute."
                )
                _loaded_artifacts["model_classes"] = _loaded_artifacts["model"].classes_
                logger.info("Model classes retrieved from model's 'classes_' attribute.")
            else:
                # If neither the file exists nor the model has a 'classes_' attribute.
                logger.error(
                    f"Could not load model classes from file ({CLASSES_PATH}) "
                    "and the loaded model does not have a 'classes_' attribute."
                )
                _loaded_artifacts["model_classes"] = [] # Default to an empty list
                # Depending on application requirements, this could also be considered a critical failure.
                # For now, vectorizer and model are primary criticals.
        elif not all_critical_artifacts_loaded:
             _loaded_artifacts["model_classes"] = [] # Ensure it's empty if model didn't load

        # Log summary of loaded classes
        if _loaded_artifacts["model_classes"] is not None and len(_loaded_artifacts["model_classes"]) > 0:
            # Convert to list if it's a NumPy array for consistent handling and logging
            if isinstance(_loaded_artifacts["model_classes"], np.ndarray):
                _loaded_artifacts["model_classes"] = _loaded_artifacts["model_classes"].tolist()
            logger.info(f"Model class list ({len(_loaded_artifacts['model_classes'])} classes): "
                        f"{_loaded_artifacts['model_classes'][:5]}..." # Log first 5 classes as a sample
                        )
        elif all_critical_artifacts_loaded: # Only warn if model was expected to be loaded
            logger.warning("Model class list is empty or could not be loaded.")
        
        # Final check for any None critical artifacts after attempts
        if _loaded_artifacts["vectorizer"] is None or _loaded_artifacts["model"] is None:
            all_critical_artifacts_loaded = False
            logger.error("One or more critical ML artifacts (vectorizer or model) failed to load.")

    except FileNotFoundError as e:
        # This broader FileNotFoundError might catch issues if paths are fundamentally wrong.
        logger.error(f"ERROR: FileNotFoundError during artifact loading. A configured model file path might be incorrect. Error: {e}", exc_info=True)
        all_critical_artifacts_loaded = False
        # Ensure artifacts are None if a global error occurred
        _loaded_artifacts["vectorizer"] = None
        _loaded_artifacts["model"] = None
        _loaded_artifacts["model_classes"] = None
    except Exception as e:
        # Catch any other unexpected errors during the loading process.
        logger.error(f"A general error occurred during loading/reloading of model artifacts: {e}", exc_info=True)
        all_critical_artifacts_loaded = False
        # Ensure artifacts are None if a global error occurred
        _loaded_artifacts["vectorizer"] = None
        _loaded_artifacts["model"] = None
        _loaded_artifacts["model_classes"] = None
    
    return all_critical_artifacts_loaded

def get_vectorizer() -> Optional[Any]:
    """
    Provides access to the loaded TF-IDF vectorizer.

    Returns:
        Optional[Any]: The loaded scikit-learn TF-IDF vectorizer object, 
                       or None if it hasn't been loaded successfully.
    """
    return _loaded_artifacts["vectorizer"]

def get_model() -> Optional[Any]:
    """
    Provides access to the loaded classification model.

    Returns:
        Optional[Any]: The loaded scikit-learn classification model object,
                       or None if it hasn't been loaded successfully.
    """
    return _loaded_artifacts["model"]

def get_model_classes() -> Optional[List[str]]:
    """
    Provides access to the list of model class names (categories).
    Ensures the returned type is a Python list of strings.

    Returns:
        Optional[List[str]]: A list of category names, 
                             or None if not loaded or empty.
    """
    model_cls = _loaded_artifacts["model_classes"]
    if model_cls is None:
        return None
    # Scikit-learn's model.classes_ attribute is often a NumPy array.
    # Convert to a Python list for consistent usage elsewhere in the application.
    if isinstance(model_cls, np.ndarray):
        return model_cls.tolist()
    # If it's already a list (or other iterable, though list is expected after loading)
    if isinstance(model_cls, list):
        return model_cls
    # Fallback for unexpected types, though ideally, loading normalizes this.
    logger.warning(f"Model classes are of an unexpected type: {type(model_cls)}. Attempting to convert to list.")
    try:
        return list(model_cls)
    except TypeError:
        logger.error(f"Could not convert model classes of type {type(model_cls)} to a list.")
        return []


def are_models_loaded() -> bool:
    """
    Checks if all essential ML artifacts (vectorizer, model, and classes)
    are loaded and available for use.

    Returns:
        bool: True if all essential artifacts are loaded and model_classes is not empty,
              False otherwise.
    """
    vectorizer_loaded = _loaded_artifacts["vectorizer"] is not None
    model_loaded = _loaded_artifacts["model"] is not None
    classes_loaded = (_loaded_artifacts["model_classes"] is not None and
                      len(_loaded_artifacts["model_classes"]) > 0)
    
    if not vectorizer_loaded:
        logger.warning("are_models_loaded check: Vectorizer is not loaded.")
    if not model_loaded:
        logger.warning("are_models_loaded check: Model is not loaded.")
    if not classes_loaded:
        logger.warning("are_models_loaded check: Model classes are not loaded or are empty.")
        
    return vectorizer_loaded and model_loaded and classes_loaded

# --- Initial Load Attempt (Optional) ---
# You could attempt an initial load when this module is first imported,
# but it's generally better to trigger this explicitly at application startup
# (e.g., in main.py's lifespan event) for more control over error handling
# and application initialization sequence.
# Example:
# if not are_models_loaded():
#     print("ModelManagement: Performing initial artifact load on module import...")
#     load_artifacts()