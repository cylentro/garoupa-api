# app/services/prediction_service.py

# Import 'logging' for recording informational messages and errors during the prediction process.
import logging
# Import 'numpy' for numerical operations, particularly for handling model scores and sorting.
import numpy as np
# Import typing hints for function signatures, improving code clarity and enabling static analysis.
from typing import List, Tuple, Optional # Removed Dict as it's not directly used in return types here

# Import text preprocessing functions from the app's ML utilities.
# 'basic_clean' performs initial cleaning (lowercase, punctuation removal).
# 'process_text_lemma' performs tokenization, stopword removal, and lemmatization.
from app.ml.preprocessing import basic_clean, process_text_lemma

# Import the model management module to access the loaded ML artifacts (vectorizer, model, classes).
from app.core import model_management

# Import Pydantic schema for structuring the category prediction output.
# This ensures the service's output is consistent with the API's response model.
from app.schemas.product_schemas import CategoryPrediction

# Import configuration values, specifically the number of top predictions to return.
from app.core.config import TOP_N_PREDICTIONS

# Get a logger instance specific to this module.
logger = logging.getLogger(__name__)

def get_category_predictions(product_name: str) -> Tuple[Optional[str], Optional[float], List[CategoryPrediction]]:
    """
    Processes a raw product name and returns the top predicted category,
    its confidence score, and a list of the top N predictions with their scores.

    This function encapsulates the entire prediction pipeline:
    1. Retrieves loaded ML artifacts (vectorizer, model, class names).
    2. Preprocesses the input product name (cleaning, lemmatization).
    3. Transforms the processed name into a numerical vector using the TF-IDF vectorizer.
    4. Uses the classification model to get decision scores for each category.
    5. Sorts the predictions and formats them according to the CategoryPrediction schema.

    Args:
        product_name (str): The raw name of the product as input by the user.

    Returns:
        Tuple[Optional[str], Optional[float], List[CategoryPrediction]]:
            - Element 1 (Optional[str]): The name of the top predicted category.
                                         None if a critical error occurs or no prediction can be made.
            - Element 2 (Optional[float]): The confidence score (or decision function value)
                                           for the top predicted category. None if error.
            - Element 3 (List[CategoryPrediction]): A list of `CategoryPrediction` objects
                                                    representing the top N predictions.
                                                    An empty list if an error occurs or no predictions.
    """
    # Attempt to retrieve the loaded ML artifacts.
    # These functions (`get_vectorizer`, `get_model`, `get_model_classes`) are defined
    # in `app.core.model_management` and return the artifacts if loaded, otherwise None.
    vectorizer = model_management.get_vectorizer()
    model = model_management.get_model()
    model_classes = model_management.get_model_classes()

    # --- Pre-condition Check: Ensure ML artifacts are loaded ---
    # If any critical artifact is missing, log an error and return indicating failure.
    # The `are_models_loaded()` function provides a comprehensive check.
    if not model_management.are_models_loaded(): # A more robust check
        logger.error("Prediction service cannot proceed: Essential ML artifacts (vectorizer, model, or classes) are not loaded.")
        return None, None, [] # Return empty/None values to signify failure

    # Defensive check, though are_models_loaded should cover this.
    # Explicitly checking each component for clarity in this step.
    if not vectorizer or not model or not model_classes:
        # This state should ideally be caught by are_models_loaded(), but as a fallback:
        logger.critical("Critical error in prediction service: One or more ML components are None even after passing initial checks.")
        return None, None, []


    try:
        # --- Step 1: Preprocessing the Input Product Name ---
        logger.debug(f"Original product name received: '{product_name}'")
        # Apply basic cleaning (lowercase, remove punctuation, etc.).
        cleaned_name = basic_clean(str(product_name)) # Ensure input is string
        # Apply advanced processing (tokenization, stopword removal, lemmatization).
        # This step requires NLTK data to be available.
        processed_name = process_text_lemma(cleaned_name)
        logger.info(f"Product name after full preprocessing: '{processed_name}'")

        # Handle cases where the text becomes empty after preprocessing.
        # An empty string cannot be vectorized and will cause errors.
        if not processed_name or not processed_name.strip():
            logger.warning(f"Input product name '{product_name}' became empty after preprocessing. Returning a specific indicator.")
            # Return a special category or handle as an error, depending on desired API behavior.
            # "UNKNOWN_EMPTY_INPUT" can be a placeholder the router can interpret.
            return "UNKNOWN_EMPTY_INPUT", 0.0, [] # Score 0.0, no top predictions

        # --- Step 2: Transform Text to Numerical Vector (TF-IDF) ---
        # The loaded TF-IDF vectorizer converts the processed text string into a numerical feature vector.
        # The vectorizer expects a list of strings as input.
        vectorized_name = vectorizer.transform([processed_name])
        logger.debug(f"Shape of vectorized name: {vectorized_name.shape}")


        # --- Step 3: Get Prediction Scores from the Model ---
        # Check if the model has the 'decision_function' method (common for SVMs like LinearSVC).
        # This method provides scores indicating the distance from the decision boundary for each class.
        if not hasattr(model, 'decision_function'):
            logger.error("Loaded classification model does not have the 'decision_function' method. Cannot get confidence scores.")
            return None, None, [] # Cannot proceed without a way to get scores

        # Get the raw scores for each class from the model.
        # `decision_function` returns a 2D array for multiple samples; we take the first row [0] for our single sample.
        scores = model.decision_function(vectorized_name)[0]
        logger.debug(f"Raw scores from model: {scores[:5]}... (for {len(scores)} classes)")

        # --- Sanity Check: Match number of scores with number of classes ---
        if len(scores) != len(model_classes):
            logger.error(
                f"Mismatch between the number of scores returned by the model ({len(scores)}) "
                f"and the number of known model classes ({len(model_classes)}). This indicates a potential model/class list mismatch."
            )
            return None, None, []


        # --- Step 4: Format Output - Get Top N Predictions ---
        # Sort the class indices based on the scores in descending order (highest score first).
        # `np.argsort` returns the indices that would sort an array. `[::-1]` reverses it.
        indices_sorted_by_score = np.argsort(scores)[::-1]

        top_predictions_list: List[CategoryPrediction] = []
        # Iterate through the sorted indices to get the top N predictions.
        # `TOP_N_PREDICTIONS` is from `config.py`.
        # Ensure we don't try to get more predictions than available classes.
        num_predictions_to_return = min(TOP_N_PREDICTIONS, len(model_classes))

        for i in range(num_predictions_to_return):
            class_index = indices_sorted_by_score[i]
            category_name = str(model_classes[class_index]) # Get category name using the index
            category_score = float(scores[class_index])      # Get the corresponding score

            # Create a CategoryPrediction object (Pydantic model) for this prediction.
            top_predictions_list.append(
                CategoryPrediction(category=category_name, score=category_score)
            )

        # If, for some reason, no predictions could be formed (e.g., if model_classes was empty initially).
        if not top_predictions_list:
            logger.warning(f"No top predictions could be generated for '{product_name}', though model processing seemed to complete.")
            return None, None, []

        # The top predicted category and its score are from the first item in the sorted list.
        top_predicted_category = top_predictions_list[0].category
        top_confidence_score = top_predictions_list[0].score

        logger.info(
            f"Prediction for '{product_name}' successful: Top category = '{top_predicted_category}' (Score: {top_confidence_score:.4f})"
        )

        return top_predicted_category, top_confidence_score, top_predictions_list

    except RuntimeError as e:
        # Specifically catch RuntimeErrors that might be raised by preprocessing functions
        # if NLTK components are not properly initialized (e.g., lemmatizer fails).
        logger.error(f"A RuntimeError occurred during prediction processing for '{product_name}'. "
                     f"This often indicates issues with NLTK data/setup. Error: {e}", exc_info=True)
        return None, None, [] # Return empty/None to signify failure
    except Exception as e:
        # Catch any other unexpected exceptions during the prediction process.
        logger.error(f"An unexpected error occurred in the prediction service for product '{product_name}': {e}", exc_info=True)
        return None, None, [] # Return empty/None to signify failure