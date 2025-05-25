# app/ml/training_pipeline.py

# Standard library imports
import os
import logging
from pathlib import Path # For modern path manipulation
from typing import List, Tuple, Dict, Any, Optional

# Type hint for Union
from typing import Union

# Third-party library imports
import pandas as pd
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving/loading scikit-learn models

# Application-specific imports
from app.ml.preprocessing import basic_clean, process_text_lemma # Assuming these are the core functions needed
from app.core import config # For accessing configuration like default paths and params

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Data Loading Function (Replaces chlib.get_file_paths and chlib.excel_to_df for this context) ---
def load_and_combine_excel_data(data_folder_path: str, sheet_name: Union[str, int] = 0, skiprows: int = 0) -> Optional[pd.DataFrame]:
    """
    Loads all Excel files (.xlsx) from a specified folder, reads a specific sheet
    from each, and concatenates them into a single Pandas DataFrame.

    Args:
        data_folder_path (str): The absolute path to the folder containing Excel files.
        sheet_name (Union[str, int], optional): The name or index of the sheet to read. Defaults to 0 (first sheet).
        skiprows (int, optional): Number of rows to skip at the beginning of each sheet. Defaults to 0.

    Returns:
        Optional[pd.DataFrame]: A concatenated DataFrame containing data from all Excel files,
                                or None if no files are found or an error occurs.
    """
    logger.info(f"Attempting to load Excel files from folder: {data_folder_path}")
    # Use pathlib for robust path handling and globbing
    excel_files = list(Path(data_folder_path).glob('*.xlsx'))

    if not excel_files:
        logger.warning(f"No Excel files (.xlsx) found in directory: {data_folder_path}")
        return None

    list_of_dfs = []
    logger.info(f"Found {len(excel_files)} Excel files. Target sheet: '{sheet_name}', skiprows: {skiprows}.")

    for i, file_path in enumerate(excel_files, 1):
        logger.debug(f"Processing Excel file {i}/{len(excel_files)}: {file_path.name}")
        try:
            df = pd.read_excel(file_path, engine="openpyxl", sheet_name=sheet_name, skiprows=skiprows)
            list_of_dfs.append(df)
            logger.debug(f"Successfully read sheet '{sheet_name}' from {file_path.name}. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading sheet '{sheet_name}' from file {file_path.name}: {e}", exc_info=True)
            # Decide if one failed file should stop the whole process or just be skipped
            # For now, we'll skip and continue.
    
    if not list_of_dfs:
        logger.error("No data could be loaded from any Excel files found.")
        return None

    try:
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        logger.info(f"Successfully combined data from {len(list_of_dfs)} Excel files. Final DataFrame shape: {combined_df.shape}")
        return combined_df
    except Exception as e:
        logger.error(f"Error concatenating DataFrames from Excel files: {e}", exc_info=True)
        return None


# --- Preprocessing Function for Training Data ---
def preprocess_dataframe_for_training(
    df_raw: pd.DataFrame,
    text_column: str = 'item_name',         # Column containing the product names/text
    target_column: str = 'Fixed Category Code' # Column containing the target category labels
) -> Optional[pd.DataFrame]:
    """
    Preprocesses a raw DataFrame for model training.
    - Handles missing values in the text column.
    - Applies basic cleaning and lemmatization to the text column.
    - Selects and renames relevant columns for training ('processed_text', 'category').
    - Drops rows where processed text is empty or target is missing.

    Args:
        df_raw (pd.DataFrame): The raw input DataFrame.
        text_column (str): The name of the column containing text to be processed.
        target_column (str): The name of the column containing target labels.

    Returns:
        Optional[pd.DataFrame]: A DataFrame ready for training with 'processed_text' and 'category' columns,
                                or None if preprocessing fails critically.
    """
    logger.info(f"Starting preprocessing of DataFrame for training. Initial shape: {df_raw.shape}")
    if df_raw.empty:
        logger.warning("Input DataFrame is empty. Cannot preprocess.")
        return None

    df_processed = df_raw.copy()

    # --- 1. Ensure text and target columns exist ---
    if text_column not in df_processed.columns:
        logger.error(f"Text column '{text_column}' not found in DataFrame.")
        return None
    if target_column not in df_processed.columns:
        logger.error(f"Target column '{target_column}' not found in DataFrame.")
        return None

    # --- 2. Handle missing text data ---
    # Fill NaN in the text column with an empty string to avoid errors in text processing functions.
    df_processed[text_column] = df_processed[text_column].fillna('')
    logger.debug(f"Filled NaN values in '{text_column}' with empty strings.")

    # --- 3. Apply Text Preprocessing (Cleaning and Lemmatization) ---
    # Create a new column for the processed text.
    # This uses the functions from app.ml.preprocessing.
    logger.info(f"Applying text cleaning and lemmatization to '{text_column}'...")
    try:
        df_processed['processed_text'] = df_processed[text_column].apply(
            lambda text: process_text_lemma(basic_clean(str(text)))
        )
    except RuntimeError as e: # Catch errors from preprocessing (e.g. NLTK data missing)
        logger.error(f"RuntimeError during text preprocessing: {e}. Ensure NLTK data is available.", exc_info=True)
        return None # Critical failure
    except Exception as e:
        logger.error(f"Unexpected error during text preprocessing: {e}", exc_info=True)
        return None # Critical failure

    logger.info("Text cleaning and lemmatization applied.")

    # --- 4. Prepare final columns for training ---
    # Select the processed text and the original target column.
    # Rename them to generic names ('processed_text', 'category') for consistency in the training function.
    df_final_for_training = df_processed[['processed_text', target_column]].copy()
    df_final_for_training.rename(columns={target_column: 'category'}, inplace=True)

    # --- 5. Data Cleaning after processing ---
    # Remove rows where 'processed_text' is empty or consists only of whitespace.
    original_len = len(df_final_for_training)
    df_final_for_training = df_final_for_training[df_final_for_training['processed_text'].str.strip().astype(bool)]
    rows_dropped_empty_text = original_len - len(df_final_for_training)
    if rows_dropped_empty_text > 0:
        logger.warning(f"Dropped {rows_dropped_empty_text} rows because 'processed_text' was empty after preprocessing.")

    # Remove rows where the 'category' (target label) is missing (NaN).
    original_len = len(df_final_for_training)
    df_final_for_training.dropna(subset=['category'], inplace=True)
    rows_dropped_empty_category = original_len - len(df_final_for_training)
    if rows_dropped_empty_category > 0:
        logger.warning(f"Dropped {rows_dropped_empty_category} rows due to missing 'category' labels.")

    if df_final_for_training.empty:
        logger.error("DataFrame became empty after preprocessing and cleaning. No data available for training.")
        return None

    logger.info(f"Preprocessing complete. DataFrame ready for training. Shape: {df_final_for_training.shape}")
    logger.debug(f"Sample of training data:\n{df_final_for_training.head()}")
    return df_final_for_training


# --- Model Training and Evaluation Function ---
def train_evaluate_classifier(
    df_ready: pd.DataFrame,
    text_feature_column: str = 'processed_text',
    target_label_column: str = 'category',
    max_tfidf_features: int = config.DEFAULT_MAX_FEATURES,
    svc_c_param: float = config.DEFAULT_C_PARAM,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Optional[LinearSVC], Optional[TfidfVectorizer], Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Trains a TF-IDF + LinearSVC classification model, evaluates it, and returns artifacts.

    Args:
        df_ready (pd.DataFrame): DataFrame with 'processed_text' and 'category' columns.
        text_feature_column (str): Name of the column containing processed text features.
        target_label_column (str): Name of the column containing target labels.
        max_tfidf_features (int): Max features for TfidfVectorizer.
        svc_c_param (float): C parameter for LinearSVC.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[Optional[LinearSVC], Optional[TfidfVectorizer], Optional[List[str]], Optional[Dict[str, Any]]]:
            - Trained LinearSVC model instance.
            - Fitted TfidfVectorizer instance.
            - List of model class names (categories).
            - Dictionary containing evaluation metrics (accuracy, report, confusion_matrix).
            Returns (None, None, None, None) if training fails.
    """
    logger.info(f"Starting model training and evaluation. Max TF-IDF features: {max_tfidf_features}, SVC C: {svc_c_param}.")
    if df_ready.empty or text_feature_column not in df_ready.columns or target_label_column not in df_ready.columns:
        logger.error("Input DataFrame for training is empty or missing required columns.")
        return None, None, None, None

    X = df_ready[text_feature_column].astype(str) # Ensure text features are strings
    y = df_ready[target_label_column]

    # --- Stratified Train-Test Split ---
    # Stratify by 'y' to ensure class proportions are similar in train and test sets,
    # which is important for imbalanced datasets.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Data split into training and testing sets: {len(X_train)} train samples, {len(X_test)} test samples.")
    except ValueError as e:
        logger.error(f"Failed to perform train-test split, possibly due to too few samples in some classes for stratification. Error: {e}", exc_info=True)
        # This can happen if a class has only 1 sample and stratify=y.
        # Further investigation or data augmentation might be needed for such classes.
        return None, None, None, None

    # --- TF-IDF Vectorization ---
    logger.info(f"Initializing TfidfVectorizer with max_features={max_tfidf_features}.")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
    
    logger.info("Fitting TF-IDF vectorizer on training data and transforming training data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    logger.info("Transforming test data using the fitted TF-IDF vectorizer...")
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    logger.info(f"TF-IDF transformation complete. Training TF-IDF matrix shape: {X_train_tfidf.shape}")

    # --- Model Training (LinearSVC) ---
    logger.info(f"Training LinearSVC model with C={svc_c_param}...")
    # dual=False is recommended when n_samples > n_features. For text data, n_features can be large.
    # Consider setting max_iter if convergence warnings appear.
    model = LinearSVC(C=svc_c_param, dual=True, random_state=random_state, max_iter=2000) # Added max_iter, dual='auto'
    try:
        model.fit(X_train_tfidf, y_train)
        logger.info("LinearSVC model training complete.")
    except Exception as e:
        logger.error(f"Error during model fitting: {e}", exc_info=True)
        return None, None, None, None
        
    # --- Model Evaluation ---
    logger.info("Evaluating model on the test set...")
    y_pred = model.predict(X_test_tfidf)
    
    # Get the class labels the model was trained on (important for classification_report)
    model_classes = model.classes_.tolist() # Convert numpy array to list
    
    accuracy = accuracy_score(y_test, y_pred)
    # `zero_division=0` prevents warnings if a class has no true samples in the test set for some metrics.
    class_report = classification_report(y_test, y_pred, labels=model_classes, zero_division=0, output_dict=False) # Get as string
    class_report_dict = classification_report(y_test, y_pred, labels=model_classes, zero_division=0, output_dict=True) # Get as dict for structured metrics
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model_classes)

    logger.info(f"--- Model Evaluation Results ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{class_report}")
    # Optionally, log confusion matrix or save it as an image
    # logger.debug(f"Confusion Matrix:\n{conf_matrix}")

    evaluation_metrics = {
        "accuracy": accuracy,
        "classification_report_str": class_report,
        "classification_report_dict": class_report_dict,
        "confusion_matrix": conf_matrix.tolist(), # Convert numpy array to list for easier serialization if needed
        "test_set_size": len(y_test),
        "training_set_size": len(y_train)
    }
    
    return model, tfidf_vectorizer, model_classes, evaluation_metrics


# --- Artifact Saving Function ---
def save_training_artifacts(
    model: Any, 
    vectorizer: Any, 
    class_labels: List[str], 
    output_directory: str
) -> bool:
    """
    Saves the trained model, TF-IDF vectorizer, and class labels to disk.

    Args:
        model: The trained classification model object.
        vectorizer: The fitted TF-IDF vectorizer object.
        class_labels (List[str]): The list of class names (categories).
        output_directory (str): The directory where artifacts will be saved.
                                (e.g., config.MODEL_STORE_DIR)

    Returns:
        bool: True if all artifacts were saved successfully, False otherwise.
    """
    logger.info(f"Attempting to save training artifacts to directory: {output_directory}")
    
    # Ensure the output directory exists.
    try:
        os.makedirs(output_directory, exist_ok=True) # exist_ok=True means no error if dir already exists
    except OSError as e:
        logger.error(f"Error creating output directory '{output_directory}': {e}", exc_info=True)
        return False

    # Define paths for saving each artifact. These names should match what `model_management.py` expects.
    vectorizer_path = os.path.join(output_directory, "tfidf_vectorizer.joblib")
    model_path = os.path.join(output_directory, "svc_model.joblib")
    classes_path = os.path.join(output_directory, "model_classes.joblib")

    try:
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"TF-IDF vectorizer saved to: {vectorizer_path}")
        
        joblib.dump(model, model_path)
        logger.info(f"Classification model saved to: {model_path}")
        
        joblib.dump(class_labels, classes_path)
        logger.info(f"Model class labels saved to: {classes_path}")
        
        return True
    except Exception as e:
        logger.error(f"An error occurred while saving one or more training artifacts: {e}", exc_info=True)
        return False


# --- Main Orchestrator for Training Pipeline ---
def run_complete_training_pipeline(
    data_input_folder: str = config.TRAINING_DATA_DIR,
    model_output_folder: str = config.MODEL_STORE_DIR,
    training_params: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Orchestrates the entire model training pipeline:
    1. Loads data.
    2. Preprocesses data.
    3. Trains and evaluates the model.
    4. Saves the trained artifacts (model, vectorizer, class labels).

    Args:
        data_input_folder (str): Path to the folder containing raw training data (Excel files).
                                 Defaults to `config.TRAINING_DATA_DIR`.
        model_output_folder (str): Path to the folder where trained model artifacts will be saved.
                                   Defaults to `config.MODEL_STORE_DIR`.
        training_params (Optional[Dict[str, Any]]): Dictionary of training parameters to override defaults.
            Expected keys: 'max_tfidf_features', 'svc_c_param', 'text_column', 'target_column', etc.
            Defaults to None, using parameters from `config.py` or function defaults.

    Returns:
        bool: True if the entire pipeline completed successfully and artifacts were saved,
              False otherwise.
    """
    logger.info("=== Starting Complete Model Training Pipeline ===")

    # --- 1. Load and Combine Data ---
    # Parameters for data loading (can be part of training_params if needed for more flexibility)
    sheet_name_to_load = 0 # Assuming first sheet
    rows_to_skip = 0
    
    df_raw = load_and_combine_excel_data(data_input_folder, sheet_name=sheet_name_to_load, skiprows=rows_to_skip)
    if df_raw is None or df_raw.empty:
        logger.error("Training pipeline failed: Could not load raw data.")
        return False

    # --- 2. Preprocess Data ---
    # Get text and target column names from training_params or use defaults
    params = training_params or {}
    text_col = params.get('text_column', 'item_name') # Default from original train.py
    target_col = params.get('target_column', 'Fixed Category Code') # Default from original train.py

    df_ready_for_training = preprocess_dataframe_for_training(df_raw, text_column=text_col, target_column=target_col)
    if df_ready_for_training is None or df_ready_for_training.empty:
        logger.error("Training pipeline failed: Data preprocessing resulted in no usable data.")
        return False

    # --- 3. Train and Evaluate Model ---
    # Get model training parameters from training_params or use defaults from config/function
    max_features = params.get('max_tfidf_features', config.DEFAULT_MAX_FEATURES)
    c_param = params.get('svc_c_param', config.DEFAULT_C_PARAM)
    
    model, vectorizer, model_classes, eval_metrics = train_evaluate_classifier(
        df_ready=df_ready_for_training,
        max_tfidf_features=max_features,
        svc_c_param=c_param
        # Other parameters like test_size, random_state use defaults from train_evaluate_classifier
    )
    
    if model is None or vectorizer is None or model_classes is None:
        logger.error("Training pipeline failed: Model training or evaluation did not complete successfully.")
        return False
    
    logger.info("Model training and evaluation successful. Metrics summary:")
    logger.info(f"  Accuracy: {eval_metrics.get('accuracy', 'N/A'):.4f}")
    # Log other key metrics if desired

    # --- 4. Save Training Artifacts ---
    if not save_training_artifacts(model, vectorizer, model_classes, model_output_folder):
        logger.error("Training pipeline failed: Could not save all trained artifacts.")
        return False

    logger.info("=== Complete Model Training Pipeline Finished Successfully ===")
    return True
