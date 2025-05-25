# app/core/nltk_utils.py
import nltk
import logging
from app.core.config import NLTK_PACKAGES # Assuming you add this to config.py

logger = logging.getLogger(__name__)

# Define the required NLTK packages and their resource IDs/names
# This could also be moved to config.py if you prefer
DEFAULT_NLTK_PACKAGES = {
    'wordnet': 'corpora/wordnet',  # For lemmatization
    'omw-1.4': 'corpora/omw-1.4', # Open Multilingual Wordnet, often needed with wordnet
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger', # For POS tagging
    'stopwords': 'corpora/stopwords', # For stopword lists
    'punkt': 'tokenizers/punkt'     # For word_tokenize
}

def download_nltk_essential_data(packages_to_download: dict = None):
    """
    Checks for and downloads essential NLTK data packages if they are not found.

    Args:
        packages_to_download (dict, optional): A dictionary where keys are NLTK package
                                               names (used for nltk.download()) and values
                                               are their resource locators (used for nltk.data.find()).
                                               Defaults to DEFAULT_NLTK_PACKAGES.
    """
    if packages_to_download is None:
        packages_to_download = DEFAULT_NLTK_PACKAGES

    all_available = True
    logger.info("Checking for NLTK essential data packages...")

    for pkg_name, resource_id in packages_to_download.items():
        try:
            # Check if the resource is already available
            nltk.data.find(resource_id)
            logger.info(f"NLTK package '{pkg_name}' ({resource_id}) is already available.")
        except LookupError:
            logger.warning(f"NLTK package '{pkg_name}' ({resource_id}) not found. Attempting to download...")
            try:
                nltk.download(pkg_name, quiet=False) # quiet=False to see download progress/errors
                # Verify again after attempting download
                nltk.data.find(resource_id)
                logger.info(f"NLTK package '{pkg_name}' successfully downloaded and verified.")
            except Exception as e:
                logger.error(f"Failed to download or verify NLTK package '{pkg_name}'. Error: {e}", exc_info=True)
                all_available = False
        except Exception as e:
            logger.error(f"An unexpected error occurred while checking NLTK package '{pkg_name}': {e}", exc_info=True)
            all_available = False

    if all_available:
        logger.info("All required NLTK data packages are available.")
    else:
        logger.error("One or more required NLTK data packages could not be made available. "
                     "Preprocessing functionality may be impaired or fail. "
                     "Check logs for details and ensure internet connectivity if downloads are needed.")
    # Note: If a download fails, the application will still start,
    # but preprocessing might raise errors later. You could choose to
    # raise an exception here to prevent the app from starting if NLTK data is critical.
    # if not all_available:
    #     raise RuntimeError("Failed to prepare all necessary NLTK data. Application cannot start.")