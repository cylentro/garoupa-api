# app/ml/preprocessing.py

# Import regular expression module for text manipulation
import re

# Import Natural Language Toolkit library
import nltk

# Import specific NLTK modules:
# stopwords: for a list of common words to ignore (e.g., "the", "is")
# wordnet: a lexical database for English, used by the lemmatizer
from nltk.corpus import stopwords, wordnet

# WordNetLemmatizer: for reducing words to their base or dictionary form (lemma)
from nltk.stem import WordNetLemmatizer

# word_tokenize: for splitting text into individual words (tokens)
from nltk.tokenize import word_tokenize

# numpy is not directly used in this version of the file but often imported if POS tagging requires it for array operations.
# We'll keep it commented out unless a future version explicitly needs it here.
# import numpy as np 

# --- NLTK Data Download (Important Note) ---
# The functions below rely on NLTK data packages (like 'wordnet', 'stopwords', 'punkt', 'averaged_perceptron_tagger').
# These packages need to be downloaded once before the functions can be used.
# Your `test.py` script or the `download_nltk_data()` function in `train.py` handles this.
# In a production environment, this download should ideally happen during the setup of the environment
# (e.g., in a Dockerfile or a setup script) to ensure the application runs correctly.
# Example download (can be run in a Python interpreter):
# nltk.download(['wordnet', 'stopwords', 'punkt', 'averaged_perceptron_tagger', 'omw-1.4'])


# --- Initialization of NLTK Components ---
# These components are initialized globally (when the module is first imported)
# to avoid re-initializing them every time a processing function is called, which is inefficient.
try:
    # Load the set of English stopwords.
    # You can extend this, e.g., with Indonesian stopwords: set(stopwords.words('english') + stopwords.words('indonesian'))
    # For this specific file, we'll stick to English as per its previous content.
    stop_words_english = set(stopwords.words('english'))
    # Initialize the WordNet Lemmatizer.
    lemmatizer = WordNetLemmatizer()
except LookupError as e:
    # This block executes if the required NLTK data (stopwords, wordnet) hasn't been downloaded.
    print(f"WARNING: NLTK data (stopwords/wordnet) not found. Error: {e}. Preprocessing might fail or be inaccurate.")
    print("Please ensure NLTK data is downloaded. You might need to run a script like `test.py` or manually download them.")
    # Set to default empty set/None so the module can still load, but functions using them will likely fail or be impaired.
    stop_words_english = set()
    lemmatizer = None
    # Consider raising an exception here if these are absolutely critical for your application to start.
    # raise RuntimeError("Failed to initialize NLTK components due to missing data. Please download them first.") from e


# --- Basic Text Cleaning Function ---
def basic_clean(text: str) -> str:
    """
    Performs basic cleaning operations on a text string.
    - Converts text to lowercase.
    - Removes punctuation and special characters (keeps alphanumeric and whitespace).
    - Removes extra whitespace (strips leading/trailing and reduces multiple spaces to one).
    
    Args:
        text (str): The input text string to clean.

    Returns:
        str: The cleaned text string.
    """

    # Ensure the input is a string and convert to lowercase.
    text = str(text).lower()

    # Remove any characters that are not word characters (alphanumeric) or whitespace.
    # re.UNICODE ensures it works correctly with Unicode characters.
    text = re.sub(r'[^\w\s]+', '', text, flags=re.UNICODE)

    # Optional: Remove digits. Uncomment the line below if you want to remove numbers.
    # text = re.sub(r'\d+', '', text, flags=re.UNICODE)
    # Remove leading and trailing whitespace.
    text = text.strip()

    # Replace multiple whitespace characters with a single space.
    text = re.sub(r'\s+', ' ', text)
    return text

# --- Helper Function for Part-of-Speech (POS) Tagging ---
def get_wordnet_pos(word: str) -> str:
    """
    Maps NLTK's Part-of-Speech (POS) tags to the format accepted by WordNetLemmatizer.
    WordNetLemmatizer's `lemmatize` method can perform more accurate lemmatization
    if it knows the POS tag of the word (e.g., noun, verb, adjective).

    Args:
        word (str): The word for which to get the POS tag.

    Returns:
        str: The WordNet POS tag (e.g., wordnet.NOUN, wordnet.VERB).
               Defaults to wordnet.NOUN if the tag is not recognized or tagger is missing.
    """
    try:
        # Get the POS tag for the word. nltk.pos_tag returns a list of (word, tag) tuples.
        # We take the first (and only) tuple, get the tag, and take the first letter of the tag.
        # Example: 'NNP' (proper noun, singular) -> 'N'
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except LookupError:
        # This error occurs if the 'averaged_perceptron_tagger' data is not downloaded.
        print("WARNING: NLTK's 'averaged_perceptron_tagger' not found. POS tagging will default to NOUN. Lemmatization might be less accurate.")
        return wordnet.NOUN # Default to Noun if POS tagging fails
    except IndexError:
        # This can happen if pos_tag returns an empty list for some reason (e.g. empty word string)
        print(f"WARNING: Could not determine POS tag for word '{word}'. Defaulting to NOUN.")
        return wordnet.NOUN

    # Dictionary to map the first letter of NLTK POS tags to WordNet POS tags.
    tag_dict = {
        "J": wordnet.ADJ,  # Adjective
        "N": wordnet.NOUN, # Noun
        "V": wordnet.VERB, # Verb
        "R": wordnet.ADV   # Adverb
    }
    # Return the corresponding WordNet tag, or wordnet.NOUN if the tag is not in the dictionary.
    return tag_dict.get(tag, wordnet.NOUN)

# --- Main Text Processing Function (with Lemmatization) ---
def process_text_lemma(text: str) -> str:
    """
    Processes a text string by:
    1. Tokenizing it (splitting into words).
    2. Removing stopwords.
    3. Lemmatizing each token (reducing to its base form using its POS tag).

    Args:
        text (str): The input text string.

    Returns:
        str: A string of processed (lemmatized) tokens, joined by spaces.
             Returns an empty string if the lemmatizer is not initialized.
    """

    # Check if the lemmatizer was initialized successfully.
    if lemmatizer is None:
        # This typically means NLTK data was missing at startup.
        print("ERROR: Lemmatizer not initialized. Cannot process text. Please ensure NLTK data is downloaded.")

        # Depending on desired behavior, you could raise an error or return the original text.
        # For now, returning an empty string or original text to avoid crashes, but this indicates a setup problem.
        raise RuntimeError("Lemmatizer not initialized. Download NLTK 'wordnet' and 'omw-1.4'.")

    # Tokenize the text into individual words.
    # word_tokenize requires the 'punkt' NLTK data package.
    try:
        tokens = word_tokenize(str(text)) # Ensure text is a string
    except LookupError as e:
        print(f"ERROR: NLTK 'punkt' tokenizer data not found. Error: {e}. Cannot tokenize text.")
        raise RuntimeError("NLTK 'punkt' tokenizer data not found. Please download it.") from e

    processed_tokens = []
    for token in tokens:
        # Check if the token is not a stopword and has a length greater than 1 (to ignore single characters).
        if token not in stop_words_english and len(token) > 1:
            # Lemmatize the token using its POS tag for better accuracy.
            lemma = lemmatizer.lemmatize(token, get_wordnet_pos(token))
            processed_tokens.append(lemma)
    
    # Join the processed tokens back into a single string, separated by spaces.
    return " ".join(processed_tokens)