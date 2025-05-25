# Base Python image
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a small script to download NLTK data, or run directly
# Option A: Run directly
# RUN python -m nltk.downloader -d /usr/share/nltk_data wordnet omw-1.4 averaged_perceptron_tagger stopwords punkt

# Option B: Using a script (e.g., setup_nltk.py)
COPY setup_nltk.py setup_nltk.py
RUN python setup_nltk.py

# Tell NLTK where to find the data (if downloaded to a non-default location)
ENV NLTK_DATA /usr/share/nltk_data

COPY ./app /app/app
# Copy other necessary files like models_store, train.py etc.
COPY ./models_store /app/models_store
COPY ./data /app/data
# COPY train.py /app/train.py # If needed inside the container

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]