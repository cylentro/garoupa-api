# Garoupa-API

Garoupa-API is a secure, high-performance RESTful API built with FastAPI to categorize products using a machine learning model. This service provides endpoints for prediction and model retraining, secured by OAuth2 Client Credentials.

## Features

* **ML-Powered Prediction** : Categorize products based on their names using a trained machine learning model.
* **On-Demand Retraining** : Trigger a model retraining process via a secure API endpoint to update the model with new data.
* **Secure Authentication** : API access is protected using the OAuth2 Client Credentials grant flow, requiring clients to authenticate with a `client_id` and `client_secret` to obtain a JWT access token.
* **Interactive Documentation** : Automatic, interactive API documentation provided by FastAPI through Swagger UI and ReDoc.
* **Scalable Architecture** : A clean, modular structure separating API logic, services, machine learning code, and data access.

## API Endpoints

| Endpoint        | Method   | Description                                            | Protected |
| :-------------- | :------- | :----------------------------------------------------- | :-------- |
| `/auth/token` | `POST` | Authenticates a client and returns a JWT Bearer token. | No        |
| `/predict/`   | `POST` | Predicts the category for a single product name.       | Yes       |
| `/training/`  | `POST` | Initiates a model retraining pipeline.                 | Yes       |
| `/docs`       | `GET`  | Serves the interactive Swagger UI documentation.       | No        |
| `/redoc`      | `GET`  | Serves the ReDoc documentation.                        | No        |

Export to Sheets

## Tech Stack

* **Backend** : FastAPI
* **Server** : Uvicorn
* **Database** : SQLAlchemy with SQLite
* **Authentication** : JWT (JSON Web Tokens), OAuth2 (Client Credentials)
* **Machine Learning** : Scikit-learn, NLTK
* **Data Validation** : Pydantic

## Prerequisites

* Python 3.9+
* `pip` for package management

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository**

**Bash**

```
git clone <your-repository-url>
cd garoupa-api
```

**2. Create and Activate a Virtual Environment**
It's highly recommended to use a virtual environment to manage project dependencies.

* **macOS / Linux**
  **Bash**

  ```
  python3 -m venv venv
  source venv/bin/activate
  ```
* **Windows**
  **Bash**

  ```
  python -m venv venv
  .\venv\Scripts\activate
  ```

**3. Install^^ Dependencies**
Install all required Python packages from the `requirements.txt` file.

**Bash**

```
pip install -r requirements.txt
```

**4. Run the Initial Model Training**
Before you can make predictions, you need to train the initial machine learning model. The project includes a script for this purpose.

**Bash**

```
python train.py
```

This will create the necessary model artifacts in the `ml_model_artifacts/` directory.

**5. Create Your First API Client**
The API is protected, so you need to create credentials to access it. Use the `manage_clients.py` script to add a new client to the database.

**Bash**

```
# Syntax: python scripts/manage_clients.py add <client_name> <client_id> <client_secret>
python scripts/manage_clients.py add "My Test App" "testclient01" "supersecret123"
```

**Important:** Save the `client_id` and `client_secret` in a safe place. You will need them to get an access token.

## Running the Application

Once the setup is complete, you can run the API server using Uvicorn. The application is defined in `app/main.py`.

**Bash**

```
uvicorn app.main:app --reload
```

The server will start, and the API will be available at `http://127.0.0.1:8000`.

## Usage Examples (with `curl`)

Here’s how to interact with the API using `curl`.

### 1. Get an Access Token

First, exchange your client credentials for a JWT Bearer token from the `/auth/token` endpoint.

**Bash**

```
# Replace with the client_id and client_secret you created
CLIENT_ID="testclient01"
CLIENT_SECRET="supersecret123"

curl -X POST "http://127.0.0.1:8000/auth/token" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=client_credentials&client_id=${CLIENT_ID}&client_secret=${CLIENT_SECRET}"
```

**The**^^ response will be a JSON object containing your `access_token`.

**JSON**

```
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Predict a Product Category

Use the `access_token` from the previous step to call the protected `/predict` endpoint.

**Bash**

```
# Copy the access_token from the previous step
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X POST "http://127.0.0.1:8000/predict/" \
-H "Authorization: Bearer ${TOKEN}" \
-H "Content-Type: application/json" \
-d '{
  "product_name": "susu ultra coklat 250ml"
}'
```

The API will return the predicted category.

**JSON**

```
{
  "predicted_category": "Susu",
  "prediction_score": 0.987,
  "model_version": "v1.0.0"
}
```

### 3. Trigger Model Retraining

You can also trigger a model retraining job by calling the `/training` endpoint.

**Bash**

```
# Use the same TOKEN
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X POST "http://127.0.0.1:8000/training/" \
-H "Authorization: Bearer ${TOKEN}" \
-H "Content-Type: application/json" \
-d '{
  "new_data_path": "data/new_training_data.csv"
}'
```

The API will respond with a confirmation that the training has started.

**JSON**

```
{
  "message": "Training job started successfully.",
  "new_model_version": "v1.0.1",
  "data_path": "data/new_training_data.csv"
}
```

## Project Structure

```
garoupa-api/
├── app/
│   ├── api/          # API endpoint routers (predict, train, auth)
│   ├── core/         # Core logic (config, security, model management)
│   ├── crud/         # CRUD operations for the database
│   ├── db/           # Database setup and models
│   ├── ml/           # Machine learning code (preprocessing, training)
│   ├── schemas/      # Pydantic models for data validation
│   ├── services/     # Business logic for services (prediction, training)
│   └── main.py       # Main FastAPI application entrypoint
├── data/             # Training data files
├── ml_model_artifacts/ # Saved ML models and vectorizers (ignored by git)
├── scripts/          # Helper scripts (e.g., manage_clients.py)
├── .gitignore        # Files and folders ignored by git
├── requirements.txt  # Project dependencies
├── train.py          # Script to run the initial training
└── README.md         # This file
```