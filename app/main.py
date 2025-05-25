# app/main.py
from fastapi import FastAPI, Depends # Add Depends
from contextlib import asynccontextmanager
import logging
import uvicorn

# Application-specific imports
from app.core import config
from app.core.model_management import load_artifacts, are_models_loaded
from app.core.nltk_utils import download_nltk_essential_data
from app.db.database_setup import create_db_tables # Using the function now
# Import your models module to ensure tables are known to Base before create_all
from app.db.models import api_client_model # This ensures ApiClient is registered with Base
from app.core.auth_dependencies import get_current_client_with_swagger_ui_fix

# Import API routers
from app.api import predict_router
from app.api import training_router
from app.api import auth_router # <-- IMPORT AUTH ROUTER

# Import the authentication dependency
from app.core.auth_dependencies import get_current_active_client # <-- IMPORT DEPENDENCY

logging.basicConfig(
    level=getattr(logging, config.LOGGING_LEVEL.upper(), logging.INFO),
    format=config.LOGGING_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Application startup sequence initiated...")
    if not create_db_tables():
        logger.critical("Failed to create/check database tables. The application might not function correctly with client auth.")
    
    logger.info("Checking and potentially downloading NLTK essential data packages...")
    download_nltk_essential_data(config.NLTK_PACKAGES)
    logger.info("NLTK data check/download process completed.")

    logger.info("Loading machine learning model artifacts...")
    if not load_artifacts():
        logger.critical("CRITICAL FAILURE: Failed to load ML artifacts. Prediction service will be unavailable.")
    else:
        logger.info("Machine learning model artifacts loaded successfully.")
    
    logger.info("Application startup sequence completed.")
    yield
    logger.info("Application shutdown sequence initiated...")
    logger.info("Application shutdown sequence completed.")

# --- FastAPI Application Instance Creation ---
app = FastAPI(
    title="Product Categorizer API",
    description="API for predicting product categories, managing model retraining, and secure client authentication.", # Updated
    version="0.5.0", # Incremented version for auth feature
    lifespan=lifespan
)

# --- Include API Routers ---
# The auth_router provides the /auth/token endpoint, which MUST be public (not have the auth dependency).
app.include_router(auth_router.router)

# Protect the prediction and training routers with our JWT validation dependency.
# Any endpoint within these routers will now require a valid Bearer token.
app.include_router(
    predict_router.router,
    dependencies=[Depends(get_current_client_with_swagger_ui_fix)] # Protects all routes in predict_router
)
app.include_router(
    training_router.router,
    dependencies=[Depends(get_current_client_with_swagger_ui_fix)] # Protects all routes in training_router
)

# Root endpoint - remains public as it's not under a protected router or global dependency here.
@app.get("/", tags=["Application Status"])
async def read_root():
    models_status_message = "Models loaded successfully." if are_models_loaded() else "Models NOT loaded."
    return {
        "message": "Welcome to the Product Categorizer API! Use /auth/token to get an access token.",
        "documentation_url": "/docs", # Link to Swagger UI
        "openapi_url": "/openapi.json", # Link to OpenAPI schema
        "model_status": models_status_message
    }

if __name__ == "__main__":
    logger.info("Attempting to run application programmatically with Uvicorn...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOGGING_LEVEL.lower()
    )