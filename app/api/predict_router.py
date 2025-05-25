# app/api/predict_router.py

# Import APIRouter from FastAPI to create a new router object.
# Routers help organize API endpoints into logical groups or modules.
# HTTPException is used to return standard HTTP error responses.
# Depends is used for FastAPI's dependency injection system.
from fastapi import APIRouter, HTTPException, Depends, Header
# Import 'logging' for recording request handling information and errors.
import logging

# Import Pydantic schemas that define the structure of the request (ProductInput)
# and the response (PredictionOutput).
from app.schemas.product_schemas import ProductInput, PredictionOutput
# Import the prediction_service module, which contains the core business logic
# for generating predictions. We import it as 'prediction_service' for clarity.
from app.services import prediction_service
# Import the model_management module to check if the ML models are loaded.
# This is used as a dependency to ensure the service is ready.
from app.core import model_management

# Get a logger instance specific to this module.
logger = logging.getLogger(__name__)

# Create an APIRouter instance.
# All routes defined with this 'router' object will be part of this router.
# - prefix: An optional path prefix for all routes in this router.
#           For example, if prefix="/api/v1/predict", then a route ""
#           defined below would be accessible at "/api/v1/predict".
#           Here, it's "/predict", so a route "" becomes "/predict".
# - tags: A list of tags used for grouping endpoints in the API documentation (e.g., Swagger UI).
router = APIRouter(
    prefix="/predict",
    tags=["Product Prediction"] # Changed tag for better grouping
)

# --- Dependency for Checking Model Readiness ---
async def check_models_are_ready():
    """
    A FastAPI dependency that checks if the essential ML models are loaded.
    If models are not loaded, it raises an HTTPException (503 Service Unavailable),
    preventing the endpoint from processing the request.
    This ensures the API doesn't try to make predictions with an uninitialized model.
    """
    # Call the function from model_management to check the status of loaded artifacts.
    if not model_management.are_models_loaded():
        logger.error("Prediction endpoint called, but essential ML models are not loaded or ready.")
        # Raise an HTTP 503 error if models are not available.
        raise HTTPException(
            status_code=503, # HTTP 503 Service Unavailable
            detail="Model Service Unavailable: The prediction models are not yet loaded or are improperly configured. Please try again later."
        )
    # If models are loaded, the dependency allows the request to proceed.
    # No explicit return value is needed for a successful dependency pass-through.


# --- Prediction Endpoint Definition ---
# Define a POST endpoint at the router's root (which will be '/predict' due to the router's prefix).
# - response_model: Specifies the Pydantic model (PredictionOutput) to be used for
#                   serializing the response data and for API documentation.
#                   FastAPI will validate that the returned data conforms to this model.
@router.post(
    "", # The path relative to the router's prefix (e.g., "/predict" + "" = "/predict")
    response_model=PredictionOutput,
    summary="Predict Product Category", # Summary for API documentation
    description="Receives a product name and returns the predicted category "
                "along with top N predictions and their confidence scores." # Detailed description
)
async def predict_category_endpoint(
    item: ProductInput, # The request body, validated against the ProductInput Pydantic model.
                        # FastAPI automatically parses the JSON request body into this model.
    models_ready: bool = Depends(check_models_are_ready) # Dependency injection:
                                                         # The `check_models_are_ready` function will be executed
                                                         # before this endpoint logic. If it raises an exception,
                                                         # the endpoint will not run.
                                                         # The return value of the dependency (if any) can be used,
                                                         # but here it's primarily for the side effect of validation.
):
    """
    API endpoint to predict the category of a given product name.

    It expects a JSON request body with a "product_name" field.
    It returns a JSON response with the predicted category, confidence score,
    and a list of top alternative predictions.

    The actual prediction logic is delegated to the `prediction_service`.
    """
    logger.info(f"Received prediction request for product: '{item.product_name}' via API endpoint.")

    # --- Delegate to the Prediction Service ---
    # Call the `get_category_predictions` function from the `prediction_service` module
    # to perform the actual prediction logic.
    # This keeps the router (controller) thin and focused on request/response handling.
    predicted_cat, confidence, top_predictions = \
        prediction_service.get_category_predictions(item.product_name)

    # --- Handle Service Response and Potential Errors ---
    # Check if the service returned an error state (indicated by predicted_cat being None
    # and no top_predictions, which our service does for critical internal errors).
    if predicted_cat is None and not top_predictions:
        logger.error(f"Prediction service failed to return a valid prediction for product: '{item.product_name}'.")
        # Return a generic 500 Internal Server Error if the service indicates a failure.
        raise HTTPException(
            status_code=500, # HTTP 500 Internal Server Error
            detail="An internal server error occurred while trying to predict the category. Please try again later."
        )

    # Handle the specific case where the input became empty after preprocessing,
    # as indicated by the `prediction_service`.
    if predicted_cat == "UNKNOWN_EMPTY_INPUT":
        logger.warning(f"Product name '{item.product_name}' resulted in an empty string after preprocessing. No prediction possible.")
        # Return a 400 Bad Request or 422 Unprocessable Entity error.
        # 400 might be more appropriate as the input, though valid in format, is unusable.
        raise HTTPException(
            status_code=400, # HTTP 400 Bad Request
            detail="Input product name became empty after preprocessing. Please provide a more descriptive product name."
        )
    
    # Handle cases where no predictions could be generated by the model,
    # but it wasn't a critical server-side error (e.g., model returned no confident results).
    # The service might return (None, None, []) or (SomeCat, SomeScore, []) but if top_predictions is empty here, it's an issue.
    if not top_predictions: # If the list is empty, even if predicted_cat might have a value
        logger.warning(f"No top predictions were generated by the service for product: '{item.product_name}'.")
        # This could be a 404 if no category fits, or 500 if it's unexpected.
        # Given the logic, if top_predictions is empty, something is unusual.
        raise HTTPException(
            status_code=422, # HTTP 422 Unprocessable Entity (semantically incorrect input for this model)
            detail="The product name could not be categorized with sufficient confidence or input was not specific enough."
        )

    # --- Construct and Return Successful Response ---
    # If predictions are successful, construct the PredictionOutput Pydantic model.
    # This ensures the response adheres to the defined schema.
    logger.info(f"Successfully generated prediction for '{item.product_name}'. Top category: '{predicted_cat}'.")
    return PredictionOutput(
        predicted_category=predicted_cat, # This should always have a value if top_predictions is not empty
        confidence_score=confidence if confidence is not None else 0.0, # Ensure float, even if service returns None
        top_predictions=top_predictions
    )

# You can add more prediction-related endpoints to this router if needed in the future.
# For example, an endpoint for batch predictions:
# @router.post("/batch", response_model=List[PredictionOutput], summary="Predict Categories for a Batch of Products")
# async def predict_batch_categories_endpoint(items: List[ProductInput], models_ready: bool = Depends(check_models_are_ready)):
#     results = []
#     for item in items:
#         # Simplified: In reality, you might want to optimize batch processing in the service layer
#         predicted_cat, confidence, top_preds = prediction_service.get_category_predictions(item.product_name)
#         if predicted_cat and top_preds: # Basic check
#             results.append(PredictionOutput(predicted_category=predicted_cat, confidence_score=confidence, top_predictions=top_preds))
#         else:
#             # Handle errors for individual items in batch
#             results.append(PredictionOutput(predicted_category="ERROR", confidence_score=0.0, top_predictions=[])) # Placeholder for error
#     return results