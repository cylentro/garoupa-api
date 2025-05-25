# app/api/training_router.py

# Import APIRouter for creating API endpoint groups.
# BackgroundTasks to run long processes like training without blocking the API.
# Depends for dependency injection (though not used directly in the endpoint here, could be for auth).
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header

# Import the Pydantic schema for the response.
from app.schemas.training_schemas import RetrainTriggerResponse
# Import the service function that handles the retraining logic.
from app.services import training_service
# Import logging for this module.
import logging

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

# Create an APIRouter instance for training-related endpoints.
# - prefix: All routes in this router will start with "/training".
# - tags: Groups these endpoints under "Model Training" in the API documentation.
router = APIRouter(
    prefix="/training",
    tags=["Model Training"]
)

# --- Placeholder for Authentication Dependency (Important for Production) ---
# In a real application, an endpoint that triggers retraining should be protected.
# async def get_current_admin_user(credentials: HTTPBasicCredentials = Depends(security_scheme)):
#     # Replace with actual authentication logic
#     if not (credentials.username == "admin" and credentials.password == "secret"): # Example
#         raise HTTPException(status_code=401, detail="Not authenticated")
#     return credentials.username

@router.post(
    "/retrain",
    response_model=RetrainTriggerResponse,
    status_code=202, # HTTP 202 Accepted: The request has been accepted for processing,
                     # but the processing has not been completed.
    summary="Trigger Model Retraining",
    description="Initiates the model retraining process in the background. "
                "The actual training might take some time. "
                "Check server logs for progress and completion status. "
                "This endpoint should be protected in a production environment."
)
async def trigger_model_retraining_endpoint(
    background_tasks: BackgroundTasks
    # For a production system, add authentication:
    # current_user: str = Depends(get_current_admin_user) # Example dependency
):
    """
    API endpoint to trigger the machine learning model retraining pipeline.

    When called, this endpoint adds the retraining task to a background queue
    and returns an immediate confirmation response. The actual training happens
    asynchronously.
    """
    logger.info(
        "'/training/retrain' endpoint called. Adding model retraining to background tasks."
        # In a real app, you'd log which user triggered this: f"by user {current_user}"
    )

    # Add the `execute_model_retraining_pipeline` function from the training_service
    # to be run in the background. This prevents the API request from hanging
    # while the (potentially long) training process executes.
    background_tasks.add_task(training_service.execute_model_retraining_pipeline_integrated)

    # Return an immediate response indicating the process has started.
    return RetrainTriggerResponse(
        message="Model retraining process has been successfully initiated in the background. "
                "Please monitor server logs for detailed progress and completion status."
    )

# Further endpoints could be added here, e.g., to check the status of a training job
# if the system is expanded to support more detailed job tracking.
# @router.get("/retrain/status/{job_id}")
# async def get_retraining_status(job_id: str):
#     # Logic to retrieve and return status of job_id
#     pass