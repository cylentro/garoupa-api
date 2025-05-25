# app/schemas/training_schemas.py

# Import BaseModel from Pydantic for creating data validation models.
from pydantic import BaseModel
# Import Optional from typing for fields that might not always be present.
from typing import Optional

class RetrainTriggerResponse(BaseModel):
    """
    Defines the response structure when a model retraining process is successfully triggered.
    Indicates that the process has started in the background.
    """
    # 'message': A confirmation message to the user.
    message: str = "Model retraining process has been initiated in the background."
    
    # 'details': Optional field for any additional details or a job ID in more complex setups.
    details: Optional[str] = None

    # Example usage for documentation:
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Model retraining process has been initiated in the background. Check server logs for progress.",
                "details": "Process ID: <some_id_if_applicable>"
            }
        }

# You could also define schemas for:
# - Request body if the retraining API needs parameters (e.g., specific dataset version to use).
# - More detailed status responses if you implement a way to query retraining job status.
# class RetrainStatusResponse(BaseModel):
#     job_id: str
#     status: str # e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED"
#     start_time: Optional[datetime] = None
#     end_time: Optional[datetime] = None
#     message: Optional[str] = None