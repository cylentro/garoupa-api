# app/schemas/product_schemas.py

# Import BaseModel and Field from Pydantic for creating data validation models.
# BaseModel is the base class for all Pydantic models.
# Field allows for additional validation and metadata for model fields (e.g., examples, min_length).
from pydantic import BaseModel, Field
# Import List from the typing module for type hinting lists.
from typing import List, Optional # Added Optional for potential future use

# --- Request Schemas ---
# These models define the expected structure of data coming INTO the API.

class ProductInput(BaseModel):
    """
    Represents the expected input structure for a product whose category needs to be predicted.
    It expects a single field, 'product_name'.
    """
    # 'product_name': The name of the product to be categorized.
    #   - Type: str (string)
    #   - Constraints:
    #     - ... (ellipsis) indicates that this field is required.
    #     - min_length=1: The product name must have at least 1 character.
    #   - example: Provides a sample value for API documentation (e.g., Swagger UI).
    product_name: str = Field(..., min_length=1, example="Sepatu Lari Pria Nike ZoomX Ultra")

    # You can add more fields here if your API needs more input for prediction, e.g.:
    # description: Optional[str] = Field(None, example="Lightweight running shoes with advanced cushioning.")
    # brand: Optional[str] = Field(None, example="Nike")


# --- Response Schemas ---
# These models define the structure of data being sent OUT FROM the API.

class CategoryPrediction(BaseModel):
    """
    Represents a single category prediction, including the category name and its associated score.
    This is used as part of a list in the main prediction output.
    """
    # 'category': The predicted category name (e.g., "FASHION", "ELECTRONIC").
    #   - Type: str (string)
    category: str = Field(..., example="FASHION") # Added example

    # 'score': The confidence score (or decision function value) associated with this category prediction.
    #   Higher scores generally indicate higher confidence, depending on the model.
    #   - Type: float (floating-point number)
    score: float = Field(..., example=0.85) # Added example

    class Config:
        """
        Pydantic model configuration.
        `from_attributes` (formerly `orm_mode`) allows the model to be created from arbitrary
        class instances (e.g., SQLAlchemy models or other objects) by reading attributes,
        not just from dictionaries. Useful if you ever populate this from non-dict sources.
        """
        from_attributes = True


class PredictionOutput(BaseModel):
    """
    Represents the overall output structure for a product categorization request.
    It includes the top predicted category, its confidence score, and a list of
    the top N predictions.
    """
    # 'predicted_category': The single best category prediction for the product.
    #   - Type: str (string)
    predicted_category: str = Field(..., example="FASHION") # Added example

    # 'confidence_score': The score associated with the 'predicted_category'.
    #   - Type: float (floating-point number)
    confidence_score: float = Field(..., example=0.85) # Added example

    # 'top_predictions': A list of `CategoryPrediction` objects, representing the top N
    #   most likely categories and their scores, ordered from most to least confident.
    #   - Type: List[CategoryPrediction] (a list where each item is a CategoryPrediction object)
    top_predictions: List[CategoryPrediction] = Field(..., example=[
        CategoryPrediction(category="FASHION", score=0.85),
        CategoryPrediction(category="MAINAN", score=0.10)
    ])

    class Config:
        """
        Pydantic model configuration.
        """
        from_attributes = True
        # You could add example data for the whole model for Swagger UI if desired:
        # json_schema_extra = {
        #     "example": {
        #         "predicted_category": "FASHION",
        #         "confidence_score": 0.85,
        #         "top_predictions": [
        #             {"category": "FASHION", "score": 0.85},
        #             {"category": "HOUSEHOLD", "score": 0.10},
        #             {"category": "MAINAN", "score": 0.03}
        #         ]
        #     }
        # }

# You can define other schemas here as needed, for example, for the retraining API's
# request or response, or for health check status, etc.
# Example for a simple status message:
# class StatusMessage(BaseModel):
#     status: str
#     message: Optional[str] = None