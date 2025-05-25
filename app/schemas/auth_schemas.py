# app/schemas/auth_schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class Token(BaseModel):
    """
    Pydantic model representing the access token response.
    This is what clients receive after successful authentication.
    """
    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...") # Ellipsis for actual token
    token_type: str = Field(default="bearer", example="bearer") # Standard token type

    class Config:
        from_attributes = True # Allows creating from ORM models or other attribute-based objects
        json_schema_extra = { # Provides an example for OpenAPI documentation
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0Y2xpZW50IiwiZXhwIjoxNzA0MDY3MjAwfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
                "token_type": "bearer"
            }
        }

class TokenData(BaseModel):
    """
    Pydantic model representing the data extracted from a validated JWT payload.
    This typically contains the subject of the token (e.g., client_id).
    """
    subject: Optional[str] = Field(None, alias="sub", example="your_client_id") # 'sub' is standard JWT claim for subject

    # You can add other claims you expect to extract from the token here
    # For example:
    # scopes: List[str] = []

# Schema for the token request (using form data, so not strictly needed here as Pydantic model
# if using FastAPI's OAuth2PasswordRequestForm, but good for clarity if building manually)
# class TokenRequestForm(BaseModel):
#     grant_type: str = Field(None, example="client_credentials") # For client credentials flow
#     client_id: str = Field(..., example="your_client_id")
#     client_secret: str = Field(..., example="your_client_secret")
#     # scope: str = "" # Optional scopes