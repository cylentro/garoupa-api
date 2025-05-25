# app/api/auth_router.py
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm # Handles form data for token requests
from fastapi import Form # Make sure to import Form
from sqlalchemy.orm import Session
from datetime import timedelta

# Application-specific imports
from app.db.database_setup import get_db_session # Dependency to get DB session
from app.crud import client_crud # CRUD functions for API clients
from app.core.security_utils import create_access_token # JWT creation utility
from app.schemas.auth_schemas import Token # Pydantic schema for the token response
from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES # Token expiration configuration

logger = logging.getLogger(__name__)

# Create an APIRouter instance for authentication-related endpoints.
router = APIRouter(
    prefix="/auth", # All routes in this router will start with "/auth"
    tags=["Authentication"] # Tag for grouping in API documentation
)

@router.post(
    "/token",
    response_model=Token,
    summary="Request Access Token (Client Credentials)",
    description="Client applications use this endpoint to authenticate with their "
                "`client_id` and `client_secret` "
                "to obtain a JWT access token. This endpoint expects 'x-www-form-urlencoded' data, "
                "mimicking an OAuth2 Client Credentials Grant flow."
)
# async def login_for_access_token(
#     form_data: OAuth2PasswordRequestForm = Depends(), # Injects form data: username, password
#     db: Session = Depends(get_db_session) # Injects DB session
# ):
#     """
#     Token endpoint for client authentication.
#     - Expects `client_id` in the `username` field of the form data.
#     - Expects `client_secret` in the `password` field of the form data.
    
#     Validates client credentials and returns a JWT access token upon success.
#     """
#     logger.info(f"Access token request received for client_id (form username): {form_data.username}")

#     # Authenticate the API client using the provided client_id and client_secret.
#     # The `authenticate_api_client` function handles checking the hashed secret and client status.
#     api_client = client_crud.authenticate_api_client(
#         db, client_id=form_data.username, plain_client_secret=form_data.password
#     )

#     # If authentication fails (client not found, secret invalid, or client inactive)
#     if not api_client:
#         logger.warning(f"Authentication failed for client_id (form username): {form_data.username}")
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect client_id or client_secret, or client is inactive.",
#             headers={"WWW-Authenticate": "Bearer"}, 
#         )

#     # If authentication is successful, create an access token (JWT).
#     # The subject of the token will be the client_id.
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token_str = create_access_token(
#         subject=api_client.client_id, expires_delta=access_token_expires
#     )
    
#     logger.info(f"Access token generated successfully for client_id: {api_client.client_id}")
#     return Token(access_token=access_token_str, token_type="bearer")

async def login_for_access_token(
        grant_type: str = Form(..., example="client_credentials", description="client_credentials"), # Client explicitly sends "client_credentials"
        client_id: str = Form(..., example="testclient01", description="testclient01"),
        client_secret: str = Form(..., example="verysecret123", description="verysecret123"),
        scope: str = Form("", example="read write"), # Optional scopes
        db: Session = Depends(get_db_session)
    ):
        logger.info(f"Access token request received. Grant Type: {grant_type}, Client ID: {client_id}")

        # Validate the grant_type
        if grant_type != "client_credentials":
            logger.warning(f"Unsupported grant type: {grant_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, # HTTP 400 Bad Request
                detail="Unsupported grant type. Must be 'client_credentials'."
            )
        
        # Authenticate the API client using the provided client_id and client_secret.
        api_client = client_crud.authenticate_api_client(
            db, client_id=client_id, plain_client_secret=client_secret
        )

        if not api_client:
            logger.warning(f"Authentication failed for client_id: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect client_id or client_secret, or client is inactive.",
                headers={"WWW-Authenticate": "Bearer"}, 
            )

        # If authentication is successful, create an access token (JWT).
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token_str = create_access_token(
            subject=api_client.client_id, expires_delta=access_token_expires
            # You could add 'scopes': scope.split() to the token if you parse and use scopes
        )
        
        logger.info(f"Access token generated successfully for client_id: {api_client.client_id}")
        return Token(access_token=access_token_str, token_type="bearer")
