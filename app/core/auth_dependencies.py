# app/core/auth_dependencies.py
import logging
from typing import Optional
from fastapi import Depends, HTTPException, status, Request, Header # Import Request
from fastapi.security import OAuth2 # Import the base OAuth2 class
from fastapi.security.utils import get_authorization_scheme_param # Import a helper
from sqlalchemy.orm import Session

# ... (other application-specific imports) ...
from app.core.security_utils import decode_access_token 
from app.schemas.auth_schemas import TokenData 
from app.db.database_setup import get_db_session
from app.crud import client_crud 
from app.db.models.api_client_model import ApiClient 

logger = logging.getLogger(__name__)

# --- START: Custom Security Class ---
# This class creates a security scheme for the "Client Credentials" flow
# and includes the logic to extract the bearer token from a request.
class ClientCredentialsBearer(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: Optional[str] = None,
        scopes: Optional[dict] = None,
        auto_error: bool = True,
    ):
        if scopes is None:
            scopes = {}
        # This dictionary defines the clientCredentials flow for the OpenAPI schema.
        flows = {"clientCredentials": {"tokenUrl": tokenUrl, "scopes": scopes}}
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        # This logic extracts the token from the "Authorization: Bearer <token>" header.
        authorization: str = request.headers.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None
        return param
# --- END: Custom Security Class ---


# Now, instantiate our custom class instead of OAuth2PasswordBearer
oauth2_bearer_scheme = ClientCredentialsBearer(tokenUrl="/auth/token")

# NO CHANGES ARE NEEDED in your get_current_active_client function.
# It will work exactly as before.
async def get_current_active_client(
    token: str = Depends(oauth2_bearer_scheme), # This now uses our custom class
    db: Session = Depends(get_db_session)
) -> ApiClient:
    # ... (no changes needed in this function's logic)
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials, token is invalid, or client is unauthorized.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        logger.warning("JWT decoding failed or token is invalid/expired.")
        raise credentials_exception
    
    client_id: Optional[str] = payload.get("sub")
    if client_id is None:
        logger.warning("Token payload is missing the 'sub' (subject/client_id) claim.")
        raise credentials_exception
    
    api_client = client_crud.get_api_client_by_client_id(db, client_id=client_id)
    if api_client is None:
        logger.warning(f"Client with client_id '{client_id}' (from token subject) not found in the database.")
        raise credentials_exception
    
    if not api_client.is_active:
        logger.warning(f"Client '{api_client.client_id}' (from token) is marked as inactive.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Client account is inactive."
        )
    
    logger.debug(f"Token validated successfully. Active client: {api_client.client_id}")
    return api_client


async def get_current_client_with_swagger_ui_fix(
    # This part adds the editable header to the Swagger UI for all endpoints using this dependency
    x_authorization: str = Header(
        None,
        alias="Authorization",
        description="Format 'Bearer <token>'"
    ),
    # This part reuses our original dependency to perform the actual security check
    current_client: ApiClient = Depends(get_current_active_client)
) -> ApiClient:
    """
    A wrapper dependency that applies a fix for the Swagger UI's disabled 'Authorize'
    header input while still enforcing the real authentication check.
    """
    # If the `get_current_active_client` dependency fails, it will raise an
    # exception before this point. If it succeeds, we just return the client it found.
    return current_client