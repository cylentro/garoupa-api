# app/crud/client_crud.py
import logging
from typing import Optional, List
from sqlalchemy.orm import Session # For type hinting the database session

# Import the ApiClient model and the secret hashing utility
from app.db.models.api_client_model import ApiClient
from app.core.security_utils import get_secret_hash, verify_secret
# Import Pydantic schemas if needed for data creation/update, though not strictly for basic CRUD here yet
# from app.schemas.client_schemas import ApiClientCreate # Example if you define creation schemas

logger = logging.getLogger(__name__)

def get_api_client_by_client_id(db: Session, client_id: str) -> Optional[ApiClient]:
    """
    Retrieves an API client from the database by its client_id.

    Args:
        db (Session): The SQLAlchemy database session.
        client_id (str): The client_id to search for.

    Returns:
        Optional[ApiClient]: The ApiClient object if found, None otherwise.
    """
    logger.debug(f"Attempting to retrieve API client with client_id: {client_id}")
    try:
        return db.query(ApiClient).filter(ApiClient.client_id == client_id).first()
    except Exception as e:
        logger.error(f"Error retrieving client by client_id '{client_id}': {e}", exc_info=True)
        return None

def get_api_client_by_name(db: Session, client_name: str) -> Optional[ApiClient]:
    """
    Retrieves an API client from the database by its client_name.

    Args:
        db (Session): The SQLAlchemy database session.
        client_name (str): The client_name to search for.

    Returns:
        Optional[ApiClient]: The ApiClient object if found, None otherwise.
    """
    logger.debug(f"Attempting to retrieve API client with client_name: {client_name}")
    try:
        return db.query(ApiClient).filter(ApiClient.client_name == client_name).first()
    except Exception as e:
        logger.error(f"Error retrieving client by client_name '{client_name}': {e}", exc_info=True)
        return None

def create_api_client(db: Session, client_name: str, client_id: str, plain_client_secret: str, is_active: bool = True) -> Optional[ApiClient]:
    """
    Creates a new API client in the database.
    The client_secret will be hashed before storing.

    Args:
        db (Session): The SQLAlchemy database session.
        client_name (str): A descriptive name for the client.
        client_id (str): The public unique identifier for the client.
        plain_client_secret (str): The plain-text secret for the client.
        is_active (bool, optional): Whether the client is active. Defaults to True.

    Returns:
        Optional[ApiClient]: The created ApiClient object if successful, None otherwise.
    """
    logger.info(f"Attempting to create new API client: {client_name} (ID: {client_id})")
    try:
        # Check if client_id or client_name already exists
        if get_api_client_by_client_id(db, client_id):
            logger.warning(f"API client with client_id '{client_id}' already exists. Creation aborted.")
            return None # Or raise an exception
        if get_api_client_by_name(db, client_name): # Ensure name is also unique if desired by business logic
            logger.warning(f"API client with client_name '{client_name}' already exists. Creation aborted.")
            return None


        hashed_secret = get_secret_hash(plain_client_secret)
        db_client = ApiClient(
            client_name=client_name,
            client_id=client_id,
            hashed_client_secret=hashed_secret,
            is_active=is_active
        )
        db.add(db_client)
        db.commit()
        db.refresh(db_client) # Refresh to get DB-generated values like ID, created_at
        logger.info(f"Successfully created API client: {client_name} (ID: {db_client.id})")
        return db_client
    except Exception as e:
        db.rollback() # Rollback in case of error during commit
        logger.error(f"Error creating API client '{client_name}': {e}", exc_info=True)
        return None

def authenticate_api_client(db: Session, client_id: str, plain_client_secret: str) -> Optional[ApiClient]:
    """
    Authenticates an API client using its client_id and plain_client_secret.

    Args:
        db (Session): The SQLAlchemy database session.
        client_id (str): The client's public identifier.
        plain_client_secret (str): The client's plain-text secret.

    Returns:
        Optional[ApiClient]: The authenticated and active ApiClient object if credentials are valid,
                             None otherwise.
    """
    client = get_api_client_by_client_id(db, client_id)
    if not client:
        logger.warning(f"Authentication failed: Client with client_id '{client_id}' not found.")
        return None
    if not client.is_active:
        logger.warning(f"Authentication failed: Client '{client_id}' is inactive.")
        return None
    if not verify_secret(plain_client_secret, client.hashed_client_secret):
        logger.warning(f"Authentication failed: Invalid secret for client_id '{client_id}'.")
        return None
    
    logger.info(f"Client '{client_id}' authenticated successfully.")
    return client

# You can add update and delete functions here as needed:
# def update_api_client(...)
# def delete_api_client(...)