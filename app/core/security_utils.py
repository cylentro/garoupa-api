# app/core/security_utils.py
import logging
from passlib.context import CryptContext # For hashing and verifying passwords/secrets

logger = logging.getLogger(__name__)

# Configure CryptContext for password hashing.
# - schemes=["bcrypt"]: Specifies bcrypt as the hashing algorithm. Bcrypt is strong and widely recommended.
# - deprecated="auto": Handles deprecated hash formats automatically if you ever change schemes.
# We will use this for hashing client secrets.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_secret(plain_secret: str, hashed_secret: str) -> bool:
    """
    Verifies a plain secret against its hashed version.

    Args:
        plain_secret (str): The plain-text secret to verify.
        hashed_secret (str): The stored hashed version of the secret.

    Returns:
        bool: True if the plain secret matches the hashed secret, False otherwise.
    """
    try:
        return pwd_context.verify(plain_secret, hashed_secret)
    except Exception as e:
        # Log unexpected errors during verification, but still return False for security.
        logger.error(f"Error during secret verification: {e}", exc_info=True)
        return False

def get_secret_hash(secret: str) -> str:
    """
    Hashes a plain secret using the configured CryptContext (bcrypt).

    Args:
        secret (str): The plain-text secret to hash.

    Returns:
        str: The hashed version of the secret.
    """
    return pwd_context.hash(secret)

# JWT related functions will also go here or in a dedicated jwt_utils.py later.
# For now, let's focus on secret hashing.
# We will add create_access_token and token decoding/validation logic here next.

from datetime import datetime, timedelta, timezone # Added timezone
from typing import Optional, Union, Any
import jwt # From python-jose
from app.core.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

def create_access_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a new JWT access token.

    Args:
        subject (Union[str, Any]): The subject of the token (e.g., client_id or username).
        expires_delta (Optional[timedelta], optional): The lifespan of the token.
            If None, uses default from config.ACCESS_TOKEN_EXPIRE_MINUTES. Defaults to None.

    Returns:
        str: The encoded JWT access token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Data to be encoded in the token payload
    to_encode = {
        "exp": expire,      # Expiration time
        "sub": str(subject), # Subject of the token (e.g., client_id)
        "iat": datetime.now(timezone.utc) # Issued at time
    }
    # Add any other claims you might need, e.g., "scopes": ["read", "write"]

    try:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error encoding JWT: {e}", exc_info=True)
        raise  # Re-raise the exception as token creation is critical


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes and validates a JWT access token.

    Args:
        token (str): The JWT token string to decode.

    Returns:
        Optional[dict]: The decoded token payload (claims) if the token is valid,
                        None otherwise.
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_aud": False} # No audience verification by default for simple tokens
        )
        # You could add more validation here, e.g., checking if the subject (client_id) exists
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token validation failed: Expired signature.")
        return None # Or raise a specific exception
    except jwt.JWTError as e: # Catches various JWT errors (invalid signature, malformed, etc.)
        logger.warning(f"Token validation failed: Invalid JWT. Error: {e}")
        return None # Or raise a specific exception
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {e}", exc_info=True)
        return None