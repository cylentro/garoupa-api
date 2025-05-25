# app/db/database_setup.py
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import SQLALCHEMY_DATABASE_URL

logger = logging.getLogger(__name__)

# Initialize engine and SessionLocal to None at the module level
# This ensures they always exist in the module's namespace for import,
# even if their setup fails.
engine = None
SessionLocal = None
Base = declarative_base() # Base can be defined regardless

# --- Try to Create Engine ---
try:
    engine_args = {}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
        engine_args["connect_args"] = {"check_same_thread": False}
    
    engine = create_engine(SQLALCHEMY_DATABASE_URL, **engine_args)
    logger.info(f"SQLAlchemy engine created for database: {SQLALCHEMY_DATABASE_URL}")

    # --- Try to Create SessionLocal (only if engine was successful) ---
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("SQLAlchemy SessionLocal created and bound to engine.")

except Exception as e:
    logger.error(f"Failed to initialize SQLAlchemy engine or SessionLocal: {e}", exc_info=True)
    # `engine` and `SessionLocal` will remain None if an error occurred here.

# --- Dependency to get a DB session ---
def get_db_session():
    """
    FastAPI dependency that provides a database session for a single request.
    It ensures the session is properly closed after the request is finished,
    even if an error occurs.
    """
    if SessionLocal is None:
        logger.critical("Database session factory (SessionLocal) is not initialized due to an earlier error. Cannot provide DB session.")
        # This is a critical failure; the application cannot function without DB sessions if they are required.
        raise RuntimeError("Database session factory (SessionLocal) is not initialized. Check database connection and setup.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Function to create tables (can be called from main.py) ---
def create_db_tables():
    """Creates all tables defined by models inheriting from Base."""
    if engine is None:
        logger.error("Database engine not initialized. Cannot create tables.")
        return False # Indicate failure
    
    logger.info("Attempting to create database tables if they do not exist...")
    try:
        Base.metadata.create_all(bind=engine) # This line needs all your models to be imported somewhere before it runs
                                              # so Base knows about them.
        logger.info("Database tables checked/created successfully.")
        return True # Indicate success
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        return False # Indicate failure