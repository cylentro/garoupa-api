# scripts/manage_clients.py
import os
import sys
import argparse
import logging
from uuid import uuid4 # For generating client IDs/secrets if needed

# --- Adjust Python Path to Find 'app' Package ---
# This allows the script to import modules from the 'app' package when run from the project root.
PROJECT_ROOT_FOR_SCRIPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Navigates up to project root
if PROJECT_ROOT_FOR_SCRIPT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_SCRIPT)

# Now import application components
try:
    from app.db.database_setup import SessionLocal, engine, Base
    from app.db.models.api_client_model import ApiClient # Ensure tables are known
    from app.crud import client_crud
    from app.core.config import SQLALCHEMY_DATABASE_URL # To know which DB we are talking to
except ImportError as e:
    print(f"Error: Could not import necessary application modules. Details: {e}")
    print("Ensure this script is run from the project root or PYTHONPATH is correctly set.")
    sys.exit(1)

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - SCRIPT - %(message)s')
logger = logging.getLogger("manage_clients_script")

def initialize_db():
    """Creates database tables if they don't exist."""
    logger.info(f"Initializing database and creating tables if not exist for: {SQLALCHEMY_DATABASE_URL}")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        raise

def add_client(client_name: str, client_id: str, client_secret: str, is_active: bool = True):
    """Adds a new API client to the database."""
    db = SessionLocal()
    try:
        logger.info(f"Attempting to add client: Name='{client_name}', ID='{client_id}'")
        # Check if client_id already exists
        existing_client_by_id = client_crud.get_api_client_by_client_id(db, client_id)
        if existing_client_by_id:
            logger.warning(f"Client with client_id '{client_id}' already exists. Skipping creation.")
            return

        # Check if client_name already exists (optional, depends on your unique constraints)
        existing_client_by_name = client_crud.get_api_client_by_name(db, client_name)
        if existing_client_by_name:
            logger.warning(f"Client with client_name '{client_name}' already exists. Skipping creation.")
            return

        created_client = client_crud.create_api_client(db, client_name, client_id, client_secret, is_active)
        if created_client:
            logger.info(f"Successfully created client: {created_client.client_name} (DB ID: {created_client.id})")
            logger.info(f"  Client ID: {created_client.client_id}")
            logger.info(f"  NOTE: The provided client_secret '{client_secret}' was hashed and stored. Keep the original secret safe if needed elsewhere.")
        else:
            logger.error(f"Failed to create client: {client_name}")
    finally:
        db.close()

def list_clients():
    """Lists all API clients in the database."""
    db = SessionLocal()
    try:
        logger.info("Listing all API clients:")
        clients = db.query(ApiClient).all()
        if not clients:
            logger.info("No API clients found in the database.")
            return
        for client in clients:
            print(f"  - Name: {client.client_name}, Client ID: {client.client_id}, Active: {client.is_active}, Created: {client.created_at}")
    finally:
        db.close()

if __name__ == "__main__":
    # Ensure DB and tables are ready before script operations
    try:
        initialize_db()
    except Exception:
        logger.error("Failed to initialize database. Exiting script.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Manage API Clients for the Product Categorizer API.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create parser for the "add" command
    parser_add = subparsers.add_parser("add", help="Add a new API client")
    parser_add.add_argument("--name", required=True, help="Descriptive name for the client application")
    parser_add.add_argument("--id", required=False, help="Public Client ID (will be generated if not provided)")
    parser_add.add_argument("--secret", required=False, help="Client Secret (will be generated if not provided)")
    parser_add.add_argument("--inactive", action="store_true", help="Create the client in an inactive state")

    # Create parser for the "list" command
    parser_list = subparsers.add_parser("list", help="List all API clients")

    args = parser.parse_args()

    if args.command == "add":
        client_id_to_add = args.id if args.id else f"client_{uuid4().hex[:12]}"
        client_secret_to_add = args.secret if args.secret else uuid4().hex # Generate a strong secret
        
        if not args.secret:
             logger.info(f"Generated Client Secret: {client_secret_to_add} (PLEASE STORE THIS SECURELY!)")

        add_client(
            client_name=args.name,
            client_id=client_id_to_add,
            client_secret=client_secret_to_add,
            is_active=not args.inactive
        )
    elif args.command == "list":
        list_clients()
    else:
        parser.print_help()