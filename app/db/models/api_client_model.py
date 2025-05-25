# app/db/models/api_client_model.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func # For default datetime
from app.db.database_setup import Base # Import Base from our setup

class ApiClient(Base):
    """
    SQLAlchemy model for representing an API client application that can authenticate
    with the service.
    """
    __tablename__ = "api_clients" # Name of the database table

    # Columns in the table
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    client_name = Column(String, nullable=False, unique=True, index=True, comment="Descriptive name for the client application")
    client_id = Column(String, nullable=False, unique=True, index=True, comment="Public unique identifier for the client")
    hashed_client_secret = Column(String, nullable=False, comment="Hashed version of the client's secret")
    is_active = Column(Boolean, default=True, nullable=False, comment="Whether the client is currently active and can authenticate")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<ApiClient(client_name='{self.client_name}', client_id='{self.client_id}', is_active={self.is_active})>"

# You would also need to create these tables in the database.
# This can be done via a startup event in main.py or a separate script.