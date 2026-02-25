import uuid
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.sqlite import CHAR
from datetime import datetime
from app.db.base import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    label = Column(String, nullable=False)
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.utcnow)
