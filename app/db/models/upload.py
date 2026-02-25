import uuid
from sqlalchemy import Column, String, DateTime, Float, ForeignKey
from sqlalchemy.dialects.sqlite import CHAR
from datetime import datetime
from app.db.base import Base

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(CHAR(36), ForeignKey("sessions.id"), nullable=False)

    original_filename = Column(String, nullable=False)
    duration_sec = Column(Float)
    sample_rate = Column(Float)
    temp_path = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)
