import uuid
from sqlalchemy import Column, String, DateTime, Float, ForeignKey
from sqlalchemy.dialects.sqlite import CHAR
from datetime import datetime
from app.db.base import Base

class Measure(Base):
    __tablename__ = "measures"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(CHAR(36), ForeignKey("sessions.id"), nullable=False)
    upload_id = Column(CHAR(36), ForeignKey("uploads.id"), nullable=False)

    measure_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
