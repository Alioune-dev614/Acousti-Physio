from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MeasureCreate(BaseModel):
    session_id: str
    upload_id: str
    measure_type: str  # ex: "duration"
    t0: Optional[float] = None
    t1: Optional[float] = None

class MeasureOut(BaseModel):
    id: str
    session_id: str
    upload_id: str
    measure_type: str
    value: float
    unit: str | None = None
    created_at: datetime

    class Config:
        orm_mode = True
