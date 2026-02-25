from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UploadOut(BaseModel):
    id: str
    session_id: str
    original_filename: str
    sample_rate: float | None = None
    duration_sec: float | None = None
    temp_path: str | None = None
    created_at: datetime
    deleted_at: Optional[datetime] = None

    class Config:
        orm_mode = True
