from pydantic import BaseModel
from datetime import datetime

class SessionCreate(BaseModel):
    label: str

class SessionOut(BaseModel):
    id: str
    label: str
    status: str
    created_at: datetime

    class Config:
        orm_mode = True
