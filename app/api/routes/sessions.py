from fastapi import APIRouter, Depends
from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session as DBSession
from app.db.session import SessionLocal
from app.db.models.session import Session
from app.schemas.session import SessionCreate, SessionOut

router = APIRouter(prefix="/sessions", tags=["sessions"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=SessionOut)
def create_session(payload: SessionCreate, db: DBSession = Depends(get_db)):
    s = Session(label=payload.label)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s

@router.get("", response_model=list[SessionOut])
def list_sessions(db: DBSession = Depends(get_db)):
    return db.query(Session).order_by(Session.created_at.desc()).all()




from app.db.models.measure import Measure
from app.schemas.measure import MeasureOut

@router.get("/{session_id}/measures", response_model=list[MeasureOut])
def list_session_measures(session_id: str, db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    return (
        db.query(Measure)
        .filter(Measure.session_id == session_id)
        .order_by(Measure.created_at.desc())
        .all()
    )
