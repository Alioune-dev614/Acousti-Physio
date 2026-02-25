from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from datetime import datetime
from pathlib import Path

from app.db.session import SessionLocal
from app.db.models.session import Session
from app.db.models.upload import Upload
from app.db.models.measure import Measure
from app.schemas.measure import MeasureCreate, MeasureOut

router = APIRouter(prefix="/measures", tags=["measures"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=MeasureOut)
def create_measure(payload: MeasureCreate, db: DBSession = Depends(get_db)):
    # 1) vérifier session
    s = db.query(Session).filter(Session.id == payload.session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2) vérifier upload
    up = db.query(Upload).filter(Upload.id == payload.upload_id).first()
    if not up:
        raise HTTPException(status_code=404, detail="Upload not found")

    if up.deleted_at is not None:
        raise HTTPException(status_code=410, detail="Upload file already deleted")

    # 3) calcul de la mesure (MVP: durée)
    if payload.measure_type != "duration":
        raise HTTPException(status_code=400, detail="Only 'duration' is supported for now")

    if up.duration_sec is None:
        raise HTTPException(status_code=500, detail="Upload has no duration metadata")

    m = Measure(
        session_id=payload.session_id,
        upload_id=payload.upload_id,
        measure_type="duration",
        value=float(up.duration_sec),
        unit="s",
    )
    db.add(m)
    db.commit()
    db.refresh(m)

    # 4) supprimer le fichier temporaire après sauvegarde de la mesure
    try:
        p = Path(up.temp_path)
        if p.exists():
            p.unlink()
    finally:
        up.deleted_at = datetime.utcnow()
        db.add(up)
        db.commit()

    return m
