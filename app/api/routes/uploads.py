from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session as DBSession
from pathlib import Path

from app.db.session import SessionLocal
from app.db.models.session import Session
from app.db.models.upload import Upload
from app.schemas.upload import UploadOut
from app.services.storage import make_temp_path
from app.services.audio import wav_metadata

router = APIRouter(prefix="/sessions", tags=["uploads"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/{session_id}/upload", response_model=UploadOut)
async def upload_recording(
    session_id: str,
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db)
):
    # 1) vérifier session existe
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2) vérifier extension supportée (MVP: wav)
    ext = Path(file.filename).suffix.lower()
    if ext != ".wav":
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Only .wav is supported for now."
        )

    # 3) sauvegarde temporaire
    temp_path = make_temp_path(file.filename)
    content = await file.read()
    temp_path.write_bytes(content)

    # 4) métadonnées wav
    meta = wav_metadata(temp_path)

    # 5) créer ligne upload en DB
    up = Upload(
        session_id=session_id,
        original_filename=file.filename,
        temp_path=str(temp_path),
        sample_rate=meta["sample_rate"],
        duration_sec=meta["duration_sec"],
    )
    db.add(up)
    db.commit()
    db.refresh(up)
    return up
