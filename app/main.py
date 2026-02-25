from fastapi import FastAPI
from app.db.session import engine
from app.db.base import Base

# Import des modèles pour que SQLAlchemy les connaisse
from app.db.models.session import Session
from app.db.models.upload import Upload
from app.db.models.measure import Measure

from app.api.routes.sessions import router as session_router
from app.api.routes.uploads import router as uploads_router
from app.api.routes.measures import router as measures_router

app = FastAPI(
    title="PFE Acoustique & Physiologie",
    description="API pour sessions et mesures phonétiques",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"status": "API running"}




app.include_router(session_router)
app.include_router(uploads_router)
app.include_router(measures_router)
