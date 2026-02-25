from app.db.session import engine
from app.db.base import Base
from app.db.models.session import Session
from app.db.models.upload import Upload
from app.db.models.measure import Measure

Base.metadata.create_all(bind=engine)
print("✅ Base de données créée")
