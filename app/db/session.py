from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False}  # requis pour SQLite: j’autorise plusieurs threads à utiliser la même connexion, par défaut "Une connexion = un seul thread”
)

SessionLocal = sessionmaker( #fabrique de sessions. Permet de créer des objets de session à la demande
    autocommit=False, #SQLAlchemy ne valide pas automatiquement les changements. 
    autoflush=False, #flush = envoyer les changements vers la DB sans commit
    bind=engine #Ça relie la session : à cette base via ce moteur :Sans ça → la session ne sait pas où envoyer les requêtes.
)
