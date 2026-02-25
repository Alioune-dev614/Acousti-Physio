from pydantic_settings  import BaseSettings

class Settings(BaseSettings):
    DB_URL: str = "sqlite:///./app.db"

    class Config:
        env_file = ".env"  #Autorise Pydantic à lire un fichier .env

settings = Settings() #Crée une instance utilisable partout dans le projet import app.core.config.settings
