from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Busca .env en backend/, en la raiz del proyecto y en el directorio actual
_BACKEND = Path(__file__).resolve().parent.parent
_ENV_CANDIDATOS = tuple(
    p for p in (_BACKEND / ".env", _BACKEND.parent / ".env", Path.cwd() / ".env") if p.is_file()
)


class Settings(BaseSettings):
    """Configuracion centralizada desde variables de entorno.

    Evita hardcodear credenciales (regla de codificacion).
    """

    model_config = SettingsConfigDict(env_file=_ENV_CANDIDATOS or None, extra="ignore")

    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    REDIS_URL: str = "redis://localhost:6379/0"
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    LDAP_URL: str = ""
    LDAP_BASE_DN: str = ""
    ENV: str = "dev"

    APP_VERSION: str = "0.1.0" 
    NBM_ANONYMIZATION_PEPPER: str 


settings = Settings()  # type: ignore[call-arg]

if __name__ == "__main__":
    print("Candidatos .env encontrados:")
    for p in _ENV_CANDIDATOS:
        print(f"  - {p}")
    print("\nURL SYNC que usa Pydantic:")
    print(f"  {settings.DATABASE_URL_SYNC}")