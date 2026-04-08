from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.db.database import init_db

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(router)


@app.on_event('startup')
def on_startup() -> None:
    init_db()


@app.get('/')
def root() -> dict:
    return {'message': settings.app_name, 'docs': '/docs'}
