import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SessionLocal
from app.models.schemas import AskRequest, AskResponse, IngestResponse
from app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix='/api', tags=['contracts'])
settings = get_settings()
pipeline = RAGPipeline()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@router.post('/ingest', response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), db: Session = Depends(get_db)) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.pdf', '.txt'}:
        raise HTTPException(status_code=400, detail='Only PDF and TXT files are supported.')

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / file.filename

    with target.open('wb') as out:
        shutil.copyfileobj(file.file, out)

    return pipeline.ingest_document(db=db, file_path=target, filename=file.filename)


@router.post('/ask', response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return pipeline.answer(question=req.question, document_id=req.document_id)
