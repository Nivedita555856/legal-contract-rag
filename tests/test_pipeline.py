from pathlib import Path

from app.db.database import SessionLocal, init_db
from app.services.rag_pipeline import RAGPipeline


def test_ingest_and_ask(tmp_path: Path):
    init_db()
    pipeline = RAGPipeline()
    sample = tmp_path / 'sample.txt'
    sample.write_text(
        '1 Termination\nEither party may terminate for convenience with 30 days notice.\n\n'
        '2 Liability\nVendor indemnifies customer for third-party IP claims.',
        encoding='utf-8',
    )

    db = SessionLocal()
    try:
        ingested = pipeline.ingest_document(db=db, file_path=sample, filename='sample.txt')
    finally:
        db.close()

    assert ingested.clauses_extracted >= 2
    answer = pipeline.answer('What are termination risks?', document_id=ingested.document_id)
    assert answer.answer
    assert len(answer.corrective_steps) >= 1
