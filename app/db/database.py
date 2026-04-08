from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

from app.core.config import get_settings

settings = get_settings()

connect_args = {'check_same_thread': False} if settings.database_url.startswith('sqlite') else {}
engine = create_engine(settings.database_url, future=True, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class Document(Base):
    __tablename__ = 'documents'

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ClauseRecord(Base):
    __tablename__ = 'clauses'

    id = Column(Integer, primary_key=True, index=True)
    clause_id = Column(String, index=True, nullable=False)
    document_id = Column(String, index=True, nullable=False)
    section = Column(String, nullable=True)
    risk_label = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
