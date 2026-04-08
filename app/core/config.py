from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'Legal Contract Intelligence System'
    openai_api_key: str | None = None

    vector_backend: str = 'pinecone'  # pinecone | weaviate | local
    pinecone_api_key: str | None = None
    pinecone_index: str = 'contracts-index'
    pinecone_env: str = 'us-east-1'
    weaviate_url: str | None = None

    database_url: str = 'sqlite:///./contracts.db'

    embedding_model: str = 'text-embedding-3-small'
    chat_model: str = 'gpt-4.1-mini'
    top_k: int = 5
    max_refinement_steps: int = 2
    upload_dir: str = 'uploads'


@lru_cache
def get_settings() -> Settings:
    return Settings()
