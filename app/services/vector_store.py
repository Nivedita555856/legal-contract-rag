from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from app.core.config import get_settings
from app.models.schemas import Clause

settings = get_settings()


@dataclass
class SearchHit:
    clause_id: str
    score: float
    text: str
    section: str | None


class BaseVectorStore:
    def upsert(self, document_id: str, clause: Clause, embedding: list[float]) -> None:
        raise NotImplementedError

    def query(self, embedding: list[float], top_k: int, document_id: str | None = None) -> list[SearchHit]:
        raise NotImplementedError


class LocalVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def upsert(self, document_id: str, clause: Clause, embedding: list[float]) -> None:
        self.rows.append(
            {
                'document_id': document_id,
                'clause_id': clause.clause_id,
                'section': clause.section,
                'text': clause.text,
                'embedding': embedding,
            }
        )

    def query(self, embedding: list[float], top_k: int, document_id: str | None = None) -> list[SearchHit]:
        candidates = [r for r in self.rows if not document_id or r['document_id'] == document_id]
        ranked = sorted(candidates, key=lambda r: self._cosine(embedding, r['embedding']), reverse=True)[:top_k]
        return [
            SearchHit(clause_id=r['clause_id'], score=self._cosine(embedding, r['embedding']), text=r['text'], section=r['section'])
            for r in ranked
        ]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sqrt(sum(x * x for x in a)) or 1.0
        nb = sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)


class PineconeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        from pinecone import Pinecone

        if not settings.pinecone_api_key:
            raise ValueError('PINECONE_API_KEY is required for Pinecone backend.')
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.client.Index(settings.pinecone_index)

    def upsert(self, document_id: str, clause: Clause, embedding: list[float]) -> None:
        self.index.upsert(
            vectors=[
                {
                    'id': clause.clause_id,
                    'values': embedding,
                    'metadata': {'document_id': document_id, 'text': clause.text, 'section': clause.section or ''},
                }
            ]
        )

    def query(self, embedding: list[float], top_k: int, document_id: str | None = None) -> list[SearchHit]:
        filt = {'document_id': {'$eq': document_id}} if document_id else None
        results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True, filter=filt)
        return [
            SearchHit(
                clause_id=m['id'],
                score=float(m['score']),
                text=(m.get('metadata') or {}).get('text', ''),
                section=(m.get('metadata') or {}).get('section') or None,
            )
            for m in results.get('matches', [])
        ]


class WeaviateVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        import weaviate

        if not settings.weaviate_url:
            raise ValueError('WEAVIATE_URL is required for Weaviate backend.')
        self.client = weaviate.connect_to_weaviate_cloud(cluster_url=settings.weaviate_url)
        self.collection_name = 'ContractClause'

    def upsert(self, document_id: str, clause: Clause, embedding: list[float]) -> None:
        collection = self.client.collections.get(self.collection_name)
        collection.data.insert(
            properties={
                'clause_id': clause.clause_id,
                'document_id': document_id,
                'text': clause.text,
                'section': clause.section or '',
            },
            vector=embedding,
        )

    def query(self, embedding: list[float], top_k: int, document_id: str | None = None) -> list[SearchHit]:
        collection = self.client.collections.get(self.collection_name)
        where = {'path': ['document_id'], 'operator': 'Equal', 'valueText': document_id} if document_id else None
        resp = collection.query.near_vector(near_vector=embedding, limit=top_k, filters=where, return_metadata=['distance'])
        hits = []
        for obj in resp.objects:
            props = obj.properties
            hits.append(
                SearchHit(
                    clause_id=props.get('clause_id', ''),
                    score=1 - float(obj.metadata.distance or 1),
                    text=props.get('text', ''),
                    section=props.get('section') or None,
                )
            )
        return hits


def build_vector_store() -> BaseVectorStore:
    backend = settings.vector_backend.lower()
    if backend == 'pinecone':
        return PineconeVectorStore()
    if backend == 'weaviate':
        return WeaviateVectorStore()
    return LocalVectorStore()
