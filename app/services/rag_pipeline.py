from __future__ import annotations

import json
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import ClauseRecord, Document
from app.models.schemas import AskResponse, Clause, Evidence, IngestResponse
from app.prompts.templates import ANSWER_PROMPT, EVALUATE_PROMPT
from app.services.extractor import ClauseExtractor
from app.services.llm import ChatService, EmbeddingService
from app.services.vector_store import BaseVectorStore, build_vector_store

settings = get_settings()


class FallbackEmbeddingService:
    def embed(self, text: str) -> list[float]:
        vals = [float((ord(c) % 47) / 47) for c in text[:256]]
        if len(vals) < 256:
            vals.extend([0.0] * (256 - len(vals)))
        return vals


class RAGPipeline:
    def __init__(self) -> None:
        self.extractor = ClauseExtractor()
        self.vector_store: BaseVectorStore = build_vector_store()
        self.embedder = EmbeddingService() if settings.openai_api_key else FallbackEmbeddingService()
        self.chat = ChatService() if settings.openai_api_key else None

    def ingest_document(self, db: Session, file_path: Path, filename: str) -> IngestResponse:
        doc_id = f'doc-{uuid.uuid4().hex[:12]}'
        text = self.extractor.extract_text(file_path)
        clauses = self.extractor.split_into_clauses(text)

        db.add(Document(id=doc_id, filename=filename))
        self._store_clauses(db, doc_id, clauses)
        db.commit()

        return IngestResponse(document_id=doc_id, filename=filename, clauses_extracted=len(clauses))

    def _store_clauses(self, db: Session, document_id: str, clauses: list[Clause]) -> None:
        for clause in clauses:
            embedding = self.embedder.embed(clause.text)
            self.vector_store.upsert(document_id=document_id, clause=clause, embedding=embedding)
            db.add(
                ClauseRecord(
                    clause_id=clause.clause_id,
                    document_id=document_id,
                    section=clause.section,
                    risk_label=clause.risk_label,
                    text=clause.text,
                    metadata_json={'risk_label': clause.risk_label, 'section': clause.section},
                )
            )

    def answer(self, question: str, document_id: str | None = None) -> AskResponse:
        steps: list[str] = []
        query = question

        for i in range(settings.max_refinement_steps + 1):
            steps.append(f'retrieve_pass_{i+1}')
            qvec = self.embedder.embed(query)
            hits = self.vector_store.query(embedding=qvec, top_k=settings.top_k, document_id=document_id)
            context = '\n\n'.join([f"[{h.clause_id}] {h.text}" for h in hits])

            if self.chat is None:
                answer = self._fallback_answer(question, hits)
                return AskResponse(
                    answer=answer,
                    grounded=bool(hits),
                    confidence=0.55 if hits else 0.1,
                    corrective_steps=steps,
                    evidence=[Evidence(clause_id=h.clause_id, score=h.score, text=h.text, section=h.section) for h in hits],
                )

            draft = self.chat.invoke(ANSWER_PROMPT.format(question=question, context=context))
            steps.append(f'evaluate_pass_{i+1}')
            eval_raw = self.chat.invoke(EVALUATE_PROMPT.format(question=question, answer=draft, context=context))
            parsed = self._parse_eval(eval_raw)

            if parsed.get('grounded', False):
                return AskResponse(
                    answer=draft,
                    grounded=True,
                    confidence=float(parsed.get('confidence', 0.8)),
                    corrective_steps=steps,
                    evidence=[Evidence(clause_id=h.clause_id, score=h.score, text=h.text, section=h.section) for h in hits],
                )

            hint = parsed.get('refinement_hint') or f'{question} contract clause obligations exceptions'
            query = hint
            steps.append(f'refine_query_pass_{i+1}')

        return AskResponse(
            answer='I could not produce a fully grounded answer with the current evidence. Please upload more relevant contract text.',
            grounded=False,
            confidence=0.2,
            corrective_steps=steps,
            evidence=[],
        )

    @staticmethod
    def _parse_eval(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {'grounded': False, 'confidence': 0.3, 'refinement_hint': '', 'reason': raw[:180]}

    @staticmethod
    def _fallback_answer(question: str, hits: list) -> str:
        if not hits:
            return 'No relevant clauses were found. Please ingest a contract first.'
        bullet = '\n'.join([f'- {h.clause_id}: {h.text[:200]}...' for h in hits[:3]])
        return f"Question: {question}\n\nMost relevant clauses:\n{bullet}\n\n(Using local fallback mode without external LLM.)"
