from pydantic import BaseModel, Field


class Clause(BaseModel):
    clause_id: str
    text: str
    section: str | None = None
    risk_label: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    clauses_extracted: int


class AskRequest(BaseModel):
    question: str = Field(min_length=5)
    document_id: str | None = None


class Evidence(BaseModel):
    clause_id: str
    score: float
    text: str
    section: str | None = None


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float
    corrective_steps: list[str]
    evidence: list[Evidence]
