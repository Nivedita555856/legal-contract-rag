import re
import uuid
from pathlib import Path

from pypdf import PdfReader

from app.models.schemas import Clause


class ClauseExtractor:
    SECTION_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')

    def extract_text(self, path: Path) -> str:
        if path.suffix.lower() == '.pdf':
            reader = PdfReader(str(path))
            return '\n'.join((page.extract_text() or '') for page in reader.pages)
        return path.read_text(encoding='utf-8', errors='ignore')

    def split_into_clauses(self, text: str) -> list[Clause]:
        chunks = [c.strip() for c in re.split(r'\n{2,}', text) if len(c.strip()) > 30]
        clauses: list[Clause] = []
        for chunk in chunks:
            lines = chunk.splitlines()
            first_line = lines[0].strip() if lines else ''
            section = None
            m = self.SECTION_PATTERN.match(first_line)
            if m:
                section = f"{m.group(1)} {m.group(2)}"
            risk = self._risk_label(chunk)
            clauses.append(
                Clause(
                    clause_id=f'clause-{uuid.uuid4().hex[:10]}',
                    text=chunk,
                    section=section,
                    risk_label=risk,
                )
            )
        return clauses

    def _risk_label(self, clause_text: str) -> str:
        text = clause_text.lower()
        risk_terms = {
            'high': ['indemnify', 'liability cap', 'termination for convenience', 'penalty'],
            'medium': ['auto-renew', 'exclusive', 'arbitration', 'governing law'],
            'low': ['notice', 'confidentiality', 'definitions'],
        }
        for label, words in risk_terms.items():
            if any(word in text for word in words):
                return label
        return 'unknown'
