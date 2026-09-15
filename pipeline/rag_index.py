"""Pure-Python BM25 index over NaCCA indicator documents. No GPU required."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*", re.IGNORECASE)
CODE_RE = re.compile(r"\bB\d(?:\.\d+){2,4}\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d["text"]) for d in docs]
        self.doc_len = [len(toks) or 1 for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(docs), 1)
        self.df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.code_to_idx: dict[str, list[int]] = defaultdict(list)

        for idx, (doc, toks) in enumerate(zip(docs, self.doc_tokens)):
            counts = Counter(toks)
            for term, tf in counts.items():
                self.df[term] += 1
                self.postings[term].append((idx, tf))
            code = str(doc.get("indicator_code") or "").upper()
            if code:
                self.code_to_idx[code].append(idx)

        n = max(len(docs), 1)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in self.df.items()
        }

    def search(
        self,
        query: str,
        k: int = 5,
        subject_slug: str | None = None,
        grade: int | None = None,
    ) -> list[dict[str, Any]]:
        q = query.strip()
        if not q:
            return []

        codes = [m.group(0).upper() for m in CODE_RE.finditer(q)]
        hits: dict[int, float] = defaultdict(float)

        for code in codes:
            for idx in self.code_to_idx.get(code, []):
                hits[idx] += 50.0

        q_terms = tokenize(q)
        q_counts = Counter(q_terms)
        for term, qtf in q_counts.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in self.postings.get(term, []):
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                hits[idx] += idf * (tf * (self.k1 + 1) / denom) * qtf

        ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
        results = []
        for idx, score in ranked:
            doc = self.docs[idx]
            rec = doc.get("record") or doc
            if subject_slug and rec.get("subject_slug") != subject_slug:
                continue
            if grade is not None and rec.get("grade") != grade:
                continue
            results.append({"score": round(float(score), 4), "record": rec, "doc": doc})
            if len(results) >= k:
                break
        return results


def load_docs(path: Path) -> list[dict[str, Any]]:
    docs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs
