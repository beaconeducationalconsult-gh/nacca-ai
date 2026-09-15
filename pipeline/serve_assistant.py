#!/usr/bin/env python3
"""Offline NaCCA curriculum assistant (BM25 RAG, no GPU).

Binds to 0.0.0.0 so the sandbox live preview can reach it.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_index import BM25Index, load_docs
from schema import SUBJECT_TITLES, subject_title

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = ROOT / "data" / "processed" / "rag_docs.jsonl"

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NaCCA Curriculum Assistant</title>
  <style>
    :root { --green:#0f6b3a; --gold:#c9a227; --ink:#122017; --muted:#4d5c53; --bg:#f4f7f4; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Georgia, "Palatino Linotype", serif; background:var(--bg); color:var(--ink); }
    header { background:linear-gradient(120deg, #0b4d2c, #147a44); color:#fff; padding:28px 20px 24px; }
    header h1 { margin:0 0 6px; font-size:1.6rem; font-weight:700; }
    header p { margin:0; opacity:.92; max-width:720px; }
    main { max-width:920px; margin:0 auto; padding:20px; }
    .card { background:#fff; border:1px solid #d7e2d9; border-radius:14px; padding:16px; margin-bottom:16px; }
    label { display:block; font-size:.85rem; color:var(--muted); margin-bottom:6px; }
    textarea, select, button { font: inherit; }
    textarea { width:100%; min-height:90px; padding:10px; border:1px solid #c5d4c9; border-radius:10px; }
    .row { display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; }
    .row > div { flex:1; min-width:160px; }
    select { width:100%; padding:8px; border:1px solid #c5d4c9; border-radius:8px; background:#fff; }
    button { background:var(--green); color:#fff; border:0; padding:10px 18px; border-radius:10px; cursor:pointer; }
    button:hover { filter:brightness(1.08); }
    .meta { font-size:.85rem; color:var(--muted); margin-bottom:8px; }
    .hit { border-left:4px solid var(--gold); padding:10px 12px; margin:10px 0; background:#fbfaf4; }
    .code { font-family: ui-monospace, Menlo, Consolas, monospace; color:var(--green); font-weight:700; }
    footer { text-align:center; color:var(--muted); font-size:.8rem; padding:24px; }
    .warn { background:#fff6e5; border:1px solid #ead7a0; padding:10px 12px; border-radius:10px; font-size:.9rem; }
  </style>
</head>
<body>
  <header>
    <h1>NaCCA Curriculum Assistant</h1>
    <p>Ask about Ghana’s basic education curriculum (B1–B9). Answers are retrieved from your indicator dump — no invented codes.</p>
  </header>
  <main>
    <div class="card">
      <label for="q">Question</label>
      <textarea id="q" placeholder="e.g. What is B4.1.1.1.1 in Mathematics?  or  B7 Science indicators on photosynthesis"></textarea>
      <div class="row">
        <div>
          <label for="subject">Subject (optional)</label>
          <select id="subject"><option value="">All subjects</option></select>
        </div>
        <div>
          <label for="grade">Grade (optional)</label>
          <select id="grade">
            <option value="">All grades</option>
            <option>1</option><option>2</option><option>3</option><option>4</option>
            <option>5</option><option>6</option><option>7</option><option>8</option><option>9</option>
          </select>
        </div>
      </div>
      <div class="row"><button id="go" type="button">Search curriculum</button></div>
    </div>
    <div id="out"></div>
    <p class="warn">Curriculum text is © NaCCA / Ministry of Education. This tool is a retrieval prototype, not an official NaCCA product. Seek written permission before commercial use.</p>
  </main>
  <footer>Beacon Educational Consult · hybrid RAG prototype · Hugging Face-ready dataset in data/processed</footer>
  <script>
    async function loadMeta() {
      const res = await fetch('/api/meta');
      const data = await res.json();
      const sel = document.getElementById('subject');
      (data.subjects || []).forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.slug; opt.textContent = s.title;
        sel.appendChild(opt);
      });
    }
    async function ask() {
      const q = document.getElementById('q').value.trim();
      const subject = document.getElementById('subject').value;
      const grade = document.getElementById('grade').value;
      const out = document.getElementById('out');
      if (!q) { out.innerHTML = '<div class="card">Enter a question or indicator code.</div>'; return; }
      out.innerHTML = '<div class="card">Searching NaCCA indicators…</div>';
      const url = '/api/ask?q=' + encodeURIComponent(q)
        + (subject ? '&subject=' + encodeURIComponent(subject) : '')
        + (grade ? '&grade=' + encodeURIComponent(grade) : '');
      const res = await fetch(url);
      const data = await res.json();
      if (!data.hits || !data.hits.length) {
        out.innerHTML = '<div class="card"><strong>No matching indicator in this dump.</strong><p>'
          + (data.answer || '') + '</p></div>';
        return;
      }
      let html = '<div class="card"><div class="meta">' + data.hits.length
        + ' retrieved indicator(s). Extractive answer — not a fine-tuned LLM.</div><p>'
        + data.answer.replaceAll('\\n', '<br>') + '</p>';
      data.hits.forEach(h => {
        const r = h.record;
        html += '<div class="hit"><span class="code">' + r.indicator_code + '</span> · '
          + r.subject + ' ' + r.grade_label
          + '<div>' + (r.indicator || '') + '</div>'
          + '<div class="meta">' + (r.strand || '') + ' · ' + (r.content_standard_code || '') + '</div></div>';
      });
      html += '</div>';
      out.innerHTML = html;
    }
    document.getElementById('go').addEventListener('click', ask);
    document.getElementById('q').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
    });
    loadMeta();
  </script>
</body>
</html>
"""


def format_answer(query: str, hits: list[dict]) -> str:
    if not hits:
        return (
            "No matching NaCCA indicator was found in this dataset. "
            "Try an official code (e.g. B4.1.1.1.1) or add a subject and grade filter. "
            "I will not invent an indicator."
        )
    lines = [
        f"Retrieved {len(hits)} NaCCA indicator(s) for: {query.strip()}",
        "",
    ]
    for item in hits:
        rec = item["record"]
        lines.extend(
            [
                f"{rec['indicator_code']} — {rec['subject']} {rec['grade_label']}",
                f"  Strand: {rec.get('strand') or '—'}",
                f"  Content standard {rec.get('content_standard_code')}: {rec.get('content_standard') or '—'}",
                f"  Indicator: {rec.get('indicator') or '—'}",
                "",
            ]
        )
    lines.append("Cite these codes in lesson notes. Source: local NaCCA JSON dump, not a generated model.")
    return "\n".join(lines)


def make_handler(index: BM25Index, stats: dict):
    subjects = [{"slug": slug, "title": subject_title(slug)} for slug in sorted(SUBJECT_TITLES)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"[{self.address_string()}] {fmt % args}")

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/meta":
                payload = {"subjects": subjects, "stats": stats, "n_docs": len(index.docs)}
                self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
                return
            if parsed.path == "/api/ask":
                qs = parse_qs(parsed.query)
                query = (qs.get("q") or [""])[0]
                subject = (qs.get("subject") or [""])[0] or None
                grade_raw = (qs.get("grade") or [""])[0]
                grade = int(grade_raw) if grade_raw.isdigit() else None
                hits = index.search(query, k=5, subject_slug=subject, grade=grade)
                payload = {
                    "query": query,
                    "answer": format_answer(query, hits),
                    "hits": hits,
                }
                self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
                return
            self._send(404, b"Not found", "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not args.docs.exists():
        raise SystemExit(f"Missing {args.docs}. Run: python3 pipeline/build_hf_dataset.py")
    docs = load_docs(args.docs)
    index = BM25Index(docs)
    stats_path = args.docs.parent / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(index, stats))
    print(f"NaCCA assistant at http://{args.host}:{args.port}  ({len(docs)} indicators)")
    server.serve_forever()


if __name__ == "__main__":
    main()
