"""API HTTP : soumettre un ticket (texte ou fichier PDF / image) et obtenir la réponse des agents."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .ingestion import IMAGE_EXT, PDF_EXT, TEXT_EXT

app = FastAPI(title="NovaShop support agents (SLM local)", version="0.1.0")
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from .graph import build_default_graph

        _graph = build_default_graph()
    return _graph


def set_graph(graph) -> None:
    """Permet d'injecter un graphe factice (tests)."""
    global _graph
    _graph = graph


def _to_response(state: dict, seconds: float) -> dict:
    return {
        "status": state.get("status"),
        "intent": state.get("intent"),
        "confidence": round(float(state.get("confidence", 0.0)), 3),
        "modality": state.get("modality"),
        "answer": state.get("answer"),
        "verdict": state.get("verdict"),
        "escalation_reason": state.get("escalation_reason"),
        "sources": sorted({d.metadata.get("source", "") for d in state.get("context_docs", [])}),
        "trace": state.get("trace", []),
        "total_seconds": round(seconds, 3),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tickets")
async def submit_ticket(text: str | None = Form(default=None), file: UploadFile | None = File(default=None)):
    if not text and file is None:
        raise HTTPException(status_code=422, detail="Provide 'text' or 'file'.")
    t0 = time.perf_counter()
    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in TEXT_EXT | PDF_EXT | IMAGE_EXT:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            state = get_graph().invoke({"source_file": tmp.name})
    else:
        state = get_graph().invoke({"ticket_text": text})
    return _to_response(state, time.perf_counter() - t0)
