"""Base de connaissances et recherche (RAG) avec LangChain.

Les fichiers Markdown sont découpés par section (une section = une intention) ; les PDF sont extraits
puis découpés en morceaux. Les embeddings sont calculés en local (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import load_config, resolve
from .ingestion import clean_text, read_pdf

SECTION_RE = re.compile(r"^## (?P<title>.+?)(?: \((?P<intent>[a-z_]+)\))?\s*$", re.MULTILINE)


def split_markdown(text: str, source: str) -> list[Document]:
    """Une section « ## Titre (intention) » devient un document, avec l'intention en métadonnée."""
    docs = []
    matches = list(SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        docs.append(Document(
            page_content=f"{m['title']}\n{body}",
            metadata={"source": source, "section": m["title"], "intent": m["intent"] or ""},
        ))
    return docs


def load_kb_documents(kb_dir: str | Path | None = None) -> list[Document]:
    cfg = load_config()["kb"]
    kb_dir = Path(kb_dir or resolve(cfg["dir"]))
    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    docs: list[Document] = []
    for path in sorted(kb_dir.iterdir()):
        if path.suffix == ".md":
            docs += split_markdown(path.read_text(encoding="utf-8"), path.name)
        elif path.suffix == ".pdf":
            text = clean_text(read_pdf(path))
            for chunk in splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata={"source": path.name, "section": "", "intent": ""}))
    return docs


def get_embeddings(model_name: str | None = None):
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = model_name or load_config()["kb"]["embedding_model"]
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


class RerankRetriever(BaseRetriever):
    """Recherche en deux étapes : recherche vectorielle large (fetch_k candidats), puis reranking par un
    cross-encoder qui lit la requête et chaque passage ensemble (plus précis, plus coûteux)."""

    store: InMemoryVectorStore
    reranker: Any
    fetch_k: int = 10
    k: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        candidates = self.store.similarity_search(query, k=self.fetch_k)
        if not candidates:
            return []
        scores = self.reranker.predict([(query, d.page_content) for d in candidates])
        ranked = sorted(zip(scores, range(len(candidates)), candidates, strict=True), key=lambda t: (-t[0], t[1]))
        return [d for _, _, d in ranked[: self.k]]


@lru_cache(maxsize=1)
def get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def build_retriever(docs: list[Document] | None = None, embeddings=None, k: int | None = None,
                    rerank: bool | None = None) -> BaseRetriever:
    cfg = load_config()["kb"]
    docs = docs if docs is not None else load_kb_documents()
    embeddings = embeddings or get_embeddings()
    k = k or cfg["top_k"]
    rerank = cfg.get("rerank", False) if rerank is None else rerank
    store = InMemoryVectorStore.from_documents(docs, embeddings)
    if rerank:
        return RerankRetriever(store=store, reranker=get_reranker(cfg["rerank_model"]),
                               fetch_k=cfg.get("fetch_k", 10), k=k)
    return store.as_retriever(search_kwargs={"k": k})


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i + 1}] ({d.metadata.get('source', '?')}) {d.page_content}" for i, d in enumerate(docs))
