"""Base de connaissances et recherche (RAG) avec LangChain.

Les fichiers Markdown sont découpés par section (une section = une intention) ; les PDF sont extraits
puis découpés en morceaux. Les embeddings sont calculés en local (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import re
from pathlib import Path

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


def build_retriever(docs: list[Document] | None = None, embeddings=None, k: int | None = None) -> BaseRetriever:
    docs = docs if docs is not None else load_kb_documents()
    embeddings = embeddings or get_embeddings()
    k = k or load_config()["kb"]["top_k"]
    store = InMemoryVectorStore.from_documents(docs, embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i + 1}] ({d.metadata.get('source', '?')}) {d.page_content}" for i, d in enumerate(docs))
