"""Ingestion multimodale : texte, PDF et image (OCR) vers du texte exploitable par les agents."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".eml"}
PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def clean_text(text: str) -> str:
    """Normalise les espaces et supprime les lignes vides en trop."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def read_image(path: str | Path) -> str:
    """OCR local (RapidOCR, modèles ONNX) : renvoie les lignes détectées de haut en bas."""
    result, _ = _ocr_engine()(str(path))
    if not result:
        return ""
    # result : liste de [boîte, texte, score] ; tri par position verticale puis horizontale
    lines = sorted(result, key=lambda r: (round(r[0][0][1] / 15), r[0][0][0]))
    return "\n".join(r[1] for r in lines)


def detect_modality(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in TEXT_EXT:
        return "text"
    if ext in PDF_EXT:
        return "pdf"
    if ext in IMAGE_EXT:
        return "image"
    raise ValueError(f"Format non pris en charge : {ext}")


def load_file(path: str | Path) -> tuple[str, str]:
    """Renvoie (texte extrait, modalité)."""
    modality = detect_modality(path)
    if modality == "text":
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    elif modality == "pdf":
        raw = read_pdf(path)
    else:
        raw = read_image(path)
    return clean_text(raw), modality
