"""Chargement d'un SLM local (Hugging Face Transformers) exposé comme modèle de chat LangChain."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from .config import load_config


@lru_cache(maxsize=2)
def get_chat_model(model_id: str | None = None, max_new_tokens: int | None = None) -> BaseChatModel:
    """SLM exécuté en local sur CPU (ou GPU s'il existe), décodage glouton pour des résultats reproductibles."""
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    cfg = load_config()["llm"]
    model_id = model_id or cfg["model"]
    max_new_tokens = max_new_tokens or cfg["max_new_tokens"]

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=llm, tokenizer=tokenizer, model_id=model_id)


def count_tokens(chat_model: BaseChatModel, text: str) -> int:
    tok = getattr(chat_model, "tokenizer", None)
    if tok is None:
        return len(text.split())
    return len(tok.encode(text, add_special_tokens=False))
