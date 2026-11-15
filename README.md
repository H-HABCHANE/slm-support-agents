# Agents IA de support client sur petits modèles de langage (SLM) locaux

Système multi-agents qui traite des tickets de support client (texte, PDF, capture d'écran) **entièrement en
local, sans GPU ni API externe** : classification d'intention par un SLM fine-tuné en LoRA, réponse rédigée
par un SLM à partir d'une base de connaissances (RAG), vérification de la fidélité de la réponse par un modèle
NLI, et escalade vers un humain quand l'agent n'est pas sûr de lui.

L'objectif n'est pas seulement de faire fonctionner la chaîne, mais de **mesurer** ce qu'elle vaut :
qualité, coût (latence, mémoire) et souveraineté (aucune donnée ne quitte la machine), et de documenter
honnêtement ses limites.

**Stack** : Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · LangChain · LangGraph ·
sentence-transformers · DeBERTa-v3 (NLI) · RapidOCR · FastAPI · Docker · GitHub Actions · pytest

## Architecture

```
 ticket (texte | PDF | image)
        │
   [ingestion]  pypdf, OCR RapidOCR (ONNX, local)
        │
   [classification]  SLM Qwen2.5-0.5B + LoRA  ou  TF-IDF  →  intention + confiance
        │
   (routage) ── intention réservée aux humains, ou confiance < 0,5 ──────────┐
        │                                                                    │
   [recherche RAG]  LangChain, embeddings all-MiniLM-L6-v2, 11 fiches + 1 PDF │
        │                                                                    │
   [rédaction]  SLM génératif (SmolLM2-360M / Qwen2.5-0.5B / Qwen2.5-1.5B)   │
        │                                                                    │
   [vérification]  chiffres présents dans la doc + NLI DeBERTa-v3            │
        │      └─ rejet : nouvel essai avec consigne (1 max), puis ──────────┤
        │                                                                    ▼
   réponse envoyée                                            [escalade vers un humain]
```

Le graphe est construit avec **LangGraph** (`src/support_agent/graph.py`) : 7 nœuds, 2 branchements
conditionnels et une boucle de correction. Chaque nœud enregistre sa durée, ce qui permet de mesurer la
latence agent par agent. Toutes les dépendances (classifieur, recherche, SLM, vérificateur) sont injectées :
les tests remplacent les modèles par des doublures et tournent en quelques secondes.

## Données

- **Entraînement et test « en distribution »** : [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
  (26 872 requêtes, 27 intentions, licence CDLA-Sharing-1.0), découpé en train / validation / test de façon
  stratifiée et reproductible (graine fixe).
- **Test « hors distribution »** : `data/eval/realistic_tickets.jsonl`, 54 tickets écrits à la main (2 par
  intention), plus longs et plus naturels (contexte inutile, fautes, demandes implicites). Les phrases Bitext
  sont générées à partir de modèles et très régulières : un score parfait sur Bitext ne dit presque rien de
  la performance sur de vrais tickets.
- **Base de connaissances** : documentation d'une boutique fictive, NovaShop (`data/kb/`) : 11 fiches
  Markdown (une section par intention) et des conditions générales en PDF.
- **Tickets multimodaux d'exemple** : `data/samples/` (e-mail texte, courrier PDF, capture d'écran d'une
  erreur de paiement), générés par `scripts/make_samples.py`.

<!-- RESULTS -->

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
```

## Utilisation

```bash
# Traiter un ticket (texte, PDF ou image)
python scripts/run_ticket.py data/samples/ticket_screenshot.png
python scripts/run_ticket.py --text "Where is my order NS-128840? It was supposed to arrive yesterday."

# API
uvicorn support_agent.api:app --port 8000
curl -F "file=@data/samples/ticket_letter.pdf" http://localhost:8000/tickets
curl -F "text=Do you deliver to pickup points?" http://localhost:8000/tickets

# Docker
docker build -t support-agent . && docker run -p 8000:8000 -v hf-cache:/models-cache support-agent
```

## Reproduire les résultats

```bash
python scripts/make_samples.py                     # PDF de la base + tickets d'exemple
python scripts/train_baseline.py                   # TF-IDF (20 s)
python scripts/train_lora.py --max-train 4000      # LoRA sur CPU (~15 min), ou notebooks/finetune_lora_colab.ipynb sur GPU
python scripts/evaluate_classifiers.py             # reports/classification.json
python scripts/evaluate_retrieval.py               # reports/retrieval.json
python scripts/evaluate_judge.py                   # reports/judge_calibration.json
python scripts/evaluate_agent.py --all             # reports/agent_*.json (≈ 1 h sur CPU)
python scripts/make_report.py                      # reports/RESULTS.md
pytest -q && ruff check .
```

## Structure

```
src/support_agent/
  ingestion.py    texte, PDF (pypdf), image (OCR RapidOCR)
  classifier.py   TF-IDF, SLM zero-shot, SLM + LoRA (interface commune)
  kb.py           base de connaissances, découpage, embeddings, recherche (LangChain)
  llm.py          SLM local (Transformers) exposé comme modèle de chat LangChain
  grounding.py    vérificateurs : NLI + contrôle des chiffres, ou juge SLM
  graph.py        orchestration multi-agents (LangGraph)
  api.py          API FastAPI
scripts/          entraînement, évaluations, rapport
notebooks/        fine-tuning LoRA sur GPU (Colab)
tests/            pytest (doublures de modèles, pas de téléchargement de SLM sauf le test NLI marqué « slow »)
```
