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

## Résultats

### Classification d'intention (27 intentions)

| Classifieur | Entraînement | Bitext test : acc. / F1 macro | Tickets réalistes : acc. / F1 macro | Latence médiane |
|---|---|---|---|---|
| tfidf_logreg | train Bitext complet | 99.7 % / 99.7 % (n=1000) | **61.1 %** / 59.2 % (n=54) | 2 ms |
| tfidf_logreg+augmentation | train Bitext complet + augmentation | 99.7 % / 99.7 % (n=1000) | **59.3 %** / 55.3 % (n=54) | 2 ms |
| slm_lora | 4000 exemples, 1 époque, cpu | 98.8 % / 98.8 % (n=1000) | **44.4 %** / 40.3 % (n=54) | 155 ms |
| slm_lora+augmentation | 4000 exemples, 1 époque, cpu | 98.2 % / 98.2 % (n=1000) | **68.5 %** / 64.1 % (n=54) | 215 ms |
| zero_shot:Qwen2.5-0.5B-Instruct | aucun (prompt) | 27.0 % / 23.2 % (n=100) | **20.4 %** / 20.5 % (n=54) | 1513 ms |
| zero_shot:Qwen2.5-1.5B-Instruct | aucun (prompt) | 59.0 % / 47.9 % (n=100) | **53.7 %** / 49.5 % (n=54) | 6602 ms |

### Recherche RAG (54 tickets réalistes, all-MiniLM-L6-v2)

| Requête | Recherche | hit@1 | hit@3 | MRR@3 | Latence médiane |
|---|---|---|---|---|---|
| texte seul | vectoriel | 66.7 % | 87.0 % | 0.756 | 12.7 ms |
| texte seul | vectoriel + reranking | 59.3 % | 88.9 % | 0.725 | 399.0 ms |
| texte + intention prédite | vectoriel | 66.7 % | 88.9 % | 0.772 | 19.3 ms |
| texte + intention prédite | vectoriel + reranking | 74.1 % | 88.9 % | 0.812 | 464.4 ms |
| texte + intention réelle (plafond) | vectoriel | 81.5 % | 100.0 % | 0.898 | 18.8 ms |
| texte + intention réelle (plafond) | vectoriel + reranking | 90.7 % | 100.0 % | 0.954 | 411.9 ms |

### Calibration des vérificateurs (38 cas : moitié fidèles, moitié aux chiffres modifiés)

| Vérificateur | Méthode | Accepte les réponses fidèles | Rejette les réponses fausses | Exactitude équilibrée |
|---|---|---|---|---|
| nli-deberta-v3-xsmall | NLI DeBERTa-v3-xsmall seul | 100.0 % | 100.0 % | **100.0 %** |
| nli-deberta-v3-xsmall | NLI + contrôle des chiffres | 100.0 % | 100.0 % | **100.0 %** |
| Qwen2.5-1.5B-Instruct | verify (SUPPORTED / NOT_SUPPORTED) | 89.5 % | 100.0 % | **94.7 %** |
| Qwen2.5-1.5B-Instruct | error? (YES / NO) | 100.0 % | 63.2 % | **81.6 %** |
| SmolLM2-360M-Instruct | verify (SUPPORTED / NOT_SUPPORTED) | 21.1 % | 89.5 % | **55.3 %** |
| Qwen2.5-0.5B-Instruct | verify (SUPPORTED / NOT_SUPPORTED) | 0.0 % | 100.0 % | **50.0 %** |
| SmolLM2-360M-Instruct | error? (YES / NO) | 63.2 % | 31.6 % | **47.4 %** |
| Qwen2.5-0.5B-Instruct | error? (YES / NO) | 0.0 % | 73.7 % | **36.8 %** |

### Vérificateur sur de vraies réponses de l'agent (15 brouillons de Qwen2.5-1.5B annotés à la main : 11 fidèles, 4 faux)

| Vérificateur | Accepte les réponses fidèles | Rejette les réponses fausses | Exactitude équilibrée |
|---|---|---|---|
| xsmall : NLI + chiffres (toutes prémisses) | 27.3 % | 100.0 % | **63.6 %** |
| xsmall : NLI + chiffres + prémisses proches | 72.7 % | 50.0 % | **61.4 %** |
| base : NLI + chiffres (toutes prémisses) | 45.5 % | 75.0 % | **60.2 %** |
| base : NLI + chiffres + prémisses proches | 81.8 % | 50.0 % | **65.9 %** |

### Agent complet : comparaison des SLM générateurs (CPU 8 cœurs, sans GPU, classifieur lora_aug, vérificateur nli)

| SLM | Paramètres | RAM | Réponses acceptées | Dont au 1er essai | Confirmées par le juge | Similarité / référence | Latence médiane (p90) | Tokens/s |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-360M-Instruct | 362 M | 1406 Mo | 8.7 % | 0.0 % | 0.0 % | 0.524 | 46.4 s (62.0 s) | 7.2 |
| Qwen2.5-0.5B-Instruct | 494 M | 1935 Mo | 43.5 % | 13.0 % | 10.0 % | 0.559 | 31.2 s (52.5 s) | 3.5 |
| Qwen2.5-1.5B-Instruct | 1544 M | 4218 Mo | 26.1 % | 17.4 % | 16.7 % | 0.526 | 73.6 s (115.7 s) | 1.7 |

23 tickets sur 27 passent par le SLM, les autres sont escaladés directement (intention réservée aux humains ou confiance trop faible). Exactitude du classifieur sur ces tickets : 70.4 %. Appels à une API externe : 0.

## Ce que montrent les mesures

- **Hors distribution, tout s'effondre.** Tous les classifieurs dépassent 98 % sur le test Bitext, mais
  tombent entre 44 % et 69 % sur les tickets écrits à la main. Le LoRA entraîné sur Bitext brut fait même
  moins bien que TF-IDF (44 % contre 61 %) : il a appris le style très régulier de Bitext.
- **L'augmentation de données change le classement.** Entraîné sur des phrases Bitext insérées dans des
  e-mails réalistes (formule d'appel, contexte, numéro de commande, signature), le même LoRA passe de 44 % à
  **68,5 %** et devient le meilleur classifieur ; TF-IDF, lui, n'en profite pas. C'est ce classifieur que
  l'agent utilise (choix automatique sur le jeu réaliste, ce qui rend son score légèrement optimiste).
- **Le reranking n'a pas été retenu.** Le cross-encoder améliore le premier résultat quand l'intention est
  ajoutée à la requête, mais pas le hit@3 (les 3 passages donnés au SLM) et il multiplie la latence de la
  recherche par 20 à 30. Il reste disponible (`kb.rerank: true`).
- **Un juge SLM n'est fiable qu'à partir de 1,5 B paramètres.** SmolLM2-360M et Qwen2.5-0.5B répondent
  presque toujours la même chose (exactitude équilibrée ≈ 50 %). La formulation du prompt compte aussi
  (94,7 % contre 81,6 % pour le même modèle).
- **Le vérificateur est le goulot d'étranglement.** Sur le jeu de calibration (sections recopiées contre
  chiffres modifiés), le vérificateur NLI est parfait ; sur de vraies réponses, il rejette encore 8 réponses
  fidèles sur 11, car un petit modèle NLI juge « contredites » des paraphrases ou des formules de politesse.
  Les variantes testées (prémisses filtrées par similarité, modèle NLI 8 fois plus gros) échangent des
  faux rejets contre des réponses fausses acceptées, sans gain net sur 15 cas. J'ai gardé la version
  stricte : en support client, une escalade vers un humain coûte moins cher qu'une réponse fausse. Le
  tableau de l'agent a été mesuré avant la correction des numéros de liste (« 1. 2. 3. » comptés comme
  chiffres inventés), qui concernait 3 des 23 tickets avec Qwen2.5-1.5B.
- **Le modèle le plus gros n'est pas celui qui répond le plus.** Qwen2.5-0.5B voit plus de réponses
  acceptées que Qwen2.5-1.5B, mais le juge (Qwen2.5-1.5B) en confirme une plus faible part. Sur CPU, le
  1,5 B produit 1,7 token/s, soit 74 s par ticket : il faudrait un GPU ou un modèle quantifié pour un usage
  réel.

## Limites et suites

- Jeux d'évaluation petits (54 tickets, 15 réponses annotées) : les écarts de quelques points ne sont pas
  significatifs.
- Pistes : vérificateur NLI fine-tuné sur des paires réponse / documentation du domaine, quantification
  (GGUF, int8) pour la latence, fine-tuning du générateur sur des réponses de référence.


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
python scripts/evaluate_verifier_real.py           # reports/verifier_real.json
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
