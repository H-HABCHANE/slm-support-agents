# Résultats

## Classification d'intention (27 intentions)

| Classifieur | Entraînement | Bitext test : acc. / F1 macro | Tickets réalistes : acc. / F1 macro | Latence médiane |
|---|---|---|---|---|
| tfidf_logreg | train Bitext complet | 99.7 % / 99.7 % (n=1000) | **61.1 %** / 59.2 % (n=54) | 2 ms |
| tfidf_logreg+augmentation | train Bitext complet + augmentation | 99.7 % / 99.7 % (n=1000) | **59.3 %** / 55.3 % (n=54) | 2 ms |
| slm_lora | 4000 exemples, 1 époque, cpu | 98.8 % / 98.8 % (n=1000) | **44.4 %** / 40.3 % (n=54) | 155 ms |
| slm_lora+augmentation | 4000 exemples, 1 époque, cpu | 98.2 % / 98.2 % (n=1000) | **68.5 %** / 64.1 % (n=54) | 215 ms |
| zero_shot:Qwen2.5-0.5B-Instruct | aucun (prompt) | 27.0 % / 23.2 % (n=100) | **20.4 %** / 20.5 % (n=54) | 1513 ms |
| zero_shot:Qwen2.5-1.5B-Instruct | aucun (prompt) | 59.0 % / 47.9 % (n=100) | **53.7 %** / 49.5 % (n=54) | 6602 ms |

## Recherche RAG (54 tickets réalistes, all-MiniLM-L6-v2)

| Requête | Recherche | hit@1 | hit@3 | MRR@3 | Latence médiane |
|---|---|---|---|---|---|
| texte seul | vectoriel | 66.7 % | 87.0 % | 0.756 | 12.7 ms |
| texte seul | vectoriel + reranking | 59.3 % | 88.9 % | 0.725 | 399.0 ms |
| texte + intention prédite | vectoriel | 66.7 % | 88.9 % | 0.772 | 19.3 ms |
| texte + intention prédite | vectoriel + reranking | 74.1 % | 88.9 % | 0.812 | 464.4 ms |
| texte + intention réelle (plafond) | vectoriel | 81.5 % | 100.0 % | 0.898 | 18.8 ms |
| texte + intention réelle (plafond) | vectoriel + reranking | 90.7 % | 100.0 % | 0.954 | 411.9 ms |

## Calibration des vérificateurs (38 cas : moitié fidèles, moitié aux chiffres modifiés)

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

## Vérificateur sur de vraies réponses de l'agent (15 brouillons de Qwen2.5-1.5B annotés à la main : 11 fidèles, 4 faux)

| Vérificateur | Accepte les réponses fidèles | Rejette les réponses fausses | Exactitude équilibrée |
|---|---|---|---|
| xsmall : NLI + chiffres (toutes prémisses) | 27.3 % | 100.0 % | **63.6 %** |
| xsmall : NLI + chiffres + prémisses proches | 72.7 % | 50.0 % | **61.4 %** |
| base : NLI + chiffres (toutes prémisses) | 45.5 % | 75.0 % | **60.2 %** |
| base : NLI + chiffres + prémisses proches | 81.8 % | 50.0 % | **65.9 %** |

## Agent complet : comparaison des SLM générateurs (CPU 8 cœurs, sans GPU, classifieur lora_aug, vérificateur nli)

| SLM | Paramètres | RAM | Réponses acceptées | Dont au 1er essai | Confirmées par le juge | Similarité / référence | Latence médiane (p90) | Tokens/s |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-360M-Instruct | 362 M | 1406 Mo | 8.7 % | 0.0 % | 0.0 % | 0.524 | 46.4 s (62.0 s) | 7.2 |
| Qwen2.5-0.5B-Instruct | 494 M | 1935 Mo | 43.5 % | 13.0 % | 10.0 % | 0.559 | 31.2 s (52.5 s) | 3.5 |
| Qwen2.5-1.5B-Instruct | 1544 M | 4218 Mo | 26.1 % | 17.4 % | 16.7 % | 0.526 | 73.6 s (115.7 s) | 1.7 |

23 tickets sur 27 passent par le SLM, les autres sont escaladés directement (intention réservée aux humains ou confiance trop faible). Exactitude du classifieur sur ces tickets : 70.4 %. Appels à une API externe : 0.
