# Le code du livre — Sous le capot de l'IA

Un dossier par chapitre, **56 en tout**. Chaque dossier contient :

- un `README.md` avec l'**énoncé** de l'exercice et son **corrigé complet**,
  plus les réponses commentées du quiz de validation ;
- le **code** du chapitre et le corrigé exécutable, quand l'exercice en
  demande ;
- des **tests** `pytest` qui vérifient ce corrigé — y compris les chiffres
  annoncés dans le README.

```bash
cd chapitre35
pip install -r requirements.txt
python exercice_scikit.py
pytest -q
```

## Comment lire les corrigés

Les chapitres théoriques n'ont pas de code : leur README répond aux questions,
avec le raisonnement, les pièges et les erreurs fréquentes. Les chapitres
pratiques ont en plus une fonction par question, testée.

Les tests ne sont pas décoratifs. Ils sont écrits pour **échouer** si le
corrigé se trompe, et plusieurs d'entre eux démontrent un point du cours :

| Test | Ce qu'il démontre |
|---|---|
| `chapitre22` | l'attention à la main donne exactement le même résultat que la formule matricielle |
| `chapitre33` | `stratify=y` réduit l'écart de proportions de 6,5 à 0,7 point |
| `chapitre37` | sans `zero_grad()`, deux `backward()` doublent le gradient |
| `chapitre39` | un gros modèle sur peu de données surapprend, un petit non |
| `chapitre40` | un modèle nul obtient 80 % d'accuracy et 0 % de rappel |
| `chapitre42` | oublier le scaler change 55 prédictions sur 69, sans erreur |

## Les six parties

### Partie I — Prérequis et environnement (1-11)

| Ch. | Sujet | Code |
|---|---|---|
| 1 | Pourquoi apprendre l'IA | corrigé écrit |
| 2 | Les prérequis | corrigé écrit |
| 3 | Installer Python | `verifier_installation.py` |
| 4 | Les environnements virtuels | `verifier_venv.py` |
| 5 | pip et requirements.txt | `analyser_requirements.py` |
| 6 | Listes, ensembles, dictionnaires | `exercice_commandes.py` |
| 7 | Les classes | `exercice_documents.py` |
| 8 | Astuces Python | `demo_astuces.py`, `exercice_astuces.py` |
| 9 | Tester son code et son IA | projet complet avec `tests/` |
| 10 | NumPy et Pandas | `exercice_courses.py` |
| 11 | Matplotlib et Seaborn | `exercice_visualisation.py` |

### Partie II — Introduction à l'IA (12-22)

| Ch. | Sujet | Code |
|---|---|---|
| 12-17 | IA, familles, apprentissage, tâches, vocabulaire | corrigés écrits |
| 18 | Les maths de l'IA | `exercice_maths.py` — la couche du chapitre, à la main |
| 19-21 | Réseaux, applications, LLM | corrigés écrits |
| 22 | Dans le moteur du Transformer | `exercice_attention.py` — l'attention de « dort » |

### Partie III — Les frameworks de l'IA : PyTorch, TensorFlow, LangChain (23-29)

| Ch. | Sujet | Code |
|---|---|---|
| 23 | Panorama des librairies | corrigé écrit |
| 24 | Installer PyTorch | `exercice_pytorch.py` |
| 25 | Installer TensorFlow/Keras | `exercice_keras.py` |
| 26 | PyTorch ou TensorFlow | corrigé écrit |
| 27 | PyTorch, les bases | `exercice_bases_torch.py` — formes, `stack` vs `cat`, dérivée automatique |
| 28 | Entraîner sur Google Colab | corrigé écrit |
| 29 | LangChain et LangGraph | `mini_chaine.py` (le `\|` et le garde-fou, en Python pur) |

### Partie IV — Développer un modèle (30-43)

Tous ces chapitres travaillent sur le même jeu de données (les manchots de
Palmer) et partagent `donnees.py`, qui bascule sur un jeu **synthétique** de
même forme si le téléchargement échoue. Les tests tournent donc hors ligne.

| Ch. | Sujet | Code |
|---|---|---|
| 30 | Le cycle de vie | corrigé écrit |
| 31 | Explorer | `exercice_exploration.py` |
| 32 | Nettoyer | `exercice_nettoyage.py` |
| 33 | Découper | `exercice_decoupage.py` |
| 34 | Choisir un modèle | `exercice_modeles.py` (+ baseline) |
| 35 | Scikit-learn | `exercice_scikit.py` |
| 36 | Keras | `exercice_reseau_keras.py` |
| 37 | PyTorch | `exercice_reseau_pytorch.py` |
| 38 | Comprendre l'entraînement | `exercice_entrainement.py` |
| 39 | Le surapprentissage | `exercice_surapprentissage.py` |
| 40 | Évaluer | `exercice_metriques.py` |
| 41 | Ajuster | `exercice_reglages.py` |
| 42 | Sauvegarder | `exercice_sauvegarde.py` |
| 43 | Projet de A à Z | `projet_manchots.py` (rejouable sur iris) |

### Partie V — Production, MLOps (44-49)

| Ch. | Sujet | Code |
|---|---|---|
| 44 | Qu'est-ce que le MLOps | corrigé écrit |
| 45 | Exposer avec une API | `mon_api.py` + tests `TestClient` |
| 46 | Docker | `Dockerfile` + tests qui le relisent sans Docker |
| 47 | Le cloud | corrigé écrit (Cloud Run, secrets, budget) |
| 48 | Surveiller | `surveillance.py` (dérive, latence p95) |
| 49 | CI/CD | `api.py`, `test_api_ci.py`, `exemple_ci.yml` |

### Partie VI — Fine-tuning et RAG (50-56)

| Ch. | Sujet | Code |
|---|---|---|
| 50 | Comprendre les LLM | corrigé écrit |
| 51 | Piloter par le prompt | `prompts.py` — testé sans clé d'API |
| 52 | Fine-tuning (LoRA) | `finetune_lora.py` + tests des **données** |
| 53 | Le RAG | `exercice_rag.py`, `decoupage.py` |
| 54 | Les bases de données vectorielles | `base_vectorielle.py` — Chroma, métadonnées, filtres |
| 55 | Projet RAG | `assistant_rag.py` — pipeline complet, garde-fou compris |
| 56 | Limites et éthique | corrigé écrit |

## Environnements

Les chapitres n'ont pas tous les mêmes besoins. Chaque dossier concerné a son
`requirements.txt` ; installez au fur et à mesure plutôt que tout d'un coup.

| Famille | Dépendances |
|---|---|
| Python pur | aucune (chapitres 3, 4, 5, 29, 48) |
| Data | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` |
| Deep learning | `tensorflow` (ch. 25, 36, 39) ou `torch` (ch. 24, 27, 37) |
| Production | `fastapi`, `uvicorn`, `httpx` |

Les tests d'un chapitre dont la dépendance manque sont **ignorés proprement**
(`pytest.importorskip`), jamais en échec.

Pour PyTorch, préférez la version processeur — cinq fois plus légère :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Dépôt

<https://github.com/btecsec/iasouslecapot>
