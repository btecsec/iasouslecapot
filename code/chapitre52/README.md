# Chapitre 52 — Spécialiser un modèle par le fine-tuning

## L'énoncé

1. Choisissez un besoin concret (ex. : un assistant qui répond dans un style
   donné).
2. Rédigez cinq exemples « entrée → sortie souhaitée ».
3. Ce besoin relève-t-il du prompt, du fine-tuning ou du RAG ? Justifiez.
4. Expliquez ce que LoRA évite de faire, et pourquoi c'est moins coûteux.
5. **Défi** : listez les cinq étapes d'un fine-tuning et reliez chacune à un
   chapitre déjà vu.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `reponse_exercice.md` | Le corrigé rédigé de l'exercice (cas du concierge de luxe). |
| `data/train.jsonl`, `data/test.jsonl` | Les exemples d'entraînement et d'évaluation. |
| `finetune_lora.py` | L'entraînement LoRA. |
| `test_finetuned_model.py`, `test_if_gpu_installed.py` | Des **scripts** à lancer à la main (ils chargent un modèle). |
| `test_donnees_finetuning.py` | Les vrais tests : ils valident les **données**, sans GPU ni torch. |
| `concierge_model/` | L'adaptateur LoRA produit. |

```bash
pytest -q                          # valide les données (50 ms, aucun GPU)
python test_if_gpu_installed.py    # vérifie votre matériel
python finetune_lora.py            # l'entraînement (long, GPU conseillé)
python test_finetuned_model.py     # essaie le modèle spécialisé
```

Les deux scripts nommés `test_*` sont ceux du chapitre. Comme leur nom trompe
pytest, `pyproject.toml` les écarte explicitement de la collecte.

## Corrigé

Le corrigé complet des questions 1 à 5 est dans **`reponse_exercice.md`**
(cas retenu : un assistant « concierge de luxe »). Ce README ajoute ce que
l'exercice ne demande pas et qui décide de la réussite : la **qualité des
données**.

### Pourquoi tester les données avant de lancer un entraînement

Un fine-tuning rate presque toujours à cause des données, pas du code. Et
l'échec est cher : plusieurs heures de GPU pour découvrir qu'une ligne du
JSONL était mal formée. Les 14 tests du dossier vérifient en 50 millisecondes :

| Vérification | L'erreur qu'elle attrape |
|---|---|
| Chaque ligne est un JSON valide | un JSONL rédigé comme un tableau JSON (`[{...},{...}]`) — l'erreur n° 1 |
| Les champs `instruction` et `output` existent et sont non vides | une ligne tronquée en fin de fichier |
| Aucun doublon dans `train.jsonl` | un exemple compté double, sur-représenté dans l'apprentissage |
| **Aucune fuite entre train et test** | la fuite du chapitre 33, version LLM : évaluation flatteuse et fausse |
| Les sorties font au moins 5 mots | des réponses d'un mot n'enseignent aucun style |
| Aucun tutoiement dans un corpus de vouvoiement | **le style doit être cohérent** — sinon le modèle apprend l'hésitation |
| Pas de `Ã©` à la place de `é` | un problème d'encodage se retrouve tel quel dans les poids |

Le test sur la cohérence du style mérite qu'on s'y arrête : le fine-tuning
apprend un **comportement moyen**. Si un exemple sur cinq tutoie, le modèle
tutoiera une fois sur cinq — et vous mettrez longtemps à comprendre pourquoi.

### Question 3, sous forme de règle de décision

| Le besoin porte sur… | Technique | Coût |
|---|---|---|
| la **forme** (ton, style, format constant) | **fine-tuning** | données annotées + calcul |
| le **fond** (connaissances, documents, actualité) | **RAG** | infrastructure d'indexation |
| une **consigne ponctuelle** | **prompt** | gratuit |

Le piège classique : vouloir « apprendre nos documents au modèle » par
fine-tuning. Cela ne fonctionne pas — le fine-tuning déplace un comportement,
il n'installe pas une base de connaissances consultable. Et il ne cite aucune
source. C'est le rôle du RAG (chapitre 53).

### Question 4 : ce que LoRA évite de faire

Un fine-tuning classique met à jour **tous** les poids : pour un modèle de
7 milliards de paramètres, il faut garder en mémoire les poids, leurs
gradients et les états de l'optimiseur — de l'ordre de 70 à 100 Go de VRAM.
Hors de portée.

LoRA **gèle le modèle de base** et n'entraîne que deux petites matrices
ajoutées à côté de certaines couches. Le nombre de paramètres entraînés tombe
à moins de 1 % du total.

Trois conséquences pratiques :

| | Fine-tuning complet | LoRA |
|---|---|---|
| Mémoire GPU | des dizaines de Go | quelques Go (souvent une carte grand public suffit) |
| Fichier produit | tout le modèle (plusieurs Go) | un **adaptateur** de quelques Mo |
| Plusieurs spécialisations | autant de copies complètes | un modèle de base + N petits adaptateurs interchangeables |

Le dernier point est le plus sous-estimé : vous pouvez charger le même modèle
de base une fois et brancher l'adaptateur « concierge », « support
technique » ou « juridique » selon la requête. Un test du dossier vérifie
d'ailleurs que l'adaptateur produit pèse bien moins de 100 Mo.

### Question 5 : les cinq étapes, et leurs chapitres

| Étape | Ce que vous faites | Chapitre |
|---|---|---|
| 1. Rassembler des exemples de qualité | quelques centaines de paires entrée/sortie, cohérentes | 32 (préparer les données) |
| 2. Les formater et les découper | JSONL, train/test **sans fuite** | 33 (découpage), et ce chapitre pour le format |
| 3. Lancer l'entraînement | perte, optimiseur, taux d'apprentissage **petit** | 38 (comprendre l'entraînement) |
| 4. Évaluer | sur le jeu de test, plus une lecture humaine | 40 (évaluer) |
| 5. Sauvegarder et servir | l'adaptateur + la configuration + le tokenizer | 42 (sauvegarder), 45 (API) |

Le point le plus délicat est le taux d'apprentissage de l'étape 3 : trop
grand, le modèle **oublie** ce qu'il savait faire au lieu de s'adapter :
c'est l'**oubli catastrophique**.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le fine-tuning poursuit l'entraînement d'un modèle existant sur vos propres données. |
| 2 | **b** | C'est bien moins coûteux qu'un entraînement de zéro, et à votre portée. |
| 3 | **b** | Le fine-tuning n'apporte pas de connaissances à jour ni de documents précis : c'est le rôle du RAG. |
| 4 | **b** | LoRA n'entraîne qu'un petit ensemble de paramètres, en figeant le reste. |

## Ce qu'il faut retenir

Le fine-tuning change le **comportement**, pas les **connaissances**. Sa
réussite se joue dans la qualité et la cohérence de quelques centaines
d'exemples — testez-les avant de payer une heure de GPU.
