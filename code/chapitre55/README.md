# Chapitre 55 — Projet pratique — construire un assistant RAG

## L'énoncé

1. Prenez un document que vous connaissez et sauvegardez-le en `.txt`.
2. Découpez-le, affichez le nombre de morceaux. Ajustez `chunk_size` : quel
   effet ?
3. Indexez, puis posez trois questions dont vous connaissez la réponse.
4. Posez une question hors sujet : l'assistant répond-il « je ne sais pas » ?
5. **Défi** : affichez, sous chaque réponse, les passages sources utilisés.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `assistant_rag.py` | Le pipeline complet : découper, indexer, retrouver, générer, oublier. |
| `test_assistant_rag.py` | 30 tests en 1,4 s — sans télécharger un seul modèle. |

```bash
pip install -r requirements.txt
python assistant_rag.py
pytest -q
```

### Deux briques volontairement remplaçables

| Brique | Ici | En production |
|---|---|---|
| Vectorisation | TF-IDF (scikit-learn) | un modèle d'embeddings (`all-MiniLM-L6-v2`, `text-embedding-3`) |
| Génération | extractive : on rend le passage tel quel | un appel LLM, via le paramètre `generateur` |

Le reste du code — découpage, index, recherche, garde-fou, sources — est
**identique** dans les deux cas. C'est ce découplage qui rend un RAG testable :
on vérifie la tuyauterie, jamais la créativité du modèle.

## Corrigé

### 2. L'effet de `chunk_size`

```text
taille=100   ->  9 morceaux
taille=250   ->  5 morceaux
taille=500   ->  2 morceaux
```

| Réglage | Ce que ça donne |
|---|---|
| **Trop petit** (< 150) | la recherche trouve la bonne phrase, mais amputée de son contexte : « huit heures en usage normal » sans savoir de quoi on parle |
| **Trop grand** (> 1000) | le bon passage est noyé dans du texte sans rapport, et le contexte du LLM se remplit vite |
| **Le bon ordre de grandeur** | 300 à 800 caractères pour de la documentation ; à ajuster selon la densité du texte |

Deux détails d'implémentation, invisibles dans l'énoncé, qui changent
réellement la qualité :

1. **Couper aux frontières naturelles** (paragraphe → phrase → mot), jamais au
   caractère près.
2. **Réaligner le chevauchement sur un espace.** Sans cela, un morceau
   commence par « teur de 5000 mAh » — inutilisable, et le vecteur produit est
   du bruit. Un test le verrouille.

### 3. Trois questions dont on connaît la réponse

```text
Q : Comment reinitialiser le mot de passe ?
R : Pour réinitialiser le mot de passe, maintenez le bouton Reset enfoncé...
    sources : manuel_robotx.txt #1 (score 0.30)

Q : Combien de temps dure la batterie ?
R : ...L'autonomie annoncée est de huit heures en usage normal...
    sources : manuel_robotx.txt #1 (score 0.18)

Q : Quelle est la duree de la garantie ?
R : ...La garantie constructeur couvre deux ans, pièces et main-d'œuvre.
    sources : manuel_robotx.txt #1 (score 0.16)
```

Notez les scores : entre 0,16 et 0,30. Ce ne sont pas des probabilités, mais
des **similarités cosinus** — ce qui compte est leur écart avec les questions
hors sujet.

### 4. La question hors sujet — et le garde-fou

```text
Q : Quelle est la capitale de l'Australie ?
R : Je ne sais pas.   (meilleur score 0.000)
```

Trois choses valent la peine d'être détaillées.

**a) Le garde-fou agit *avant* le modèle.** Si aucun passage ne dépasse le
seuil, on renvoie « Je ne sais pas. » sans jamais appeler le LLM. C'est ce qui
rend le refus fiable : un modèle à qui l'on donne un contexte hors sujet
essaiera *quand même* de répondre. Un test vérifie que le générateur n'est pas
appelé du tout.

**b) Il a fallu retirer les mots vides.** Première version : la question sur
l'Australie obtenait un score de **0,185** et passait le seuil ! En cause,
« quelle est la » qui ressemble à n'importe quel paragraphe français. Une fois
les mots vides retirés, le score tombe à **0,000**.

C'est une leçon générale : **un garde-fou se calibre avec des questions hors
sujet**, pas seulement avec des questions pertinentes. Testez-le à l'envers.

**c) Le seuil est un curseur, et il dépend de vos documents.**

| Seuil | Conséquence |
|---|---|
| trop bas | l'assistant répond à tout, y compris n'importe quoi |
| trop haut | il refuse des questions auxquelles il pouvait répondre |
| ici : 0,10 | les questions pertinentes sont ≥ 0,16, les hors-sujet à 0,00 — la marge est nette des deux côtés |

Et la consigne dans le prompt double le garde-fou :

```text
Réponds UNIQUEMENT à partir du contexte ci-dessous.
Si la réponse ne s'y trouve pas, réponds exactement "Je ne sais pas."
CONTEXTE : ...
Rappel : si le contexte ne contient pas la réponse, réponds "Je ne sais pas."
```

La consigne est répétée **avant et après** le contexte. Un modèle suit
d'autant mieux une contrainte qu'elle encadre les données — surtout quand le
contexte est long.

### 5. Défi : afficher les sources

Chaque réponse porte ses `Passage` : texte, score, document d'origine, rang.

```python
reponse = assistant.repondre("Comment contacter le support ?")
for passage in reponse.sources:
    print(f"[{passage.source} #{passage.rang}] score {passage.score:.2f}")
```

Ce n'est pas cosmétique. Les sources sont ce qui distingue un RAG d'un LLM
bavard : l'utilisateur peut **vérifier**. Et lorsqu'une réponse est fausse,
elles disent immédiatement si le problème vient de la recherche (mauvais
passage retrouvé) ou de la génération (bon passage, mauvaise lecture) — deux
pannes qui ne se réparent pas au même endroit.

### Bonus : mettre à jour un document

```python
assistant.oublier("manuel_robotx.txt")          # retire ses morceaux
assistant.indexer(nouveau_texte, source="manuel_robotx.txt")
```

C'est le défi du chapitre 53, rendu concret. D'où la nécessité de **stocker la
source avec chaque morceau** : sans elle, impossible de savoir quoi retirer.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Pour retrouver un passage précis, et tenir dans le contexte du LLM. |
| 2 | **b** | `similarity_search` retrouve les morceaux les plus proches **par le sens** de la question. |
| 3 | **b** | L'indexation se fait une fois par lot de documents ; seule l'interrogation se rejoue. |
| 4 | **b** | Exiger « je ne sais pas » quand le contexte ne contient pas la réponse. |

## Ce qu'il faut retenir

Un RAG utile tient sur trois piliers : un découpage soigné, un garde-fou
calibré **avec des questions hors sujet**, et des sources affichées. Le modèle
de génération, lui, est la pièce la plus facile à changer.
