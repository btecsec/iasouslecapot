# Chapitre 53 — Donner de la mémoire au modèle avec le RAG

## L'énoncé

1. Pourquoi un LLM seul ne peut-il pas répondre sur vos documents privés ?
2. Décrivez les six étapes d'un RAG avec vos propres mots.
3. Donnez deux phrases de sens proche mais sans mot commun : un embedding les
   rapprocherait-il ?
4. Pour trois besoins concrets, choisissez RAG ou fine-tuning.
5. **Défi** : comment mettre à jour un assistant RAG quand un document change ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_rag.py` | Le RAG complet du chapitre (LangChain + modèle local). |
| `mon_document.txt` | Le document d'exemple. |
| `decoupage.py` | Le découpage isolé, **sans dépendance** — la brique la plus décisive. |
| `test_decoupage.py` | 19 tests en 20 ms, sans télécharger un seul modèle. |

```bash
python decoupage.py           # l'effet de chunk_size, chiffré
pytest -q test_decoupage.py   # les tests rapides

pip install -r requirements.txt
python exercice_rag.py        # le RAG complet (télécharge un modèle)
```

## Corrigé

### 1. Pourquoi un LLM seul ne peut pas répondre sur vos documents

> Ses connaissances sont figées dans ses poids au moment de l'entraînement, et
> vos documents privés n'y étaient pas. Il ne peut donc pas les connaître — et,
> pire, il ne peut pas savoir qu'il ne les connaît pas : interrogé quand même,
> il produira une réponse plausible et fausse.

Le second point est le vrai danger. Un moteur de recherche répond « aucun
résultat » ; un LLM, lui, répond toujours quelque chose.

### 2. Les six étapes d'un RAG

| # | Étape | En une phrase |
|---|---|---|
| 1 | **Charger** | lire les documents (txt, PDF, pages web) |
| 2 | **Découper** | les couper en morceaux de quelques centaines de caractères |
| 3 | **Vectoriser** | transformer chaque morceau en vecteur de sens (*embedding*) |
| 4 | **Indexer** | ranger ces vecteurs dans une base vectorielle |
| 5 | **Retrouver** | vectoriser la question, chercher les k morceaux les plus proches |
| 6 | **Générer** | donner ces morceaux au LLM avec la question, et lui demander de répondre **uniquement** à partir d'eux |

Les étapes 1 à 4 se font **une fois** (l'indexation) ; les étapes 5 et 6 se
rejouent **à chaque question** (l'interrogation). C'est la distinction qui
explique la question 3 du quiz — et qui rend la mise à jour si simple
(question 5).

### 3. Deux phrases proches sans mot commun

```text
A : Quelle est la durée de vie de la batterie ?
B : Combien de temps tient l'accumulateur avant recharge ?

mots significatifs partagés : 0
```

**Un embedding les rapprocherait, oui.** C'est exactement ce que la recherche
par mots-clés ne sait pas faire : sur ces deux phrases, elle renvoie zéro
résultat, alors qu'elles disent la même chose.

Un embedding place chaque phrase comme un point dans un espace à plusieurs
centaines de dimensions, appris de sorte que **la proximité géométrique
traduise la proximité de sens**. « Batterie » et « accumulateur » ont été
rencontrés dans les mêmes contextes pendant l'entraînement : leurs vecteurs
sont voisins.

C'est pour cela qu'on parle de **recherche sémantique** — et c'est la vraie
innovation du RAG par rapport à un `Ctrl+F` amélioré.

*(La démonstration est dans `decoupage.py` et vérifiée par
`test_deux_phrases_de_sens_proche_nont_aucun_mot_commun`.)*

### Le découpage : là où se jouent les mauvaises réponses

`chunk_size` a plus d'influence sur la qualité d'un RAG que le choix du modèle.
Voici l'effet mesuré sur un document :

```text
taille=100   ->  4 morceaux, moyenne  91.8 caracteres
taille=300   ->  2 morceaux, moyenne 185.0 caracteres
taille=500   ->  1 morceau,  moyenne 341.0 caracteres
```

| Réglage | Avantage | Inconvénient |
|---|---|---|
| **Petits morceaux** (100-200) | recherche très précise | la réponse est tronquée : le morceau contient la question, pas le contexte qui l'explique |
| **Gros morceaux** (1000+) | contexte complet | beaucoup de bruit autour de la réponse, et le contexte du LLM se remplit vite |
| **Le chevauchement** | une phrase coupée en deux reste lisible dans l'un des morceaux | quelques % de stockage en plus |

Deux détails d'implémentation qui font la différence, et que le corrigé
teste :

- **couper aux frontières naturelles** (paragraphe, puis phrase, puis mot)
  plutôt qu'au caractère près — un morceau qui commence par « teur de 5000mAh »
  est inutilisable ;
- **garantir que le découpage se termine** même sur un texte sans espace ni
  ponctuation. Un chevauchement supérieur ou égal à la taille produirait une
  boucle infinie : le corrigé lève une `ValueError`.

### 4. RAG ou fine-tuning ?

| Besoin | Choix | Pourquoi |
|---|---|---|
| Répondre sur 3 000 pages de documentation interne | **RAG** | de la connaissance, qui change ; et il faut pouvoir citer la source |
| Répondre toujours dans le ton de la marque | **fine-tuning** | du comportement, pas du savoir |
| Répondre sur le catalogue produits, mis à jour chaque semaine | **RAG** | réindexer prend une minute ; réentraîner, des heures |

La règle : **le fond change souvent → RAG. La forme doit être constante →
fine-tuning.** Et les deux se combinent très bien : un modèle fine-tuné pour le
ton, alimenté par un RAG pour les faits.

### 5. Défi : mettre à jour un document

C'est là que le RAG écrase le fine-tuning :

```text
1. Retirer de l'index les morceaux de l'ancienne version du document.
2. Redécouper et revectoriser la nouvelle version.
3. Les ajouter à l'index.
```

Quelques secondes pour un document, et **le modèle n'est pas touché**. À
comparer à un réentraînement : plusieurs heures de GPU, un nouveau jeu
d'exemples à annoter, et aucune garantie que l'ancienne information ait
disparu des poids.

Deux conseils de production :

- **stocker un identifiant de source et une version** avec chaque morceau,
  sinon impossible de savoir quoi supprimer ;
- **réindexer par lot** (la nuit, ou sur événement), pas à chaque sauvegarde
  d'un fichier.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **a** | RAG = *Retrieval-Augmented Generation*, génération augmentée par la recherche. |
| 2 | **b** | Retrouver les documents pertinents, puis générer la réponse en s'appuyant dessus. |
| 3 | **b** | Un embedding transforme un texte en vecteur qui capture son sens. |
| 4 | **c** | Le RAG s'impose pour des réponses fondées sur des documents à jour et vérifiables. |

## Ce qu'il faut retenir

Le RAG, c'est l'examen à livre ouvert : le modèle ne récite plus, il lit. Et
avant de comparer les modèles ou les bases vectorielles, soignez le découpage —
c'est lui qui décide de ce que le modèle aura sous les yeux.
