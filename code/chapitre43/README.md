# Chapitre 43 — Projet pratique — entraîner un modèle de A à Z

## L'énoncé

1. Rejouez les six étapes sur les manchots, sans recopier aveuglément.
2. Notez le score de la baseline, puis après réglage. Combien avez-vous gagné ?
3. Rechargez le modèle sauvegardé dans un nouveau script et vérifiez.
4. **Défi** : refaites toute la chaîne sur un autre dataset (iris, titanic).
   La démarche tient-elle ?
5. Écrivez, en trois phrases, la recette générale d'un projet de ML.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `projet_manchots.py` | Les six étapes, dans un script réutilisable. |
| `test_projet_manchots.py` | Les tests, dont la chaîne rejouée sur iris. |

```bash
pip install -r requirements.txt
python projet_manchots.py          # sur les manchots
python projet_manchots.py --iris   # le défi, sur un autre jeu
pytest -q
```

## Corrigé

### 1. Les six étapes

```text
1. exploration : 344 lignes, 7 colonnes, 19 valeurs manquantes,
                 classe majoritaire 0.44
2. nettoyage   : 9 colonnes apres encodage
3. decoupage   : 80/20 stratifie, graine 42
4. baseline    : 1.0000
   apres reglage : 1.0000
5-6. modele sauvegarde, rechargement fidele : True
```

Le point de conception le plus important est invisible à la lecture du
résultat : **`projet_complet(df, cible)` ne contient pas une seule ligne qui
parle de manchots.** Elle détecte les colonnes numériques, les colonnes
textuelles, comble les trous selon le type, encode, découpe, entraîne. C'est
ce qui rend le défi de la question 4 possible.

### 2. Le gain du réglage

**+0,0000.** Zéro.

C'est la bonne réponse, et il ne faut pas la maquiller : sur les manchots, la
baseline atteint déjà 100 % sur le jeu de test. Aucune grille ne peut faire
mieux que parfait.

Le chiffre à retenir n'est donc pas le gain, mais le **plancher** : 0,44,
c'est-à-dire ce qu'obtient un modèle qui répond toujours « Adelie ». C'est
face à cette valeur que se juge une performance, jamais dans l'absolu.

Sur iris, le même code donne quelque chose de plus intéressant :

```text
apres reglage : 0.9667
meilleurs reglages : {'randomforestclassifier__max_depth': 3}
gain : +0.0667  (a comparer au plancher de 0.33)
```

Ici le réglage rapporte **6,7 points** — parce que `max_depth=3` limite le
surapprentissage sur un jeu de 150 exemples. Même code, deux datasets, deux
conclusions opposées : c'est exactement pour cela qu'on mesure au lieu de
supposer.

### 3. Le rechargement

```python
paquet = joblib.load("modele_final.joblib")
predictions = paquet["modele"].predict(X[paquet["colonnes"]])
```

Deux choses dans le fichier : le **pipeline complet** (scaler inclus) et
l'**ordre des colonnes**. Les tests vérifient que le rechargement redonne les
mêmes prédictions, et qu'un ordre de colonnes mélangé ne change rien.

*(Rappel du chapitre 42 : `joblib.load` exécute du code. Ne chargez que des
fichiers que vous avez produits vous-même.)*

### 4. Le défi : la même chaîne sur iris

```bash
python projet_manchots.py --iris
```

```text
1. exploration : 150 lignes, 5 colonnes, 0 manquants,
                 classe majoritaire 0.33
4. baseline    : 0.9000  ->  apres reglage : 0.9667
```

**La démarche tient**, sans modifier une ligne. Et c'est le vrai enseignement
de la partie IV : ce que vous avez appris n'est pas « comment classer des
manchots », c'est une **méthode**. Le contenu de chaque étape change ; la
succession des étapes, non.

Ce qui change d'un dataset à l'autre :

| | Manchots | Iris | Titanic |
|---|---|---|---|
| Valeurs manquantes | 19 | 0 | beaucoup (âge, cabine) |
| Colonnes textuelles | 3 | 0 | plusieurs (nom, billet, port) |
| Classes | 3, déséquilibrées | 3, parfaites | 2, déséquilibrées |
| Plancher | 0,44 | 0,33 | 0,62 |
| Gain du réglage | 0 | +6,7 pts | dépend des features créées |

Sur Titanic, une étape supplémentaire deviendrait payante : construire des
features (titre extrait du nom, taille de la famille). C'est le seul vrai
ajout — et il relève de la connaissance du domaine, pas de l'algorithme.

### 5. La recette générale, en trois phrases

> 1. **Regarder les données avant de coder**, et corriger ce qui cloche : un
>    modèle n'est jamais meilleur que ce qu'on lui donne.
> 2. **Isoler un jeu de test dès le départ** et n'y toucher qu'à la fin : c'est
>    la seule mesure honnête de ce que vaudra le modèle en production.
> 3. **Partir du modèle le plus simple**, mesurer contre une baseline, et ne
>    complexifier que si la mesure le justifie.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Tout projet commence par explorer et comprendre les données. |
| 2 | **b** | Le jeu de test sert une seule fois, à la fin. |
| 3 | **b** | `random_state` rend le découpage et l'entraînement reproductibles. |
| 4 | **b** | Le `classification_report` détaille précision, rappel et F1 **par classe** — l'accuracy globale peut masquer une classe entièrement ratée. |

## Ce qu'il faut retenir

Vous savez maintenant mener un projet de bout en bout, et le refaire sur
n'importe quel jeu de données tabulaire. Ce qui suit — partie V — répond à
la question restante : comment le faire tourner ailleurs que sur votre
machine.
