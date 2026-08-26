# Chapitre 41 — Ajuster et améliorer son modèle

## L'énoncé

1. Notez le score de la baseline (réglages par défaut).
2. Lancez un `GridSearchCV` sur `n_estimators` et `max_depth` d'une forêt.
3. Affichez les meilleurs réglages et le gain par rapport à la baseline.
4. Refaites avec `RandomizedSearchCV(n_iter=10)`. Aussi bon, plus vite ?
5. Vérifiez le modèle final **une seule fois** sur le test.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `ajuster_model.py` | Le script du chapitre. |
| `exercice_reglages.py` | Le corrigé : baseline, grille, aléatoire, vérification. |
| `test_exercice_reglages.py` | Les tests (≈ 20 s : plusieurs dizaines de forêts sont entraînées). |

```bash
pip install -r requirements.txt
python exercice_reglages.py
pytest -q
```

## Corrigé

### 1. La baseline

```text
baseline (reglages par defaut) : 0.9708
```

Deux points de méthode :

- La baseline se mesure **en validation croisée sur le train**, jamais sur le
  test. Sinon vous auriez déjà « consommé » votre examen final avant même de
  commencer à régler.
- Les réglages par défaut de Scikit-learn sont bons. Ce n'est pas un hasard :
  ils ont été choisis pour bien fonctionner dans la majorité des cas. Vous
  partez donc de 97 %, pas de zéro.

### 2-3. La recherche par grille

```text
recherche par grille (12 combinaisons)
  meilleurs reglages : {'max_depth': None, 'n_estimators': 50}
  score de validation : 0.9744
  gain : +0.0036
  duree : 7.0 s
```

**12 combinaisons** = 3 valeurs de `n_estimators` × 4 de `max_depth`. Chacune
évaluée en validation croisée à 5 plis, soit **60 entraînements** de forêt pour
7 secondes. Retenez la multiplication : elle explose vite.

```text
2 hyperparamètres × 4 valeurs × 5 plis  =   80 entraînements
4 hyperparamètres × 5 valeurs × 5 plis  = 3125 entraînements
```

C'est pour cela qu'on ne met pas « tout » dans une grille : on choisit deux ou
trois hyperparamètres qui comptent vraiment.

**Le gain : +0,0036.** Trois millièmes. Autant le dire franchement : sur ce
jeu de données, **le réglage ne sert presque à rien**. Et c'est un résultat
utile, pas un échec de l'exercice — il illustre la hiérarchie réelle des
leviers :

| Levier | Gain typique |
|---|---|
| De meilleures données, plus de données | +10 à +30 points |
| Des features mieux construites | +5 à +15 points |
| Un modèle mieux adapté à la tâche | +2 à +10 points |
| **Le réglage des hyperparamètres** | **+0,1 à +2 points** |

Le réglage est un **affinage de fin de parcours**. Si votre modèle plafonne à
70 %, aucune grille ne le sauvera : le problème est dans les données.

### 4. La recherche aléatoire

```text
recherche aleatoire (10 tirages)
  meilleurs reglages : {'n_estimators': 50, 'max_depth': None}
  score de validation : 0.9744
  ecart avec la grille : +0.0000
```

**Même résultat, moins d'essais.** Sur une petite grille de 12, l'économie est
symbolique. Elle devient décisive dès que l'espace grandit :

| Espace de recherche | Grille exhaustive | Recherche aléatoire |
|---|---|---|
| 12 combinaisons | 12 essais | 10 essais — peu d'intérêt |
| 3 125 combinaisons | 3 125 essais | 60 essais, souvent à 1 % du meilleur |
| Un paramètre continu (ex. `learning_rate`) | **impossible** | naturel : on tire dans une plage |

L'argument théorique, dû à Bergstra et Bengio (2012) : dans la plupart des
problèmes, **seuls un ou deux hyperparamètres comptent vraiment**. Une grille
gaspille ses essais à faire varier les inutiles ; un tirage aléatoire explore
davantage de valeurs distinctes des paramètres qui comptent.

### 5. La vérification finale

```text
verification finale sur le test : 1.0000
```

Une seule ligne, à la toute fin, **après** que tous les choix ont été faits.

Pourquoi cette discipline ? Parce que si vous regardez le test après chaque
essai et que vous gardez « celui qui marche le mieux dessus », vous réglez vos
hyperparamètres **sur le test**. Il cesse alors d'être un examen : il devient
une seconde validation, et votre estimation de performance est fausse — trop
optimiste, et vous ne le saurez qu'en production.

Les trois jeux, et leur rôle :

| Jeu | Sert à | Combien de fois on le regarde |
|---|---|---|
| Entraînement | ajuster les **paramètres** (les poids) | en permanence |
| Validation | choisir les **hyperparamètres** | à chaque essai |
| **Test** | estimer la performance réelle | **une fois** |

### Paramètres ou hyperparamètres ?

| | Paramètres | Hyperparamètres |
|---|---|---|
| Qui les fixe | l'algorithme, pendant `fit` | **vous**, avant `fit` |
| Exemples | poids d'un réseau, seuils d'un arbre | `n_estimators`, `max_depth`, `learning_rate` |
| Combien | des milliers à des milliards | quelques-uns |
| Comment les régler | descente de gradient | essais et validation croisée |

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Les paramètres sont appris par le modèle ; les hyperparamètres sont fixés par vous avant l'entraînement. |
| 2 | **a** | La recherche par grille teste toutes les combinaisons d'une liste de valeurs. |
| 3 | **b** | La recherche aléatoire trouve souvent un bon réglage plus vite, en explorant plus largement. |
| 4 | **b** | Sur la validation, jamais sur le test. |

## Ce qu'il faut retenir

Réglez sur la validation, vérifiez sur le test une seule fois — et gardez la
mesure : quelques millièmes gagnés par une grille ne valent pas une heure de
travail sur les données.
