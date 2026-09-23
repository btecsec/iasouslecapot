# Chapitre 33 — L'art du découpage

## L'énoncé

Reprenez vos données nettoyées (features X, `species` en y) :

1. Découpez en 80 % / 20 % avec `random_state=42`.
2. Affichez la taille de chaque jeu.
3. Utilisez `stratify=y` et vérifiez que les proportions sont conservées.
4. Expliquez pourquoi il faut normaliser **après** le découpage.
5. Lancez une validation croisée à 5 plis sur un modèle simple.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `split_data.py` | Le script du chapitre. |
| `donnees.py` | Le chargement des manchots. |
| `exercice_decoupage.py` | Le corrigé, une fonction par question. |
| `test_exercice_decoupage.py` | Les tests, dont la démonstration chiffrée de `stratify`. |

```bash
pip install -r requirements.txt
python exercice_decoupage.py
pytest -q
```

## Corrigé, question par question

### 1-2. Le découpage 80/20

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

```text
train : 273 exemples
test  :  69 exemples        (342 lignes exploitables sur 344)
```

`random_state=42` n'a rien de magique : c'est une graine. Son rôle est la
**reproductibilité** — même découpage à chaque exécution, donc scores
comparables entre deux essais. Sans elle, vous ne sauriez jamais si un gain de
1 % vient de votre amélioration ou du tirage au sort.

### 3. L'effet de `stratify=y`

Voici la mesure, sur le vrai dataset :

| | Adelie | Gentoo | Chinstrap | Écart max avec le jeu complet |
|---|---|---|---|---|
| Dataset complet | 44,2 % | 36,0 % | 19,9 % | — |
| Test **sans** stratify | 50,7 % | 31,9 % | 17,4 % | **6,5 points** |
| Test **avec** stratify | 43,5 % | 36,2 % | 20,3 % | **0,7 point** |

Sans stratification, le hasard a mis 6,5 points d'Adelie en trop dans le jeu de
test. Sur 69 exemples, c'est mécanique : plus le jeu de test est petit, plus le
tirage est instable. Et une classe rare peut carrément disparaître du test —
auquel cas vous ne mesurez plus rien sur elle.

**Règle** : en classification, `stratify=y` par défaut. Il n'y a pas de cas où
cela nuit.

### 4. Pourquoi normaliser après le découpage ?

> Parce que `StandardScaler` calcule une moyenne et un écart-type : si on les
> calcule sur toutes les données, ces deux nombres contiennent une information
> venue du jeu de test, qui se retrouve injectée dans l'entraînement. C'est
> une **fuite de données**, et le score obtenu est trop optimiste.

En pratique :

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit_transform : sur le train
X_test = scaler.transform(X_test)          # transform seul : sur le test
```

`fit_transform` sur le test est l'erreur classique. Retenez la règle : **on
n'apprend jamais rien sur le jeu de test, on ne fait que l'utiliser.** La
solution qui rend l'erreur impossible est le `Pipeline` (chapitre 35).

### 5. Validation croisée à 5 plis

```python
modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(modele, X, y, cv=5)
```

```text
[1.0000  0.9855  0.9706  1.0000  1.0000]
moyenne 0.9912  (± 0.0117)
```

Trois choses à lire dans ces cinq nombres :

- **La moyenne (99,1 %)** est une estimation bien plus fiable qu'un score
  unique : elle repose sur 5 découpages différents, et chaque exemple a servi
  une fois au test.
- **L'écart-type (1,2 %)** dit la stabilité. Petit = le modèle se comporte
  pareil partout. S'il était de 15 %, le score moyen ne voudrait pas dire
  grand-chose.
- **Le pipeline dans `cross_val_score`** garantit que le scaler est réajusté
  sur le train de *chaque* pli. Normaliser avant l'appel aurait produit un
  score légèrement — et faussement — meilleur.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Tester sur les données d'entraînement mesure la mémorisation, pas la généralisation. |
| 2 | **c** | Le jeu de test estime la performance réelle, une seule fois, à la fin. |
| 3 | **b** | Une fuite de données, c'est une information du test qui contamine l'entraînement. |
| 4 | **b** | La validation croisée découpe en 5, entraîne 5 fois et moyenne les scores. |

## Ce qu'il faut retenir

Le jeu de test est un examen : on l'ouvre une fois, à la fin. Tout ce qui se
règle — normalisation, hyperparamètres, choix de modèle — se règle sur le
train, par validation croisée.
