# Chapitre 10 — Python pour la data : NumPy et Pandas

## L'énoncé

1. Créez un DataFrame avec deux colonnes : `produit` (« pain », « lait »,
   « œufs ») et `prix` (1.2, 0.9, 2.5).
2. Affichez-le, puis affichez la colonne `prix` seule.
3. Calculez le prix moyen avec `df["prix"].mean()`.
4. Filtrez les produits dont le prix dépasse 1 euro.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_courses.py` | Le corrigé exécutable, une fonction par question. |
| `test_exercice_courses.py` | Les tests unitaires du corrigé. |
| `requirements.txt` | pandas, numpy, pytest. |

```bash
pip install -r requirements.txt
python exercice_courses.py
pytest -q
```

## Corrigé commenté

```python
import pandas as pd

df = pd.DataFrame({
    "produit": ["pain", "lait", "œufs"],
    "prix": [1.2, 0.9, 2.5],
})
```

Un DataFrame se construit à partir d'un **dictionnaire** : une clé = une
colonne. Les listes doivent avoir la même longueur, sinon pandas refuse.

```python
print(df)              # le tableau entier
print(df["prix"])      # une colonne = une Series
print(df["prix"].mean())          # 1.5333333333333332
print(df[df["prix"] > 1])         # le filtrage booléen
```

### Le point qui bloque tout le monde : `df[df["prix"] > 1]`

Lisez-le de l'intérieur vers l'extérieur :

1. `df["prix"] > 1` ne renvoie pas un booléen mais **une Series de booléens**,
   une par ligne : `[True, False, True]`. C'est un *masque*.
2. `df[masque]` garde les lignes où le masque vaut `True`.

D'où l'erreur classique : `df[df["prix"] > 1 and df["prix"] < 2]` échoue avec
`ValueError: The truth value of a Series is ambiguous`. Python ne sait pas
transformer une Series de 3 booléens en un seul `True`/`False`. Il faut les
opérateurs *élément par élément*, avec des parenthèses obligatoires :

```python
df[(df["prix"] > 1) & (df["prix"] < 2)]   # & et non and
df[(df["prix"] < 1) | (df["prix"] > 2)]   # | et non or
```

### Réponses chiffrées

| Question | Résultat |
|---|---|
| Prix moyen | `1.5333...` — soit `(1.2 + 0.9 + 2.5) / 3` |
| Produits > 1 € | `pain` (1.2) et `œufs` (2.5) |
| Type de `df["prix"]` | `pandas.Series` |
| Type de `df[["prix"]]` | `pandas.DataFrame` (double crochet = sous-tableau) |

### Le lien avec NumPy

`df["prix"].mean()` ne fait pas une boucle Python : pandas délègue à NumPy,
qui calcule en C sur tout le tableau d'un coup. C'est la *vectorisation*, et
c'est pour cela qu'une opération sur un million de lignes reste instantanée
alors que la même boucle `for` prendrait plusieurs secondes (chapitre 8).

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le calcul vectorisé : `tableau * 2` agit sur tout le tableau, sans boucle, et en C. |
| 2 | **b** | Un DataFrame est un tableur programmable : lignes et colonnes nommées, pilotées par du code. |
| 3 | **c** | `df.head()` affiche les 5 premières lignes (`df.head(10)` pour 10). |
| 4 | **b** | `df.describe()` sort le résumé statistique des colonnes numériques : count, mean, std, min, quartiles, max. |

## Ce qu'il faut retenir

NumPy fournit le tableau rapide, pandas lui ajoute des noms de colonnes et des
outils de tableur. Le filtrage booléen (`df[df["colonne"] > valeur]`) est le
geste que vous répéterez le plus dans toute la partie IV.
