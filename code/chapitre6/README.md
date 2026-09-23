# Chapitre 6 — Ranger ses données : listes, ensembles et dictionnaires

## L'énoncé

Vous récupérez les commandes d'une boutique, sous la forme d'une liste de
dictionnaires :

```python
commandes = [
    {"client": "Alice", "produit": "clavier", "prix": 45},
    {"client": "Bob", "produit": "souris", "prix": 25},
    {"client": "Alice", "produit": "écran", "prix": 180},
    {"client": "Chris", "produit": "souris", "prix": 25},
]
```

1. Affichez chaque commande numérotée à partir de 1, sous la forme
   `1. Alice a commandé un clavier (45 €)`.
2. Construisez l'ensemble des clients distincts, puis affichez combien ils sont.
3. Construisez un dictionnaire `total_par_client` qui associe à chaque client la
   somme de ses commandes.
4. Trouvez le produit le plus cher, puis affichez le nom de son acheteur.
5. Répondez à « Alice a-t-elle commandé une souris ? » sans boucle `for`.

*Bonus : les prix montent de 10 %. Écrivez la nouvelle liste sans modifier
l'originale.*

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_commandes.py` | Le corrigé exécutable, une fonction par question. |
| `test_exercice_commandes.py` | Les tests unitaires du corrigé. |
| `requirements.txt` | pytest seulement : le chapitre n'utilise que la bibliothèque standard. |

```bash
pip install -r requirements.txt
python exercice_commandes.py
pytest -q
```

## Corrigé commenté

### 1. Numéroter avec `enumerate`

```python
[
    f"{numero}. {c['client']} a commandé un {c['produit']} ({c['prix']} €)"
    for numero, c in enumerate(commandes, start=1)
]
```

`start=1` évite l'éternel `indice + 1`. Notez les guillemets **simples** à
l'intérieur de la f-string : les doubles fermeraient la chaîne trop tôt.

### 2. Dédoublonner sans y penser

```python
{c["client"] for c in commandes}     # -> {'Alice', 'Bob', 'Chris'}
```

Alice passe deux commandes et n'apparaît qu'une fois. Aucun `if`, aucun test :
c'est la structure qui fait le travail. C'est tout l'intérêt du `set`.

### 3. Le point qui bloque : `.get(client, 0)`

```python
totaux = {}
for c in commandes:
    totaux[c["client"]] = totaux.get(c["client"], 0) + c["prix"]
```

À la **première** commande d'un client, la clé n'existe pas encore. Écrire
`totaux[c["client"]] + c["prix"]` lèverait une `KeyError`. `.get(cle, 0)`
renvoie 0 dans ce cas, ce qui amorce l'addition proprement — et supprime le
`if client in totaux:` qu'on écrit spontanément.

### 4. `max()` avec une clé de comparaison

```python
max(commandes, key=lambda c: c["prix"])
```

Sans `key=`, Python essaierait de comparer des dictionnaires entre eux et
lèverait `TypeError: '>' not supported between instances of 'dict' and 'dict'`.
`key=` dit sur **quoi** comparer. La commande entière est renvoyée, donc on
récupère l'acheteur avec elle.

### 5. Répondre sans boucle

```python
produits = {c["produit"] for c in commandes if c["client"] == "Alice"}
"souris" in produits        # -> False
```

On construit l'ensemble une fois, puis on l'interroge autant qu'on veut. Le
test `in` sur un `set` est immédiat quelle que soit la taille — c'est la
propriété qu'on exploitera à fond au chapitre 8.

### Bonus : le piège de la copie

```python
[{**c, "prix": round(c["prix"] * 1.10, 2)} for c in commandes]
```

`{**c, "prix": ...}` fabrique un **nouveau** dictionnaire. Sans cela — par
exemple avec `for c in commandes: c["prix"] *= 1.10` — on modifierait les
dictionnaires d'origine, car une liste ne contient pas des copies mais des
**références** vers les mêmes objets. Le test
`test_l_augmentation_ne_touche_pas_la_liste_d_origine` échoue si on l'oublie.

### Réponses chiffrées

| Question | Résultat |
|---|---|
| Clients distincts | 3 — Alice, Bob, Chris |
| Total d'Alice | `225` — soit 45 + 180 |
| Produit le plus cher | `écran` (180 €), acheté par Alice |
| Alice a commandé une souris ? | `False` |
| Prix du clavier après +10 % | `49.5` |

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `fruits[-1]` renvoie le dernier élément : les indices négatifs comptent depuis la fin. |
| 2 | **b** | Un `set` supprime les doublons en silence : `{"a", "b", "a"}` contient deux éléments. |
| 3 | **c** | Les crochets lèvent une `KeyError` si la clé manque, `.get()` renvoie `None` (ou le défaut fourni). |
| 4 | **c** | `enumerate(..., start=1)` produit une paire (indice, valeur), l'indice commençant à 1. |
| 5 | **c** | `.items()` donne les paires clé/valeur ; `.keys()` et `.values()` n'en donnent qu'une moitié. |

## Ce qu'il faut retenir

Le choix de la structure remplace le code. Le dédoublonnage devient un `set`,
le « total par nom » devient un `dict` avec `.get(cle, 0)`, la numérotation
devient `enumerate(..., start=1)`. Chaque fois, la bonne structure fait
disparaître une boucle ou un `if`.
