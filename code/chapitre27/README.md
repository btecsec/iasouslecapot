# Chapitre 27 — PyTorch, les bases

## L'énoncé

1. Créez un tenseur de 24 valeurs allant de 0 à 23, puis affichez sa forme.
2. Réorganisez-le en `(2, 3, 4)`, puis en `(6, 4)`, en vérifiant la forme à
   chaque fois.
3. Extrayez la première colonne de la version `(6, 4)`, puis sa deuxième rangée.
4. Créez deux tenseurs aléatoires de forme `(4, 5)`. Devinez, **avant** de
   l'exécuter, la forme donnée par `torch.stack`, par `torch.cat(..., dim=0)` et
   par `torch.cat(..., dim=1)`. Vérifiez ensuite.
5. Créez `x = torch.tensor(2.0, requires_grad=True)`, calculez `y = x ** 3`,
   appelez `y.backward()` et vérifiez que `x.grad` vaut bien 12.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_bases_torch.py` | Le corrigé exécutable, une fonction par question. |
| `test_exercice_bases_torch.py` | Les tests unitaires (ignorés proprement si PyTorch n'est pas installé). |
| `requirements.txt` | torch, pytest. |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest

python exercice_bases_torch.py
pytest -q
```

## Corrigé commenté

### 1 et 2. Créer, puis changer de forme

```python
t = torch.arange(24)          # forme (24,)

t.reshape(2, 3, 4)            # (2, 3, 4) : 2 × 3 × 4 = 24
t.reshape(6, -1)              # (6, 4)    : le -1 se déduit tout seul
```

La seule règle à respecter est que le **produit des dimensions ne change pas**.
`t.reshape(5, 3)` réclamerait 15 valeurs sur 24 : PyTorch lève une erreur plutôt
que d'en inventer ou d'en perdre. Un test du dossier le vérifie explicitement.

Le `-1` n'est pas une taille : c'est une inconnue que PyTorch résout à votre
place. Pratique dès que l'une des dimensions dépend de la taille du lot.

### 3. Découper

```python
tableau = t.reshape(6, 4)

tableau[:, 0]   # [0, 4, 8, 12, 16, 20] : toutes les rangées, colonne 0
tableau[1]      # [4, 5, 6, 7]          : la deuxième rangée (l'indice part de 0)
```

**Le piège** : `tableau[1]` est une **vue**, pas une copie. Les deux tenseurs
partagent la même mémoire, donc écrire dans la vue écrit dans l'original :

```python
vue = tableau[1]
vue[0] = -1
print(tableau[1, 0])   # -1 : l'original a bougé
```

C'est voulu — dupliquer des tableaux énormes coûterait cher — mais cela
surprend au moins une fois. `tableau[1].clone()` donne une copie indépendante.

### 4. Empiler ou coller

| Opération | Forme obtenue | Pourquoi |
|---|---|---|
| `torch.stack([a, b])` | `(2, 4, 5)` | une dimension **nouvelle** en tête : deux « feuilles » de (4, 5) |
| `torch.cat([a, b], dim=0)` | `(8, 5)` | on allonge les rangées : 4 + 4 |
| `torch.cat([a, b], dim=1)` | `(4, 10)` | on allonge les colonnes : 5 + 5 |

`stack` exige des formes **strictement identiques** ; `cat` tolère une
différence sur l'axe que l'on colle, mais exige l'égalité sur tous les autres.
Empiler un `(4, 5)` avec un `(4, 6)` lève donc une erreur — c'est le dernier
test du dossier.

### 5. La dérivée, retrouvée toute seule

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)          # tensor(12.) : 3x² en x = 2
```

`requires_grad=True` demande à PyTorch d'enregistrer les opérations subies par
`x`. `backward()` les remonte en sens inverse et dépose le résultat dans
`x.grad`. Vous n'avez écrit nulle part que la dérivée de x³ est 3x² : c'est tout
l'intérêt.

## Le geste qui sert vraiment plus tard

```python
x = torch.stack([donnees[i:i + taille_contexte] for i in debuts])
```

La compréhension produit autant de tenseurs de forme `(taille_contexte,)` qu'il
y a de positions tirées ; `torch.stack` les empile en un lot de forme
`(taille_lot, taille_contexte)`. C'est ainsi qu'un modèle de langage fabrique
ses exemples d'entraînement — la fonction `lot_dexemples` du corrigé est cette
ligne, isolée et testée.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | 12 valeurs en 3 rangées : le `-1` vaut donc 4, soit `torch.Size([3, 4])`. |
| 2 | **b** | `*` multiplie terme à terme, `@` fait le produit matriciel — deux opérations différentes, pas deux écritures. |
| 3 | **c** | `stack` ajoute une dimension en tête : 8 tenseurs de `(16,)` donnent `(8, 16)`. `(128,)` serait le résultat de `cat`. |
| 4 | **b** | « Long » désigne les entiers 64 bits, « Float » les décimaux : il manque un `.float()`. |
| 5 | **c** | `requires_grad=True` fait suivre les opérations pour que `backward()` puisse calculer les dérivées. |

## Ce qu'il faut retenir

Presque tous vos bugs seront des erreurs de forme. Le réflexe le plus rentable
du métier tient en une ligne : `print(x.shape)` avant et après chaque étape
suspecte.
