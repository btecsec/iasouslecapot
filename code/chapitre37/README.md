# Chapitre 37 — Construire un réseau de neurones simple avec PyTorch

## L'énoncé

1. Convertissez vos données en tenseurs.
2. Définissez une classe réseau avec deux couches cachées ReLU.
3. Choisissez `CrossEntropyLoss` et l'optimiseur Adam.
4. Écrivez la boucle d'entraînement sur 50 époques, affichez la perte.
5. Évaluez sur le test avec `torch.no_grad()`.
6. Comparez : retrouvez-vous chaque étape de la version Keras ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `net_neuronne_pytorch.py` | Le script du chapitre. |
| `exercice_reseau_pytorch.py` | Le corrigé, une fonction par question. |
| `test_exercice_reseau_pytorch.py` | Les tests, dont la démonstration de `zero_grad`. |

```bash
pip install -r requirements.txt
python exercice_reseau_pytorch.py
pytest -q
```

## Corrigé

### 1. Les tenseurs — et le piège du type

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)      # ← long, pas float !
```

`CrossEntropyLoss` attend des **indices de classe** entiers, pas des nombres
réels. Avec des labels en float, l'erreur est immédiate mais peu explicite :
`RuntimeError: expected scalar type Long but found Float`. Souvenez-vous-en,
vous économiserez une demi-heure.

### 2. Le réseau

```python
class ReseauManchots(nn.Module):
    def __init__(self):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 3),          # pas de softmax ici
        )

    def forward(self, x):
        return self.couches(x)
```

**243 paramètres** : exactement les mêmes qu'en Keras au chapitre 36. Même
architecture, même compte — seule l'écriture change.

**Pourquoi pas de softmax en sortie ?** Parce que `CrossEntropyLoss` l'applique
elle-même, dans une version numériquement stable (`log_softmax` fusionné).
L'ajouter reviendrait à l'appliquer deux fois : l'entraînement fonctionnerait
plus mal, sans le moindre message d'erreur. C'est **la** différence de
convention entre Keras et PyTorch, et la source d'erreur n° 1 quand on passe
de l'un à l'autre.

### 3-4. La boucle, quatre lignes qui ne changent jamais

```python
for epoch in range(50):
    optimiseur.zero_grad()          # 1. effacer les gradients du tour précédent
    sorties = modele(X_train)       # 2. prédire
    perte = critere(sorties, y_train)
    perte.backward()                # 3. calculer les gradients
    optimiseur.step()               # 4. corriger les poids
```

```text
perte : 1.0648 -> 0.0322
```

La perte initiale (≈ 1,06) n'est pas quelconque : un modèle non entraîné
répartit sa confiance uniformément sur 3 classes, ce qui donne
`-log(1/3) ≈ 1,0986`. Si votre perte de départ vaut 7 ou 0,01, quelque chose
cloche dans les données ou la sortie. Le test
`test_la_perte_initiale_est_proche_de_log_3` vérifie ce repère — le même qui
donne `log(2000) ≈ 7,6` sur un modèle de langage à 2000 jetons de vocabulaire.

**Pourquoi `zero_grad()` ?** Parce que PyTorch **accumule** les gradients au
lieu de les remplacer. Sans cet appel, chaque `backward()` s'ajoute au
précédent : au dixième tour, vous corrigez les poids avec la somme de dix
gradients, et l'entraînement diverge. Le test
`test_sans_zero_grad_les_gradients_saccumulent` le prouve : après deux
`backward()` sans remise à zéro, le gradient vaut exactement le double.

### 5. L'évaluation

```python
modele.eval()
with torch.no_grad():
    predictions = modele(X_test).argmax(dim=1)
```

Deux gestes distincts, souvent confondus :

| Appel | Ce qu'il fait |
|---|---|
| `modele.eval()` | change le **comportement** des couches Dropout et BatchNorm |
| `torch.no_grad()` | cesse de **construire le graphe** des gradients |

Ici le réseau n'a ni dropout ni batchnorm : `eval()` ne change rien, mais on
l'écrit quand même, par habitude — le jour où vous ajouterez un `Dropout`
(chapitre 39), l'oublier fausserait toutes vos prédictions.

Résultat : **accuracy 1.0000**, identique à Keras. Rien d'étonnant : même
architecture, même données, même optimiseur.

### 6. La correspondance Keras ↔ PyTorch

| Keras | PyTorch | Qui écrit quoi |
|---|---|---|
| `Sequential([...])` | `nn.Sequential(...)` dans une classe | vous, des deux côtés |
| `compile(optimizer, loss)` | `nn.CrossEntropyLoss()` + `torch.optim.Adam()` | vous |
| `fit(X, y, epochs=50)` | **la boucle `for`** | Keras / **vous** |
| `evaluate(X, y)` | `no_grad()` + comparaison | Keras / **vous** |
| softmax en sortie | **inclus dans la perte** | attention au piège |

Toute la différence est là : Keras vous donne `fit()`, PyTorch vous donne les
quatre lignes. Plus verbeux, mais rien n'est caché — et le jour où vous voulez
faire quelque chose d'inhabituel au milieu de la boucle (accumuler les
gradients sur plusieurs lots, par exemple), vous êtes déjà au bon endroit.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Un tenseur est un tableau de nombres, la base de tous les calculs. |
| 2 | **b** | `perte.backward()` calcule les gradients par rétropropagation. |
| 3 | **a** | `zero_grad()` remet les gradients à zéro pour éviter qu'ils s'accumulent. |
| 4 | **b** | En PyTorch, on écrit soi-même la boucle d'entraînement. |

## Ce qu'il faut retenir

`zero_grad`, `forward`, `backward`, `step` : quatre lignes, toujours dans cet
ordre. Vous les réécrirez à l'identique pour entraîner un modèle mille fois
plus gros : la boucle, elle, ne change pas.
