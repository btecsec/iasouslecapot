# Chapitre 36 — Construire un réseau de neurones simple avec Keras

## L'énoncé

1. Encodez le label en nombres (0, 1, 2) si nécessaire.
2. Construisez un `Sequential` avec deux couches cachées `relu` et une sortie
   `softmax`.
3. Compilez avec l'optimiseur `adam`.
4. Entraînez sur 50 époques avec `validation_split=0.2`.
5. Tracez les courbes d'accuracy. Voyez-vous du surapprentissage ?
6. Évaluez sur le jeu de test.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `net_neurone_keras.py` | Le script du chapitre. |
| `exercice_reseau_keras.py` | Le corrigé, une fonction par question. |
| `test_exercice_reseau_keras.py` | Les tests : architecture, apprentissage, score. |

```bash
pip install -r requirements.txt
python exercice_reseau_keras.py
pytest -q
```

## Corrigé

### 1. Encoder le label — et normaliser X

```python
encodeur = LabelEncoder()
y = encodeur.fit_transform(df["species"])     # Adelie→0, Chinstrap→1, Gentoo→2
```

Deux pièges d'un coup :

- **Gardez l'encodeur.** Le modèle prédira `2`, pas « Gentoo ». Sans
  `encodeur.classes_`, votre API du chapitre 45 ne saura pas retraduire.
- **Normalisez X.** L'énoncé le rappelle entre parenthèses, et ce n'est pas
  facultatif : sans normalisation, la masse (~4 200) écrase le bec (~44) dans
  chaque produit scalaire, et l'apprentissage patine (chapitre 18).

### 2-3. Le réseau

```python
modele = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])
modele.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
```

**243 paramètres au total.** Comptons pour vérifier :

```text
couche 1 :  4 × 16 + 16 =  80
couche 2 : 16 ×  8 +  8 = 136
sortie   :  8 ×  3 +  3 =  27
                    total = 243
```

**Le choix de la perte est la question piège.** Trois cas :

| Vos labels ressemblent à | Perte à utiliser |
|---|---|
| `0, 1, 2` (entiers) | `sparse_categorical_crossentropy` |
| `[1,0,0], [0,1,0]` (one-hot) | `categorical_crossentropy` |
| `0` ou `1` seulement (2 classes) | `binary_crossentropy` |

Se tromper donne une erreur de forme obscure, souvent longue à diagnostiquer.

### 4-5. L'entraînement et les courbes

```python
historique = modele.fit(X_train, y_train, epochs=50, validation_split=0.2)
```

Résultat typique :

```text
ecart entrainement/validation : -0.0048
```

**Voyez-vous du surapprentissage ? Non.** L'écart final est quasi nul — et
même légèrement négatif, la validation faisant marginalement mieux que
l'entraînement (rien d'anormal sur de petits effectifs).

Comment lire deux courbes d'accuracy :

| Ce que vous voyez | Diagnostic |
|---|---|
| Les deux montent ensemble et se rejoignent | tout va bien |
| L'entraînement monte, la validation stagne ou redescend | **surapprentissage** |
| Les deux stagnent bas | sous-apprentissage : réseau trop petit ou trop peu d'époques |
| La courbe zigzague violemment | taux d'apprentissage trop grand (chapitre 38) |

Le chapitre 39 vous fera provoquer volontairement le deuxième cas.

### 6. L'évaluation finale

```text
test : accuracy 1.0000, perte 0.0485
```

Même remarque qu'au chapitre 35 : 100 % sur 69 exemples d'un problème très
séparable n'a rien d'extraordinaire. **Regardez plutôt la perte (0,0485)** :
elle, continue de distinguer un modèle sûr de lui d'un modèle qui a eu raison
de justesse. Deux modèles à 100 % d'accuracy peuvent avoir des pertes très
différentes — et c'est la perte qui prédit lequel tiendra en production.

### Keras face à Scikit-learn : le vrai gain ?

Aucun, ici. La régression logistique du chapitre 35 obtenait déjà le même
score, en une ligne, en une milliseconde. **Le réseau n'apporte rien sur des
données tabulaires bien séparables** — c'est précisément la leçon du
chapitre 23. L'exercice sert à maîtriser l'outil, pas à démontrer sa
supériorité.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **d** | `Sequential` empile des couches dans l'ordre. |
| 2 | **b** | `softmax` transforme les sorties en probabilités dont la somme vaut 1. |
| 3 | **b** | `epochs=50` : le réseau parcourt 50 fois l'ensemble des données. |
| 4 | **b** | Le surapprentissage se voit quand la validation redescend alors que l'entraînement continue de monter. |

## Ce qu'il faut retenir

Trois lignes déclaratives — empiler, compiler, entraîner — et vous avez un
réseau. Le travail sérieux est ailleurs : préparer les données, choisir la
bonne perte, et savoir lire les courbes.
