# Chapitre 34 — Choisir un premier modèle simple

## L'énoncé

Sur vos données de manchots découpées au chapitre 33 (classer l'espèce) :

1. Entraînez une régression logistique, notez son score de test.
2. Entraînez un arbre de décision avec `max_depth=3` et comparez.
3. Entraînez une forêt aléatoire de 100 arbres et comparez.
4. Lequel est le meilleur ? Le plus simple suffit-il ?
5. Pourquoi est-il utile d'avoir une baseline avant de tester la forêt ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `select_model.py` / `prepare_data.py` | Les scripts du chapitre. |
| `exercice_modeles.py` | Le corrigé : les trois modèles **plus** une baseline. |
| `test_exercice_modeles.py` | Les tests, dont la vérification que chaque modèle écrase la baseline. |

```bash
pip install -r requirements.txt
python exercice_modeles.py
pytest -q
```

## Corrigé

### 1-3. Les scores

```text
baseline (classe majoritaire)      0.4348
regression logistique              1.0000
arbre de decision (max_depth=3)    0.9855
foret aleatoire (100 arbres)       1.0000
```

*(Sur 69 exemples de test, avec `random_state=42`. Vos chiffres peuvent varier
d'un point ou deux selon la graine — c'est normal, et instructif : un écart de
1 point représente **un seul manchot**.)*

### 4. Lequel est le meilleur ? Le plus simple suffit-il ?

La régression logistique et la forêt sont à égalité parfaite ; l'arbre bridé à
3 niveaux se trompe sur un seul manchot. Autrement dit : **oui, le plus simple
suffit largement.**

Ce n'est pas un accident. Regardez ce que fait chaque modèle :

| Modèle | Sa façon de raisonner | Paramètres | Explicable ? |
|---|---|---|---|
| Régression logistique | trace des frontières droites entre les classes | 15 | oui, un coefficient par mesure |
| Arbre (profondeur 3) | pose 3 questions oui/non | quelques dizaines | oui, on peut le dessiner |
| Forêt (100 arbres) | fait voter 100 arbres différents | des milliers | difficilement |

Les espèces de manchots se séparent presque par des droites (les Gentoo sont
plus lourds, les Adelie ont un bec plus court). Une frontière droite suffit
donc — et la forêt, plus puissante, n'a rien de plus à apporter. Sa complexité
serait payée sans contrepartie : modèle plus lourd, plus lent, moins
explicable, plus difficile à défendre devant un client.

**La règle professionnelle** : le modèle le plus simple qui atteint l'objectif
gagne. On ne complexifie que quand la mesure le justifie.

### 5. À quoi sert la baseline ?

À donner une échelle. Sans elle, « 98,5 % » ne veut rien dire.

Ici, un modèle qui répond **toujours « Adelie »**, sans rien regarder, obtient
déjà **43,5 %**. C'est le vrai plancher. Un modèle à 60 % ne serait pas « pas
mal », il serait mauvais.

Le cas extrême rend l'argument évident : sur une détection de fraude où 0,1 %
des transactions sont frauduleuses, répondre toujours « légitime » donne
**99,9 % d'accuracy** — et zéro fraude détectée. La baseline est ce qui
empêche de fêter ce genre de score (voir chapitre 40).

En pratique, `DummyClassifier` de Scikit-learn fait ce travail en une ligne :

```python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline.score(X_test, y_test)      # 0.4348
```

### Un détail qui compte : le scaler pour la régression seulement

Dans le corrigé, seule la régression logistique est enveloppée dans un
`make_pipeline(StandardScaler(), ...)`. Les arbres, eux, n'ont **pas besoin de
normalisation** : ils comparent des seuils colonne par colonne, l'échelle leur
est indifférente. Les modèles à base de distances ou de gradients (régression,
SVM, réseaux de neurones) y sont au contraire très sensibles.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | La régression linéaire prédit une valeur numérique continue. |
| 2 | **b** | Malgré son nom, la régression logistique **classe** : elle sort une probabilité d'appartenance. |
| 3 | **b** | Un arbre raisonne par cascade de questions oui/non. |
| 4 | **b** | Une forêt aléatoire est un ensemble d'arbres qui votent. |

## Ce qu'il faut retenir

Commencez toujours par une baseline, puis par le modèle le plus simple. Sur des
données bien séparables, la régression logistique fait aussi bien qu'une forêt
— pour un dixième du poids et une explicabilité entière.
