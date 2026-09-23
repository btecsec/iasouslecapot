# Chapitre 18 — Les maths dont l'IA a vraiment besoin

## L'énoncé

Reprenez la couche du chapitre, avec les mêmes `W` et `b`, mais une nouvelle
personne : **1,60 m et 62 kg** (plages inchangées : 1,50–2,00 m et 50–110 kg).

1. Normalisez la taille et le poids : vous devez obtenir un vecteur X.
2. Calculez la norme de X.
3. Calculez les trois scores bruts `Y = X × W + b`, neurone par neurone.
4. Appliquez la sigmoïde à chaque score.
5. Appliquez plutôt une softmax : que remarquez-vous ?
6. Réflexion : si la vraie valeur attendue pour le premier neurone était 1.0,
   quelle serait sa contribution à la MSE ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_maths.py` | Toutes les fonctions du chapitre, écrites à la main puis vérifiées avec NumPy. |
| `test_exercice_maths.py` | Les tests : chaque chiffre du corrigé ci-dessous y est vérifié. |
| `requirements.txt` | numpy, pytest. |

```bash
python exercice_maths.py     # refait tout l'exercice, étape par étape
pytest -q
```

## Corrigé, chiffre par chiffre

### 1. Normalisation min-max

```text
taille : (1.60 - 1.50) / (2.00 - 1.50) = 0.10 / 0.50 = 0.2
poids  : (62   - 50)   / (110  - 50)   = 12   / 60   = 0.2

X = [ 0.2  0.2 ]
```

Le résultat est instructif : **cette personne occupe la même position relative
sur les deux échelles** (20 % de la plage), alors que les valeurs brutes (1,60
et 62) n'ont rien de comparable. C'est tout l'intérêt de la normalisation.

### 2. Norme du vecteur

```text
||X|| = √(0.2² + 0.2²) = √0.08 ≈ 0.2828
```

À comparer à la personne du chapitre (`[0.5, 0.8]`, norme ≈ 0.943) : notre
nouvelle entrée est un vecteur **court**, proche de l'origine. Elle activera
donc faiblement tous les neurones — ce qu'on va vérifier tout de suite.

### 3. Les trois scores bruts

Rappel des poids :

```text
       [  0.1   0.4  -0.2 ]
W  =   [  0.3  -0.5   0.6 ]        b = [ 0.1   0.0   0.2 ]
```

```text
Neurone 1 : 0.2×0.1  + 0.2×0.3    + 0.1 = 0.02 + 0.06 + 0.1 =  0.18
Neurone 2 : 0.2×0.4  + 0.2×(-0.5) + 0.0 = 0.08 - 0.10 + 0.0 = -0.02
Neurone 3 : 0.2×(-0.2)+ 0.2×0.6   + 0.2 = -0.04 + 0.12 + 0.2 = 0.28

Y = [ 0.18  -0.02   0.28 ]
```

Remarquez que le biais fait ici presque tout le travail : avec une entrée aussi
faible, les scores sont proches de `b`. Le biais est bien ce qui permet à un
neurone de s'activer **même quand les entrées sont nulles**.

### 4. Sigmoïde de chaque score

```text
σ( 0.18) ≈ 0.5449
σ(-0.02) ≈ 0.4950
σ( 0.28) ≈ 0.5695
```

Trois valeurs collées à 0,5 : le réseau est **indécis** sur cette personne, ce
qui est cohérent avec un vecteur d'entrée proche de zéro. Et surtout, la somme
vaut 1,609 : **les sorties sigmoïdes ne forment pas une distribution de
probabilités.** Chaque neurone répond dans son coin, à la question « moi, je
m'active ou pas ? ».

### 5. Softmax des mêmes scores bruts

```text
softmax([0.18, -0.02, 0.28]) ≈ [ 0.3420  0.2800  0.3780 ]
                        somme  =   1.0000
```

**Ce qu'on remarque** — et c'est le cœur de la question :

| | Sigmoïde | Softmax |
|---|---|---|
| Somme des sorties | 1,609 (quelconque) | **1,000 (toujours)** |
| Les neurones sont… | indépendants | **en concurrence** |
| Question posée | « ce neurone s'active-t-il ? » | « lequel des trois gagne ? » |
| Cas d'usage | étiquettes multiples (une photo peut être « plage » *et* « coucher de soleil ») | classes exclusives (un manchot est d'*une* espèce) |

L'ordre est le même dans les deux cas (neurone 3 > neurone 1 > neurone 2) : la
softmax ne change pas le classement, elle change **l'interprétation**.

### 6. Contribution à la MSE

Avec la sortie sigmoïde du premier neurone (0,5449) et une vraie valeur de 1.0 :

```text
écart          = 1.0 - 0.5449 = 0.4551
écart au carré = 0.4551²      ≈ 0.2071      ← sa contribution à la somme
divisé par 3 neurones          ≈ 0.0690     ← sa contribution à la moyenne
```

Le carré n'est pas une coquetterie : il empêche les erreurs positives et
négatives de s'annuler, et il **punit davantage les grosses erreurs**. Une
erreur de 0,9 pèse 0,81 ; deux erreurs de 0,45 pèsent 0,2 chacune. La MSE
préfère donc deux erreurs moyennes à une énorme — un choix, pas une loi de la
nature.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le produit scalaire mesure à quel point deux vecteurs pointent dans la même direction. |
| 2 | **b** | Pour que les variables aux grandes valeurs (le poids en kilos) n'écrasent pas les autres (la taille en mètres). |
| 3 | **b** | La dérivée concerne une fonction à une variable ; le gradient rassemble les dérivées partielles d'une fonction à plusieurs variables. |
| 4 | **b** | La softmax produit une liste de probabilités dont la somme fait 1. |
| 5 | **b** | Le carré empêche les erreurs de signes opposés de s'annuler et pénalise plus fort les grosses erreurs. |
| 6 | **c** | La cross-entropy vaut `-log(p)` sur la bonne catégorie, soit `-log(0.10) ≈ 2,30`. |

## Ce qu'il faut retenir

Une couche de réseau, c'est trois gestes : normaliser l'entrée, faire un
produit matriciel, appliquer une activation. Vous venez de les faire à la main.
Tout le reste du livre les délègue à une bibliothèque — mais vous saurez
désormais ce que la bibliothèque calcule.
