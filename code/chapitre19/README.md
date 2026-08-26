# Chapitre 19 — Les réseaux de neurones expliqués simplement

Exercice de compréhension : **pas de code** (le code arrive aux chapitres 36
et 37, où vous construirez ce réseau pour de vrai).

## Corrigé de l'exercice

### 1. Dans l'analogie du jury, que représente le poids d'une entrée ?

**La confiance accordée à ce conseiller.** Un poids élevé signifie « quand
celui-là parle, j'écoute » ; un poids proche de zéro, « son avis ne compte
pas » ; un poids **négatif**, « quand il dit oui, je penche pour non ».

Ce dernier cas est celui qu'on oublie : un poids négatif n'est pas une erreur,
c'est une information — la variable joue *contre* la sortie. Vous l'avez déjà
croisé au chapitre 18, avec le `-0.5` de la matrice W.

Et puisque l'apprentissage consiste à ajuster ces poids (chapitre 14), on peut
dire les choses ainsi : **le réseau apprend à qui faire confiance.**

### 2. Pourquoi empiler des couches cachées rend-il un réseau plus puissant ?

Parce que chaque couche travaille sur les **résultats** de la précédente, et
non sur les données brutes. Elle peut donc construire des notions de plus en
plus abstraites :

```text
photo brute → contours → formes → yeux, museau, oreilles → « c'est un chat »
   entrée     couche 1   couche 2       couche 3              sortie
```

Personne n'a programmé « cherche des oreilles » : ces représentations
intermédiaires **émergent** de l'entraînement. C'est la différence
fondamentale avec le machine learning classique, où c'est vous qui fabriquez
les caractéristiques à la main (*feature engineering*, chapitre 17).

Une précision qui compte : cela ne marche **que grâce aux fonctions
d'activation**. Sans elles, empiler dix couches linéaires reviendrait à une
seule couche linéaire — une suite de multiplications de matrices se réduit à
une multiplication de matrices. C'est la non-linéarité (`relu`, sigmoïde) qui
donne sa valeur à la profondeur.

### 3. Remettre les étapes dans l'ordre

```text
1. propagation avant   (forward)     → le réseau produit une prédiction
2. mesurer l'erreur    (perte)       → de combien s'est-il trompé ?
3. rétropropagation    (backward)    → qui est responsable de cette erreur ?
4. ajuster les poids   (optimiseur)  → chacun corrige sa part
```

Puis on recommence, des milliers de fois. Retenez la logique : on ne peut pas
corriger avant d'avoir mesuré, et on ne peut pas mesurer avant d'avoir prédit.
Vous écrirez ces quatre lignes explicitement au chapitre 37 :

```python
sorties = modele(X)              # 1
perte = critere(sorties, y)      # 2
perte.backward()                 # 3
optimiseur.step()                # 4
```

### 4. Quelle architecture pour reconnaître des objets sur des photos ?

**Un réseau convolutif (CNN).** Deux raisons, et elles sont concrètes :

- **Le voisinage compte.** Un pixel n'a de sens qu'avec ses voisins ; la
  convolution regarde de petites fenêtres au lieu de traiter chaque pixel
  isolément.
- **Un chat reste un chat où qu'il soit dans l'image.** Le même filtre est
  appliqué partout, ce qui divise le nombre de paramètres par des milliers face
  à un réseau dense.

Pour mémoire, la correspondance générale : images → CNN ; séquences et texte →
Transformers (chapitres 21 et 22) ; tableaux → souvent pas de réseau du tout,
mais Scikit-learn (chapitre 34).

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Un neurone pondère et additionne ses entrées, puis une fonction d'activation décide de la suite. |
| 2 | **b** | Les couches cachées transforment l'information entre l'entrée et la sortie. « Cachées » signifie simplement qu'on n'observe pas directement leurs valeurs. |
| 3 | **b** | « Profond » = plusieurs couches cachées empilées. |
| 4 | **b** | La rétropropagation renvoie l'erreur en arrière pour attribuer à chaque poids sa part de responsabilité. |

## Ce qu'il faut retenir

Un neurone pondère, additionne, active. Une couche en aligne plusieurs, un
réseau en empile plusieurs couches, et l'entraînement ajuste tous les poids par
l'aller-retour prédire → mesurer → rétropropager → corriger.
