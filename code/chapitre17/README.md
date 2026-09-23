# Chapitre 17 — Vocabulaire essentiel de l'IA

## L'énoncé

On veut prédire si un étudiant réussira son examen. Le dataset contient, pour
chaque étudiant : heures de révision, notes de l'année, assiduité, et résultat
final (réussite / échec).

## Corrigé de l'exercice

| Question | Réponse |
|---|---|
| 1. Qu'est-ce qu'un **exemple** ? | Un étudiant, c'est-à-dire **une ligne** du tableau. |
| 2. Les **features** ? | heures de révision, notes de l'année, assiduité — les trois colonnes d'entrée. |
| 3. Le **label** ? | le résultat final (réussite / échec). |
| 4. Classification ou régression ? | **Classification** binaire : la sortie est une catégorie, pas un nombre. |
| 5. La réponse du modèle pour un nouvel étudiant ? | une **prédiction**, produite pendant la phase d'**inférence**. |

### Le tableau, avec les mots à leur place

```text
              features (X)                          label (y)
     ┌──────────────┬────────┬────────────┐    ┌──────────────┐
     │ heures révis.│ notes  │ assiduité  │    │  résultat    │
  ┌──┼──────────────┼────────┼────────────┤    ├──────────────┤
  │  │      12      │  11.5  │    0.90    │    │  réussite    │  ← un exemple
  │  │       3      │   8.0  │    0.40    │    │  échec       │  ← un exemple
  └──┴──────────────┴────────┴────────────┘    └──────────────┘
```

En code, cette séparation est **le** geste de départ de tout projet
supervisé — vous l'écrirez au chapitre 33 :

```python
X = df.drop(columns=["resultat"])   # les features
y = df["resultat"]                  # le label
```

### Les trois pièges de vocabulaire

1. **Feature ≠ colonne, toujours.** Le label est une colonne mais n'est pas une
   feature. Confondre les deux, c'est donner la réponse au modèle : il aura
   100 % en entraînement et sera nul en production. C'est une *fuite de
   données* (chapitre 33).
2. **Algorithme ≠ modèle.** L'algorithme (« régression logistique ») est la
   recette ; le modèle est le gâteau, cuit avec **vos** données. Le même
   algorithme sur deux datasets donne deux modèles différents.
3. **Paramètre ≠ hyperparamètre.** Les paramètres sont appris par la machine
   (les poids) ; les hyperparamètres sont choisis par vous *avant*
   l'entraînement (le taux d'apprentissage, la profondeur d'un arbre). C'est
   tout l'objet du chapitre 41.

### Une feature à surveiller ici

« Assiduité » est probablement mesurée *pendant* l'année, donc avant l'examen :
utilisable. Mais si le dataset contenait « nombre d'heures passées à la
session de rattrapage », ce serait une information postérieure au résultat —
inutilisable en prédiction, même si elle fait grimper le score. Cherchez
toujours d'où vient chaque colonne, et surtout **quand** elle est connue.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Une feature est une information d'entrée qui décrit un exemple. |
| 2 | **b** | Le label est la réponse attendue, celle que le modèle apprend à prédire. |
| 3 | **b** | L'algorithme est la méthode d'apprentissage ; le modèle est le produit entraîné sur vos données. |
| 4 | **c** | L'inférence : la phase où un modèle entraîné produit des réponses sur de nouveaux cas. |

## Ce qu'il faut retenir

Exemple, feature, label, modèle, prédiction, inférence : six mots qui suffisent
à décrire n'importe quel projet supervisé. Les employer précisément, c'est déjà
la moitié du travail de conception.
