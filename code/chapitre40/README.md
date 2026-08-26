# Chapitre 40 — Valider et évaluer son modèle

## L'énoncé

1. Affichez la matrice de confusion, identifiez faux positifs et faux négatifs.
2. Calculez précision et rappel via `classification_report`.
3. Si rater un positif coûte très cher : précision ou rappel ?
4. Sur un problème de régression (prédire une masse), calculez MAE et R².
5. Expliquez pourquoi l'accuracy seule serait trompeuse sur une classe rare.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `evaluation_model.py` | Le script du chapitre. |
| `exercice_metriques.py` | Le corrigé : détection d'une espèce rare + régression sur la masse. |
| `test_exercice_metriques.py` | Les tests, dont les métriques recalculées à la main. |

```bash
pip install -r requirements.txt
python exercice_metriques.py
pytest -q
```

## Le problème choisi

Détecter les **Chinstrap** : 68 manchots sur 344, soit 20 %. Assez rare pour
que l'accuracy commence à mentir, assez fréquent pour rester lisible.

## Corrigé

### 1. La matrice de confusion

```text
20 positifs sur 103 exemples de test

vrais_negatifs   83
faux_positifs     0        ← fausses alertes
faux_negatifs     5        ← Chinstrap manqués
vrais_positifs   15
```

Quatre cases, deux erreurs — et ces deux erreurs n'ont **jamais** le même coût :

| | Le modèle dit « non » | Le modèle dit « oui » |
|---|---|---|
| **Vraiment non** | vrai négatif (83) | **faux positif** (0) — on dérange pour rien |
| **Vraiment oui** | **faux négatif** (5) — on est passé à côté | vrai positif (15) |

### 2. Précision et rappel

```text
precision  1.0000
rappel     0.7500
f1         0.8571
```

Les deux formules, et surtout les deux **questions** :

| Métrique | Formule | La question posée |
|---|---|---|
| Précision | VP / (VP + FP) | « Quand il crie au loup, a-t-il raison ? » |
| Rappel | VP / (VP + FN) | « Combien de loups a-t-il vus passer ? » |
| F1 | moyenne harmonique | le compromis, quand aucune des deux ne domine |

Ici : **précision parfaite** (les 15 alertes sont toutes justes) mais **rappel
de 75 %** (5 Chinstrap sur 20 sont passés inaperçus). Le modèle est prudent :
il ne se mouille que lorsqu'il est sûr.

Notez que le F1 est une moyenne **harmonique**, pas arithmétique : elle est
tirée vers le bas par la plus faible des deux. Précision 1,0 et rappel 0,0
donneraient un F1 de 0, pas de 0,5 — c'est fait exprès.

### 3. Rater un positif coûte cher : précision ou rappel ?

**Le rappel.** Rater un positif, c'est un faux négatif, et le rappel est
précisément la métrique qui les compte.

Et voici le levier concret, qui ne coûte **aucun réentraînement** — il suffit
de déplacer le seuil de décision :

```text
seuil 0.5  -> precision 1.000 | rappel 0.750
seuil 0.3  -> precision 0.950 | rappel 0.950
seuil 0.1  -> precision 0.800 | rappel 1.000
```

À 0,1, on retrouve **tous** les Chinstrap, au prix de quelques fausses
alertes. C'est exactement l'arbitrage d'un dépistage médical : on préfère
convoquer dix personnes en bonne santé pour un examen complémentaire plutôt
que de laisser passer un malade.

```python
probabilites = modele.predict_proba(X)[:, 1]
predictions = (probabilites >= 0.1).astype(int)     # au lieu de .predict()
```

**Le réflexe à prendre** : `predict()` applique un seuil de 0,5 qui n'a rien
d'universel. C'est une valeur par défaut, pas une décision métier. La décision
métier, c'est vous qui la prenez, en connaissant le coût de chaque erreur.

| Contexte | Erreur la plus coûteuse | On privilégie | Seuil |
|---|---|---|---|
| Dépistage médical | rater un malade | **rappel** | bas |
| Anti-spam | jeter un vrai message | **précision** | haut |
| Fraude bancaire | dépend du montant | F1, ou un coût pondéré | à calibrer |

### 4. Régression : prédire la masse

```text
MAE  266.9 g
RMSE 330.9 g
R2   0.8347
```

Les trois se complètent, et chacune répond à une question différente :

- **MAE (266,9 g)** — l'erreur moyenne, **dans l'unité d'origine**. C'est la
  seule que vous pouvez annoncer à un non-spécialiste : « on se trompe de
  267 grammes en moyenne ».
- **RMSE (330,9 g)** — la même chose, mais les grosses erreurs pèsent plus
  lourd (elles sont mises au carré). RMSE est **toujours ≥ MAE** ; plus l'écart
  entre les deux est grand, plus vos erreurs sont irrégulières.
- **R² (0,83)** — la part de variance expliquée, sans unité. Le repère est
  **0 = aussi bon que prédire toujours la moyenne**, 1 = parfait. Un R² négatif
  est possible : il signifie que votre modèle fait pire que la moyenne.

### 5. Pourquoi l'accuracy est trompeuse sur une classe rare

```text
un modele qui repond toujours « non » obtient une accuracy de
0.8058 — sans detecter un seul Chinstrap.
```

Voilà. **80,6 % d'accuracy pour un modèle qui n'a rien appris**, et dont le
rappel est de **zéro**.

Et 20 % de positifs, c'est encore confortable. Sur un cas de fraude à 0,1 %,
le même modèle nul afficherait **99,9 %**. C'est ainsi que des projets se font
valider sur un chiffre flatteur avant de ne rien détecter en production.

**La règle** : sur des classes déséquilibrées, on ne regarde jamais l'accuracy
seule. On regarde la matrice de confusion, puis précision et rappel **par
classe**.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Sur des données déséquilibrées, un modèle inutile peut afficher un score élevé. |
| 2 | **a** | Le rappel est la proportion de vrais positifs retrouvés parmi tous les positifs réels. |
| 3 | **b** | On privilégie la précision quand une fausse alerte coûte plus cher qu'un oubli (anti-spam). |
| 4 | **b** | En régression, on utilise la MAE ou le R² — la matrice de confusion et le F1 n'ont aucun sens sur des valeurs continues. |

## Ce qu'il faut retenir

Une métrique n'est pas une note, c'est une **question**. Choisissez-la en
fonction de ce que coûte chaque erreur, et rappelez-vous que le seuil de 0,5
est un réglage par défaut — pas une vérité.
