# Chapitre 35 — Entraîner son premier modèle avec Scikit-learn

## L'énoncé

1. Chargez, nettoyez et encodez les données.
2. Découpez en 80/20 stratifié.
3. Entraînez une régression logistique, affichez le score de test.
4. Affichez le `classification_report` et repérez la classe la moins bien
   prédite.
5. Reconstruisez le tout avec un `make_pipeline` incluant un `StandardScaler`.
   Le score change-t-il ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `train_model.py` | Le script du chapitre. |
| `exercice_scikit.py` | Le corrigé : version manuelle **et** version pipeline. |
| `test_exercice_scikit.py` | Les tests, dont la comparaison des deux versions. |

```bash
pip install -r requirements.txt
python exercice_scikit.py
pytest -q
```

## Corrigé

### 1-3. Le résultat

```text
273 exemples d'entrainement, 69 de test
9 colonnes apres encodage
score : 1.0000
```

### Un score de 100 % : faut-il se réjouir ?

**Non — il faut enquêter.** Un score parfait est un signal d'alarme avant
d'être une réussite. Les trois causes possibles, dans l'ordre où il faut les
vérifier :

1. **Une fuite de données.** La cible est-elle passée dans X ? Ici non : le
   test `test_la_cible_a_ete_retiree_des_features` le vérifie.
2. **Un jeu de test trop petit.** 69 exemples : passer de 100 % à 98,5 %, c'est
   **un seul manchot**. Le score n'a pas la précision que ses quatre décimales
   suggèrent.
3. **Un problème réellement facile.** C'est le cas ici, et c'est la bonne
   réponse. Les trois espèces se distinguent nettement : les Gentoo pèsent
   1 300 g de plus que les autres, et l'île apporte une information massive
   (les Gentoo ne vivent que sur Biscoe). Avec de telles séparations, une
   frontière linéaire suffit.

À retenir : sur un problème réel — un désabonnement client, un diagnostic —
vous ne verrez jamais 100 %. Si cela arrive, cherchez la fuite avant de fêter.

### 4. Le `classification_report`

```text
              precision    recall  f1-score   support

      Adelie       1.00      1.00      1.00        30
   Chinstrap       1.00      1.00      1.00        14
      Gentoo       1.00      1.00      1.00        25

    accuracy                           1.00        69
```

Toutes les classes sont parfaites, donc pas de « moins bien prédite » ici. Ce
qui compte, c'est **de savoir la lire** — car dès que le score descend, c'est
cette table qui dit *où* :

| Colonne | Question à laquelle elle répond |
|---|---|
| `precision` | quand le modèle dit « Chinstrap », a-t-il raison ? |
| `recall` | parmi les vrais Chinstrap, combien en a-t-il trouvés ? |
| `f1-score` | la moyenne harmonique des deux — le résumé |
| `support` | combien d'exemples de cette classe dans le test |

Regardez toujours le `support` : ici, 14 Chinstrap seulement. Un rappel de
0,93 sur cette classe signifierait *un* manchot raté. Ne surinterprétez pas
des métriques calculées sur quatorze exemples.

**Le piège du dictionnaire.** `classification_report(..., output_dict=True)`
contient aussi les entrées `macro avg` et `weighted avg`. Chercher le F1
minimum sans les exclure renverrait parfois « macro avg » comme nom de classe.
Le corrigé les filtre, et un test le vérifie.

### 5. Avec un pipeline : le score change-t-il ?

**Non, exactement le même score.** Et c'est tout l'intérêt de la question :
le pipeline n'améliore pas les performances, il supprime une **classe
d'erreurs**.

```python
modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modele.fit(X_train, y_train)      # le scaler s'ajuste sur le train seul
modele.score(X_test, y_test)      # et s'applique automatiquement au test
```

Trois erreurs deviennent impossibles :

- appeler `fit_transform` sur le test (fuite de données) ;
- oublier de normaliser au moment de prédire (prédictions fausses, sans
  message d'erreur) ;
- perdre le scaler en sauvegardant seulement le modèle (chapitre 42).

Le pipeline se sauvegarde d'un bloc et s'utilise sur des données **brutes** —
c'est exactement ce que votre API attendra au chapitre 45.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `fit`, `predict`, `score` : la même interface pour tous les modèles de Scikit-learn. |
| 2 | **b** | Sur le jeu de test, jamais vu à l'entraînement. |
| 3 | **b** | L'accuracy est la proportion de prédictions correctes. |
| 4 | **b** | Le pipeline enchaîne les étapes proprement et évite la fuite de données. |

## Ce qu'il faut retenir

`fit`, `predict`, `score` : trois verbes, et vous savez entraîner n'importe
quel modèle de Scikit-learn. Prenez tout de suite l'habitude du pipeline : il
ne coûte rien et rend inatteignables les erreurs les plus coûteuses.
