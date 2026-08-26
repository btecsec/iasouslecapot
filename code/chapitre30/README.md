# Chapitre 30 — Le cycle de vie complet d'un projet IA

Exercice de cadrage : **pas de code**. Vous choisissez un projet imaginaire et
rédigez une phrase par étape du cycle.

## Corrigé sur un exemple : prédire le retard d'un train

| # | Étape | Réponse pour ce projet |
|---|---|---|
| 1 | **Problème et tâche** | Prédire, au départ, le retard en minutes à l'arrivée. Sortie numérique continue → **régression** (ch. 19). Pas « détecter les trains en retard » : ce serait une classification, un autre projet. |
| 2 | **Données à collecter** | Historique des circulations (départ, arrivée théorique/réelle, ligne, matériel), météo horaire, jours fériés et vacances, travaux planifiés, affluence en gare. |
| 3 | **Nettoyage** | Retards négatifs (trains en avance) : garder ou ramener à 0 ? Trains supprimés : ce ne sont pas des retards de 0, il faut les **exclure**. Météo manquante sur certaines heures → imputation. Noms de gares écrits de trois façons → harmonisation. |
| 4 | **Découpage** | **Chronologique**, surtout pas aléatoire : entraînement sur janvier-septembre, test sur octobre-décembre. Un découpage au hasard laisserait le modèle voir l'après pour prédire l'avant — une fuite de données garantie (ch. 29). |
| 5 | **Baseline simple** | « Le retard moyen historique de cette ligne, à cette heure ». Une ligne de code, aucun apprentissage, et un score à battre. Si votre réseau de neurones ne fait pas mieux, il ne sert à rien. |
| 6 | **Évaluation** | MAE en minutes (interprétable : « on se trompe de 4 minutes en moyenne »), à comparer à la MAE de la baseline. Le R² complète, la RMSE si les gros retards comptent double. |
| 7 | **Amélioration** | Ajouter des features (retard du train précédent sur la même voie), régler les hyperparamètres par validation croisée (ch. 37), essayer un gradient boosting avant tout réseau. |
| 8 | **Surveillance** | Suivre la distribution des entrées (nouvel horaire = dérive des données) et l'erreur réelle une fois les vrais retards connus (ch. 44). Réentraîner à chaque changement de service. |

### Les trois pièges que ce cadrage évite

1. **Se tromper de tâche (étape 1).** « Prédire les retards » est ambigu.
   Régression et classification n'ont ni le même modèle, ni la même perte, ni
   la même métrique. Une phrase floue à l'étape 1 se paie à l'étape 6.
2. **Découper au hasard une série temporelle (étape 4).** L'erreur la plus
   fréquente des débutants, et la plus difficile à détecter : le score est
   *excellent* en test et catastrophique en production.
3. **Sauter la baseline (étape 5).** Sans point de comparaison, un score de
   « MAE 4 minutes » ne veut rien dire. Peut-être que « toujours prédire 3
   minutes » fait aussi bien.

### Pourquoi un cycle et non une ligne droite

Le vrai déroulement ressemble plutôt à ceci :

```text
1 → 2 → 3 → 4 → 5 → 6 → « c'est mauvais » → retour à 2 (données insuffisantes)
                     ↘ « c'est bon » → 7 → 8 → dérive détectée → retour à 2
```

On revient toujours en arrière. Et l'étape où l'on revient le plus souvent est
la **collecte de données** : c'est là que se gagnent les points, bien plus que
dans le choix du modèle.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **c** | Environ 80 % du temps part dans la préparation des données. Le chiffre choque, il est pourtant constant d'un projet à l'autre. |
| 2 | **b** | Parce qu'on revient souvent en arrière pour itérer. |
| 3 | **b** | Une baseline est un modèle simple servant de point de comparaison. |
| 4 | **c** | Commencer simple, complexifier seulement si nécessaire — et seulement avec une mesure à l'appui. |

## Ce qu'il faut retenir

Huit étapes, une boucle, et une règle : commencer simple. Le cycle ne dépend ni
du dataset, ni du framework — vous le rejouerez à l'identique au chapitre 43
sur les manchots, puis en production dans toute la partie V.
