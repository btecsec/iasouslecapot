# Chapitre 2 — Les prérequis nécessaires

Exercice de « traduction » : **pas de code**. On relie une situation courante à
la notion mathématique qui la décrit.

## Corrigé de l'exercice

| Situation | Notion | Pourquoi |
|---|---|---|
| « 30 % de risque de neige » | **Probabilité** | Un nombre entre 0 et 1 qui mesure une confiance, exactement ce que sort un classifieur (chapitre 16) avant qu'on tranche. |
| Décrire une voiture par puissance, poids, prix, année | **Vecteur** | Une liste ordonnée de nombres : `[110, 1250, 18500, 2019]`. C'est *littéralement* la ligne d'un dataset (chapitre 17). |
| Chercher le chemin le plus rapide pour descendre une piste dans le brouillard | **Dérivée / descente de gradient** | Vous ne voyez pas le bas de la piste ; vous sentez seulement la pente sous vos skis et vous suivez la plus forte descente. C'est l'algorithme d'apprentissage au complet (chapitre 14). |
| Une machine transforme « un thé » en une tasse de thé | **Fonction** | Une entrée → une sortie, de façon reproductible. Un modèle entraîné n'est rien d'autre qu'une (grosse) fonction. |

### Le détail qui compte

Ces quatre notions reviennent dans tout le livre :

- la **probabilité** ressort du `softmax` (chapitres 18 et 22) ;
- le **vecteur** devient l'*embedding* d'un mot (chapitre 21) ;
- la **dérivée** devient le gradient que `backward()` calcule (chapitre 37) ;
- la **fonction** devient votre modèle, sauvegardé puis servi par une API
  (chapitres 42 et 45).

Vous n'aurez jamais à les calculer à la main — sauf aux chapitres 18 et 22, où
c'est justement l'exercice, et où le corrigé est fourni en code testé.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | L'ingénieur IA *interprète* les calculs que la bibliothèque exécute. Il ne les refait pas à la main, mais il doit savoir ce qu'ils signifient pour diagnostiquer une panne. |
| 2 | **c** | Un vecteur est une simple liste de nombres. Le tableau à deux dimensions, c'est une matrice. |
| 3 | **b** | La dérivée est une pente : elle dit dans quel sens et à quelle vitesse ça monte. |
| 4 | **b** | La logique et la pensée par étapes. Un programme qui marche est d'abord un raisonnement propre. |

## Ce qu'il faut retenir

Quatre intuitions suffisent pour démarrer : probabilité, vecteur, pente,
fonction. Tout le reste du livre les réutilise sous d'autres noms.
