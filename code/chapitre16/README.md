# Chapitre 16 — Les grands types de tâches en IA

Exercice de classement : **pas de code**. Quelle tâche pour chaque problème ?

## Corrigé de l'exercice

| Problème | Tâche | Justification |
|---|---|---|
| Prédire le nombre de vélos loués demain | **Régression** | La sortie est un nombre continu (247 vélos). On mesurera l'erreur en MAE ou RMSE (chapitre 40). |
| Transaction frauduleuse ou légitime ? | **Classification** (binaire) | Deux catégories définies à l'avance. Attention : classes très déséquilibrées, l'accuracy sera trompeuse. |
| Rassembler 10 000 chansons en styles, sans liste imposée | **Regroupement** | Aucune catégorie fournie : les groupes doivent émerger des données. |
| Écrire le résumé d'un article | **Génération** | La sortie est un texte inédit, ni une classe ni un nombre. |
| Reconnaître le chiffre 0-9 écrit à la main | **Classification** (10 classes) | Dix catégories connues d'avance. C'est le fameux MNIST. |

### Les deux confusions classiques

**Classification vs régression.** Le critère n'est pas « y a-t-il des chiffres
en sortie » mais « les valeurs ont-elles un ordre et des écarts qui ont un
sens ? ».

- Prédire une note de 1 à 5 étoiles : les deux réponses se défendent.
  Classification si l'on veut juste la bonne case, régression si l'on
  considère que se tromper de 3 étoiles est bien pire que d'une seule.
- Reconnaître un chiffre manuscrit : **classification**, toujours. Le chiffre
  « 8 » n'est pas « deux fois plus » que le « 4 » ; confondre 8 et 4 n'est pas
  moins grave que confondre 8 et 1.

**Classification vs regroupement.** Cas 3 contre cas 2 et 5 : dans le
regroupement, **personne ne connaît les catégories avant**. C'est aussi ce qui
rend l'évaluation difficile — il n'y a pas de bonne réponse à comparer, donc
pas d'accuracy possible.

### Le tableau de correspondance à garder

| Tâche | Sortie | Métrique typique | Exemple du livre |
|---|---|---|---|
| Classification | une catégorie | accuracy, précision, rappel, F1 | l'espèce d'un manchot (ch. 31) |
| Régression | un nombre continu | MAE, RMSE, R² | la masse d'un manchot (ch. 36) |
| Regroupement | des groupes découverts | silhouette, inertie | — |
| Génération | un contenu inédit | perplexité, jugement humain | le mini-LLM (ch. 54) |

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | La classification sort une catégorie choisie dans une liste connue d'avance. |
| 2 | **b** | Un prix exact est un nombre continu : régression. |
| 3 | **b** | En classification les catégories sont définies à l'avance ; en regroupement elles émergent des données. |
| 4 | **c** | Un LLM qui rédige fait de la génération. |

## Ce qu'il faut retenir

Nommer la tâche est le premier geste d'un projet : il détermine le modèle, la
fonction de perte et la métrique d'évaluation. Une tâche mal nommée, et les
trois suivent dans l'erreur.
