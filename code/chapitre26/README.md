# Chapitre 26 — Choisir son camp : PyTorch ou TensorFlow ?

Exercice de décision argumentée : **pas de code**. Un framework par profil,
justifié en une phrase.

## Corrigé de l'exercice

| Profil | Choix | Justification |
|---|---|---|
| Étudiante en recherche qui veut reproduire un article récent sur les LLM | **PyTorch** | Le code des articles et les modèles de Hugging Face sont publiés en PyTorch : reproduire signifie souvent *exécuter tel quel*. |
| Start-up qui met un modèle de vision dans son application mobile | **TensorFlow (Lite)** | La chaîne d'export vers Android/iOS est éprouvée : quantification, taille du binaire, accélérateurs matériels. |
| Débutant qui construit son premier réseau ce week-end | **Keras** | Le chemin le plus court entre une idée et un `fit()` qui tourne. |
| Équipe qui expérimente une architecture inédite | **PyTorch** | Le graphe dynamique permet de déboguer avec un simple `print()` au milieu du `forward`, comme du Python ordinaire. |

### La bonne façon de justifier

Une justification faible parle de goût (« PyTorch est plus élégant »). Une
justification solide parle de **contrainte** :

- contrainte de **cible** (mobile, navigateur, serveur, GPU unique) ;
- contrainte d'**écosystème** (le modèle pré-entraîné dont j'ai besoin existe-t-il
  déjà dans ce framework ?) ;
- contrainte d'**équipe** (qui va maintenir ce code dans deux ans ?).

C'est aussi ce qui rend la question moins dramatique qu'elle n'en a l'air : les
concepts — tenseur, perte, optimiseur, boucle d'entraînement, sauvegarde des
poids — sont identiques des deux côtés. Vous l'avez d'ailleurs vérifié aux
chapitres 24 et 25, où les deux exercices se ressemblaient ligne pour ligne.

### Tableau de décision rapide

| Votre situation | Prenez |
|---|---|
| Je débute en deep learning | Keras |
| Je travaille sur des LLM ou du NLP moderne | PyTorch |
| Je vise le mobile ou le navigateur | TensorFlow |
| Je reproduis un article de recherche | PyTorch |
| Mon équipe a déjà du code dans l'un des deux | **Celui-là**, sans hésiter |

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **c** | Les deux ont convergé : Keras s'utilise avec plusieurs moteurs, PyTorch a gagné la compilation. Les écarts de performance dépendent plus du code que du logo. |
| 2 | **b** | PyTorch domine la recherche récente et l'écosystème LLM (Hugging Face, entraînement distribué). |
| 3 | **b** | TensorFlow, via TF Lite et TensorFlow.js. |
| 4 | **c** | Le choix compte peu : ce sont les concepts qui comptent, et la pratique. Passer de l'un à l'autre prend une semaine. |

## Ce qu'il faut retenir

Ne choisissez pas un framework « pour la vie ». Choisissez celui qu'impose
votre cible de déploiement ou l'écosystème du modèle que vous réutilisez — et
apprenez les concepts, qui, eux, sont transférables.
