# Chapitre 23 — Introduction aux librairies IA

Exercice d'aiguillage : **pas de code**. Pour chaque situation, quelle
bibliothèque ?

## Corrigé de l'exercice

| Situation | Bibliothèque | Pourquoi |
|---|---|---|
| Prédire le prix d'un appartement à partir d'un tableau de 5 000 annonces | **Scikit-learn** | Données tabulaires, petit volume, tâche de régression classique. Un réseau de neurones ferait *moins bien* ici, pour dix fois plus de code. |
| Un laboratoire invente une nouvelle architecture pour la vision | **PyTorch** | On écrit soi-même la boucle et les couches ; c'est le standard de la recherche, et les articles publient leur code en PyTorch. |
| Déployer un modèle sur des millions de téléphones | **TensorFlow** | Son écosystème d'export embarqué (TF Lite, TF.js, TF Serving) reste le plus mature pour le mobile et le navigateur. |
| Un débutant veut son premier petit réseau, le plus simplement possible | **Keras** | Trois lignes déclaratives : `Sequential`, `compile`, `fit`. |

### Le raisonnement derrière les quatre réponses

Deux questions suffisent à trancher presque tous les cas :

1. **Mes données tiennent-elles dans un tableau (lignes × colonnes) ?**
   Oui → Scikit-learn. Non (images, son, texte brut) → deep learning.
2. **Dois-je inventer, ou assembler ?**
   Inventer une architecture → PyTorch. Assembler des couches connues →
   Keras. Livrer sur du matériel contraint → TensorFlow.

### Le contre-exemple utile

« 5 000 annonces » est un *petit* jeu de données. Un réseau de neurones a des
millions de paramètres : sur 5 000 lignes, il apprend le bruit par cœur
(surapprentissage, chapitre 39). Une forêt aléatoire de Scikit-learn le battra
presque toujours, s'entraînera en deux secondes et restera explicable. **La
complexité du modèle doit être proportionnée au volume de données**, pas à
l'enthousiasme de l'ingénieur.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Scikit-learn règne sur le machine learning classique en données tabulaires. |
| 2 | **c** | PyTorch, pour sa flexibilité et son graphe dynamique — d'où son adoption massive par la recherche. |
| 3 | **b** | Keras est une surcouche de haut niveau, intégrée à TensorFlow, qui simplifie la construction des réseaux. |
| 4 | **b** | Commencer par Scikit-learn : les concepts (découpage, perte, évaluation) comptent plus que l'outil, et ils se transposent partout. |

## Ce qu'il faut retenir

Il n'existe pas de « meilleure » bibliothèque, seulement une meilleure
bibliothèque **pour un problème donné**. Le réflexe professionnel : commencer
par la solution la plus simple qui puisse marcher, et ne complexifier qu'avec
une mesure à l'appui.
