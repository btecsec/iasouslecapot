# Solution de l'exercice pratique — Chapitre 46 : Surveiller son modèle en production

## 1. Indicateurs techniques à suivre
- **Temps de réponse (Latence)** : Mesurer le temps moyen et le 95ème percentile (P95) pour s'assurer que l'API reste rapide.
- **Taux d'erreur** : Pourcentage de requêtes se terminant par une erreur (ex: HTTP 500) par rapport au total.
- **Débit (Throughput)** : Nombre de requêtes par seconde pour détecter des pics de charge ou des chutes anormales de trafic.

## 2 & 3. Code de monitoring
Le code est disponible dans `monitoring_exercice.py`. Il calcule les références sur les données d'entraînement et implémente une fonction d'alerte basée sur un seuil d'écart.

## 4. Exemples de dérive
- **Dérive des données (Data Drift)** : 
  - *Exemple* : Un modèle de prédiction de prix d'appartements entraîné sur Paris. Si on commence à lui envoyer des données d'appartements de province (surfaces plus grandes, prix au m² plus bas), les données d'entrée ne ressemblent plus à l'entraînement.
- **Dérive du concept (Concept Drift)** : 
  - *Exemple* : Un modèle de détection de fraude bancaire. Les fraudeurs changent soudainement leur méthode : ils n'utilisent plus de gros montants mais une multitude de micro-transactions. La "forme" des transactions est similaire, mais le lien entre "caractéristiques de la transaction" et "est-ce une fraude" a changé.

## 5. Défi : Obtenir la "vraie réponse"
Pour mesurer l'exactitude réelle en production, on peut utiliser plusieurs stratégies :
- **L'échantillonnage manuel (Human-in-the-loop)** : Un expert humain vérifie aléatoirement 1% des prédictions du modèle pour calculer un taux d'erreur réel.
- **La donnée différée** : Dans le cas d'un prêt bancaire, on sait si le client a fait défaut (vraie réponse) seulement après plusieurs mois. On stocke la prédiction et on la compare à la réalité quand elle arrive.
- **Le feedback utilisateur** : L'utilisateur signale si la réponse était correcte (ex: "Ce message est-il un spam ? Oui/Non").
