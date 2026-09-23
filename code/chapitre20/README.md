# Chapitre 20 — Panorama des applications concrètes

Exercice ouvert : **pas de code**, et pas de réponse unique. On vous demande de
concevoir une application dans un secteur qui vous intéresse, puis de la
décrire avec le vocabulaire des chapitres 15 à 17.

## Corrigé : trois exemples complets

Prenez-les comme des modèles de *forme*, pas comme des réponses à recopier.

### Exemple A — Santé : dépister la rétinopathie diabétique

| Rubrique | Réponse |
|---|---|
| Application | Analyser une photo du fond de l'œil et signaler les cas suspects à l'ophtalmologue. |
| **Tâche** | Classification (chapitre 16) : suspect / non suspect. |
| **Mode d'apprentissage** | Supervisé (chapitre 15) : chaque image est étiquetée par le diagnostic d'un spécialiste. |
| **Limite / risque éthique** | Un faux négatif (patient malade déclaré sain) est bien plus grave qu'un faux positif. On privilégiera donc le **rappel** (chapitre 40), quitte à générer des alertes inutiles. Et le modèle **assiste** le médecin : il ne décide pas. |

### Exemple B — Écologie : prédire la consommation électrique d'un bâtiment

| Rubrique | Réponse |
|---|---|
| Application | Prévoir la consommation de demain pour lisser les pics. |
| **Tâche** | Régression : un nombre en kWh. |
| **Mode d'apprentissage** | Supervisé : l'historique fournit la vraie consommation de chaque jour passé. |
| **Limite / risque** | Le modèle apprend sur le passé ; une rénovation thermique ou un été caniculaire cassent la relation apprise. C'est de la **dérive du concept** (chapitre 48), à surveiller. |

### Exemple C — Jeu vidéo : équilibrer automatiquement la difficulté

| Rubrique | Réponse |
|---|---|
| Application | Regrouper les joueurs par style de jeu pour adapter les défis. |
| **Tâche** | Regroupement (*clustering*) : les profils ne sont pas connus d'avance. |
| **Mode d'apprentissage** | Non supervisé. |
| **Limite / risque** | Un « profil » découvert par la machine n'a pas de sens intrinsèque : c'est un humain qui l'interprète, et qui peut se tromper. Risque supplémentaire : optimiser l'engagement peut glisser vers l'addiction — un objectif technique bien atteint peut être un objectif humain mal choisi. |

### La grille à réutiliser

Pour n'importe quelle idée d'application, quatre questions suffisent :

1. **Quelle tâche ?** classification, régression, regroupement, génération.
2. **Quelles données, et sont-elles étiquetées ?** → supervisé / non supervisé.
3. **Quelle métrique, et quelle erreur coûte le plus cher ?** → précision vs rappel.
4. **Qui subit une erreur du modèle ?** → c'est la question éthique, et elle
   se pose *avant* d'écrire du code, pas après le déploiement.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | En santé, l'IA assiste le soignant, qui garde la responsabilité de la décision. |
| 2 | **b** | La fraude bancaire est avant tout de la détection d'anomalies : les cas positifs sont rarissimes et changeants. |
| 3 | **b** | L'IA générative et les LLM ont fait entrer l'IA dans les usages quotidiens. |
| 4 | **b** | L'IA hérite des biais présents dans ses données — elle ne les invente pas, elle les reproduit et parfois les amplifie. |

## Ce qu'il faut retenir

Une application d'IA se décrit toujours par un quadruplet : tâche, données,
métrique, risque. Si vous ne savez pas remplir les quatre cases, le projet
n'est pas encore prêt à être codé.
