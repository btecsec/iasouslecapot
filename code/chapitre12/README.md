# Chapitre 12 — Qu'est-ce que l'intelligence artificielle, vraiment ?

Exercice d'observation : **pas de code**. Notez cinq moments où vous avez
utilisé une IA sans y penser, et répondez à trois questions pour chacun.

## Corrigé de l'exercice

Voici cinq cas typiques d'une journée ordinaire, traités selon la grille
demandée.

| Usage | Tâche précise | Données d'apprentissage probables | Faible ou forte ? |
|---|---|---|---|
| Le clavier du téléphone propose le mot suivant | prédire le mot suivant | des milliards de phrases, plus **vos** frappes | faible |
| Le filtre anti-spam de la boîte mail | classer un message en spam / légitime | des millions de mails déjà étiquetés par les utilisateurs (« signaler comme spam ») | faible |
| Le GPS annonce 32 minutes de trajet | prédire une durée (régression) | l'historique de circulation, la météo, la position anonymisée des téléphones | faible |
| La galerie photo trouve « chien » | classer des images | des bases d'images étiquetées, plus vos albums | faible |
| La plateforme de vidéo propose la suivante | ordonner des recommandations | l'historique de visionnage de millions de comptes | faible |

### Pourquoi la réponse à la question 3 est toujours « faible »

Une IA **forte** (ou générale) résoudrait n'importe quel problème intellectuel
comme un humain, transférerait ce qu'elle apprend d'un domaine à l'autre, et
aurait conscience de ce qu'elle fait. **Elle n'existe pas.** Pas « pas encore
au point » : elle n'existe pas du tout, à aucun degré.

Le piège de l'exercice est là : un LLM *paraît* général parce qu'il traite du
texte sur tous les sujets. Mais il fait une seule chose — prédire le jeton
suivant (chapitre 21). Sa polyvalence apparente vient de l'étendue de son
corpus, pas d'une compréhension.

### La question qui démasque une « fausse IA »

Beaucoup de produits vendus comme « intelligents » ne sont que des `if`. Le
test décisif : **le comportement a-t-il été appris à partir d'exemples, ou
écrit à la main par un développeur ?**

- Thermostat « si T < 19 alors chauffer » → des règles écrites. Pas du ML.
- Filtre anti-spam entraîné sur des millions de mails → du ML.

Ce critère est celui du chapitre 13, et il vous servira toute votre carrière —
notamment pour ne pas construire un réseau de neurones là où trois lignes de
règles suffisent.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Seule l'IA faible (étroite), spécialisée sur une tâche, existe aujourd'hui. |
| 2 | **c** | Elle ajuste ses paramètres à partir de nombreux exemples (chapitre 14). |
| 3 | **b** | La rencontre de trois choses : les données, la puissance de calcul et de meilleurs algorithmes. Aucune n'aurait suffi seule. |
| 4 | **b** | Il prédit le mot (jeton) suivant le plus probable. Rien de plus, rien de moins. |

## Ce qu'il faut retenir

L'IA d'aujourd'hui est un ensemble d'outils spécialisés très performants, pas
un cerveau artificiel. Savoir dire *quelle tâche* un système accomplit et *sur
quelles données* il a appris vous protège des deux excès : la peur et la
crédulité.
