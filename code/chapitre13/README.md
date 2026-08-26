# Chapitre 13 — Les grandes familles d'IA

Exercice de classement : **pas de code**. Pour chaque application, la famille
la plus précise + une justification.

## Corrigé de l'exercice

| Application | Famille | Justification |
|---|---|---|
| Thermostat « si température < 19 °C, allumer » | **IA classique** (symbolique) | La règle a été *écrite* par un humain. Rien n'est appris, rien ne change avec l'usage. |
| Prédire le prix d'un appartement à partir d'un tableau | **Machine learning** | La relation surface/quartier/étage → prix est *apprise* à partir d'exemples ; personne ne l'a écrite. Données tabulaires, donc pas besoin de deep learning. |
| Reconnaître des panneaux sur des photos | **Deep learning** | L'entrée est une image brute : il faut des couches qui construisent elles-mêmes leurs indices visuels (contours, formes, symboles). Un CNN. |
| Écrire un article à partir d'un titre | **IA générative** | La sortie est un contenu inédit, pas une catégorie ni un nombre. C'est du deep learning, sous-famille générative. |

### La règle de lecture : « la plus précise »

L'énoncé demande la famille **la plus précise**, et c'est là que se joue
l'exercice. Les cases s'emboîtent :

```text
IA  ⊃  Machine learning  ⊃  Deep learning  ⊃  IA générative
```

Répondre « IA » pour le cas 4 n'est pas faux, c'est juste inutile — comme
répondre « un véhicule » quand on demande la marque de la voiture. Et
répondre « deep learning » pour le cas 2 est une **erreur d'ingénieur** : sur
un tableau de quelques milliers de lignes, un réseau profond surapprend
(chapitre 39) là où une régression fait mieux en deux secondes.

### Les deux questions qui suffisent à classer

1. **Le comportement a-t-il été appris à partir d'exemples ?**
   Non → IA classique. Oui → machine learning.
2. **Si oui : l'entrée est-elle brute et complexe (image, son, texte) ?**
   Non (un tableau propre) → ML classique.
   Oui → deep learning ; et si la sortie est un *contenu créé*, générative.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `IA ⊃ ML ⊃ DL ⊃ IA générative` : chaque famille est un cas particulier de la précédente. |
| 2 | **b** | Le ML apprend ses règles à partir d'exemples, au lieu de les recevoir écrites. |
| 3 | **b** | Le DL empile des couches et sait traiter des données brutes complexes, en construisant lui-même ses caractéristiques. |
| 4 | **b** | Les LLM sont de l'IA générative, donc du deep learning. |

## Ce qu'il faut retenir

Ces familles ne sont pas des camps rivaux mais des poupées russes. Le réflexe
professionnel : descendre dans la complexité **seulement quand la famille
au-dessus a échoué**, mesures à l'appui.
