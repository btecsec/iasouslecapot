# Réponse de l'exercice — Chapitre 50 : Spécialiser un modèle par le fine-tuning

## 1. Besoin concret
**Assistant Concierge de Luxe** : Un assistant qui répond avec un ton extrêmement sophistiqué, accueillant et exclusif, reflétant l'image d'une Maison de haute couture.

## 2. Exemples de qualité (entrée $\rightarrow$ sortie souhaitée)

```jsonl
{"instruction": "Où est ma commande ?", "output": "Ce serait un plaisir de suivre votre exquise acquisition. Pourriez-vous nous communiquer votre numéro de commande ?"}
{"instruction": "Avez-vous un code promo ?", "output": "Nos collections sont façonnées avec une qualité sans compromis, préservant ainsi leur valeur. Nous vous invitons à découvrir notre cercle exclusif pour vos futurs privilèges."}
{"instruction": "L'article est-il disponible ?", "output": "Permettez-moi de vérifier la disponibilité de cette pièce unique au sein de notre atelier. Un instant, je vous prie."}
{"instruction": "Comment retourner un produit ?", "output": "Nous visons la perfection dans chaque détail. Si une pièce ne comblait pas vos attentes, notre concierge vous accompagnera dans notre processus de retour fluide."}
{"instruction": "Qui êtes-vous ?", "output": "Je suis votre concierge dédié, ici pour veiller à ce que votre voyage au sein de notre Maison soit absolument exceptionnel."}
```

## 3. Choix de la méthode
**Choix : Fine-tuning.**
**Justification :** Le besoin porte sur le **style** et le **ton** (le comportement). Le prompt pourrait fonctionner, mais le fine-tuning ancre cette « voix » dans le modèle, garantissant une cohérence parfaite sans consommer de tokens de contexte à chaque interaction avec un prompt système volumineux. Le RAG est exclu car on ne cherche pas à ajouter des connaissances externes, mais à modifier la forme de la réponse.

## 4. LoRA (Low-Rank Adaptation)
LoRA évite de modifier la totalité des milliards de paramètres (poids) du modèle original. À la place, il gèle le modèle de base et n'entraîne que de petites matrices supplémentaires ajoutées aux couches existantes.
**Pourquoi c'est moins coûteux :**
- **Moins de mémoire** : On ne stocke et ne met à jour que très peu de paramètres.
- **Vitesse** : L'entraînement est beaucoup plus rapide.
- **Matériel** : Permet de faire du fine-tuning sur des GPU grand public plutôt que sur des clusters industriels.

## 5. Les 5 étapes du fine-tuning et liens
1. **Rassembler des exemples de qualité** $\rightarrow$ *chapitre 30 (Nettoyer et préparer les données)* : La qualité des données d'entrée détermine la qualité du modèle.
2. **Formater les exemples** $\rightarrow$ *chapitre 55 (Préparer les données d'entraînement)* : Transformer le texte brut en format structuré (ex: JSONL).
3. **Lancer le fine-tuning** $\rightarrow$ *chapitre 36 (Comprendre l'entraînement)* : Ajustement des poids via la descente de gradient.
4. **Évaluer le modèle affiné** $\rightarrow$ *chapitre 38 (Valider et évaluer son modèle)* : Comparer les performances sur un jeu de test réservé.
5. **Déployer le modèle** $\rightarrow$ *Partie IV (MLOps et Mise en production)* : Mise à disposition via une API et monitoring.
