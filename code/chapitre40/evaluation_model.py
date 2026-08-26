import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score

# --- 1. Classification: Rare Species Detection ---
# Objectif : Détecter une espèce rare (classe minoritaire)
print("=== 1. Classification : Détection d'espèces rares ===")

# Simulation : 100 échantillons, seulement 5 sont de l'espèce rare (Données déséquilibrées)
# y_true : étiquettes réelles (0 = espèce commune, 1 = espèce rare)
y_true = np.array([0]*95 + [1]*5)

# Simulation des prédictions du modèle
# On crée un vecteur de zéros et on ajoute quelques prédictions positives
y_pred = np.zeros(100)
y_pred[95:100] = [1, 1, 0, 0, 1] # 3 Vrais Positifs (TP), 2 Faux Négatifs (FN)
y_pred[10] = 1 # 1 Faux Positif (FP)
y_pred[20] = 1 # 1 Faux Positif (FP)

# Matrice de confusion : permet de voir précisément où le modèle se trompe
# Format : [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_true, y_pred)
print("Matrice de confusion :\n", cm)
print(f"Faux Positifs (FP) : {cm[0,1]} (L'espèce est commune mais prédite comme rare)")
print(f"Faux Négatifs (FN) : {cm[1,0]} (L'espèce est rare mais prédite comme commune)")

# Rapport de classification : Précision, Rappel, F1-Score
print("\nRapport de classification :\n", classification_report(y_true, y_pred))

# --- Discussion sur le choix de la métrique ---
# Imaginez que rater un positif (FN) coûte très cher (ex: on rate une espèce en danger critique) :
# On doit privilégier le RAPPEL (Recall).
# Le rappel répond à : "Sur tous les vrais positifs, combien en ai-je trouvé ?"
# On acceptera d'avoir plus de Faux Positifs (baisse de précision) pour être sûr de ne rater aucun positif.



# --- 2. Régression : Prédiction de masse ---
print("\n=== 2. Régression : Prédiction de masse ===")

# Simulation : Masses réelles vs Masses prédites
y_true_reg = np.array([10.5, 20.2, 15.8, 30.1, 25.4])
y_pred_reg = np.array([10.0, 21.0, 14.5, 31.0, 24.0])

# MAE (Mean Absolute Error) : Erreur moyenne en unités réelles (ex: kg)
mae = mean_absolute_error(y_true_reg, y_pred_reg)
# R² (Coefficient de détermination) : Proportion de la variance expliquée par le modèle (0 à 1)
r2 = r2_score(y_true_reg, y_pred_reg)

print(f"MAE (Erreur absolue moyenne) : {mae:.2f}")
print(f"R2 Score (Coefficient de détermination) : {r2:.2f}")

# --- 3. Pourquoi l'accuracy seule est trompeuse sur une classe rare ? ---
#print("\n=== 3. Le piège de l'Accuracy ===")
# Si on a 99% de classe 'Commune' et 1% de 'Rare',
# un modèle qui prédit TOUJOURS 'Commune' aura 99% d'accuracy.
# Pourtant, il est totalement inutile car il rate 100% des espèces rares.
# C'est pour cela qu'on utilise la matrice de confusion et le rappel pour les classes rares.
