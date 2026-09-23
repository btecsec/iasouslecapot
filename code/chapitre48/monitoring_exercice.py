import pandas as pd
import numpy as np

def envoyer_alerte(message):
    print(f"⚠️ ALERTE : {message}")

def calculer_references(df_train, colonne):
    """Calcule la moyenne et l'écart-type comme références."""
    mean_ref = df_train[colonne].mean()
    std_ref = df_train[colonne].std()
    return mean_ref, std_ref

def verifier_derive(donnees_recues, colonne, mean_ref, std_ref, seuil_sigma=2):
    """
    Vérifie si la moyenne des données reçues s'écarte trop de la référence.
    Seuil par défaut : 2 fois l'écart-type (règle empirique).
    """
    moyenne_actuelle = donnees_recues[colonne].mean()
    ecart = abs(moyenne_actuelle - mean_ref)

    print(f"Moyenne ref: {mean_ref:.2f} | Moyenne actuelle: {moyenne_actuelle:.2f} | Écart: {ecart:.2f}")

    if ecart > (seuil_sigma * std_ref):
        envoyer_alerte(f"Dérive détectée sur {colonne}. L'écart ({ecart:.2f}) dépasse le seuil ({seuil_sigma * std_ref:.2f}).")
    else:
        print("✅ Aucune dérive significative détectée.")

# --- Simulation ---
if __name__ == "__main__":
    # 1. Création de données d'entraînement (Manchots adultes)
    np.random.seed(42)
    df_train = pd.DataFrame({
        "masse": np.random.normal(loc=3000, scale=200, size=1000) # Moyenne 3000g
    })

    # Calcul des références
    ref_mean, ref_std = calculer_references(df_train, "masse")
    print(f"Références établies -> Moyenne: {ref_mean:.2f}, Std: {ref_std:.2f}\n")

    # 2. Cas normal : données de production similaires
    print("Test 1 : Données normales")
    df_prod_ok = pd.DataFrame({"masse": np.random.normal(loc=3050, scale=200, size=100)})
    verifier_derive(df_prod_ok, "masse", ref_mean, ref_std)

    print("\n" + "-"*30 + "\n")

    # 3. Cas de dérive : données de production différentes (ex: jeunes manchots)
    print("Test 2 : Données avec dérive")
    df_prod_drift = pd.DataFrame({"masse": np.random.normal(loc=2400, scale=200, size=100)})
    verifier_derive(df_prod_drift, "masse", ref_mean, ref_std)
