import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_penguins_data():
    """
    Charge et prépare le dataset des manchots pour l'entraînement d'un modèle.
    Retourne X_train, X_test, y_train, y_test et le scaler utilisé.
    """
    # Charger un dataset d'exemple intégré à Seaborn
    df = sns.load_dataset("penguins")

    # Remplir les valeurs nulles par la médiane
    colonnes_numeriques = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    for col in colonnes_numeriques:
        df[col] = df[col].fillna(df[col].median())

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Transformez la colonne island et sex en variables numériques avec get_dummies
    df = pd.get_dummies(df, columns=["island", "sex"], drop_first=False)

    # Définir la cible y et les caractéristiques X
    y = df["species"]
    X = df.drop(columns=["species"])

    # Découpage 80% train / 20% test, reproductible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20 % pour le test
        stratify=y,
        random_state=42     # pour un découpage reproductible
    )

    # Normalisez les colonnes numériques avec StandardScaler
    scaler = StandardScaler()

    # fit_transform SEULEMENT sur X_train (apprend moyenne/écart-type + transforme)
    X_train[colonnes_numeriques] = scaler.fit_transform(X_train[colonnes_numeriques])

    # transform SEULEMENT sur X_test (réutilise les paramètres appris sur train)
    X_test[colonnes_numeriques] = scaler.transform(X_test[colonnes_numeriques])

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Ce bloc permet de tester le script individuellement
    X_train, X_test, y_train, y_test, scaler = prepare_penguins_data()
    print("Données préparées avec succès.")
    print(f"Taille de X_train: {X_train.shape}")
    print(f"Taille de X_test: {X_test.shape}")
