import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Charger un dataset d'exemple intégré à Seaborn
df = sns.load_dataset("penguins")

# les features (colonnes de dataset)
#| Nom de colonne (anglais) | Traduction en français           |
#| ------------------------ | -------------------------------- |
#| bill_length_mm           | Longueur du bec (mm)             |
#| bill_depth_mm            | épaisseur du bec (mm) |
#| flipper_length_mm        | Longueur des nageoires (mm)      |
#| body_mass_g              | Masse corporelle (g)             |

# Afficher les 5 premières lignes
print("-------df.head()---------")
print(df.head(10))

print("-------remplir les valeurs null de bill_length_mm par le median---------")
df["bill_length_mm"] = df["bill_length_mm"].fillna(df["bill_length_mm"].median())

print("-------remplir les valeurs null de bill_depth_mm par le median---------")
df["bill_depth_mm"] = df["bill_depth_mm"].fillna(df["bill_depth_mm"].median())

print("-------remplir les valeurs null de flipper_length_mm par le median---------")
df["flipper_length_mm"] = df["flipper_length_mm"].fillna(df["flipper_length_mm"].median())

print("-------remplir les valeurs null de body_mass_g par le median---------")
df["body_mass_g"] = df["body_mass_g"].fillna(df["body_mass_g"].median())

print("-------df.isnull().sum()---------")
print(df.isnull().sum())

#---------------------------------------------------------------
print("-------afficher les doublons---------")
print(df.duplicated().sum())

print("-------supprimer les doublons---------")
df = df.drop_duplicates()

print("-------ré-afficher les doublons---------")
print(df.duplicated().sum())
print (df.shape)

#-------------------------------------------
print("-------Transformez la colonne  island en variables numériques avec get_dummies---------")

print ("Avant dummies :\n" , df.head())

df = pd.get_dummies(df, columns=["island", "sex"], drop_first=False)
print ("Après dummies : \n", df.head())


#--------------------------------------------
y = df["species"]
X = df.drop(columns=["species"])

# Découpage 80% train / 20% test, reproductible
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20 % pour le test
    stratify=y,
    random_state=42     # pour un découpage reproductible
)

print("Taille X_train :", X_train.shape)
print("Taille X_test  :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test  :", y_test.shape)

print("Proportions dans y (complet) :")
print(y.value_counts(normalize=True))

print("\nProportions dans y_train :")
print(y_train.value_counts(normalize=True))

print("\nProportions dans y_test :")
print(y_test.value_counts(normalize=True))

#----------------------------------------------------------------

#print("-------Normalisez les colonnes numériques avec StandardScaler.---------")
scaler = StandardScaler()
colonnes = ["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]

# fit_transform SEULEMENT sur X_train (apprend moyenne/écart-type + transforme)
X_train[colonnes] = scaler.fit_transform(X_train[colonnes])

# transform SEULEMENT sur X_test (réutilise les paramètres appris sur train)
X_test[colonnes] = scaler.transform(X_test[colonnes])



#---------------------------------------------------------------
# Modèle simple : on verra le choix de modele  dans les chapitre suivant
model = LogisticRegression(max_iter=1000, random_state=42)

# Validation croisée à 5 plis sur les données d'entraînement déjà normalisées
scores = cross_val_score(model, X_train, y_train, cv=5)

print("Scores par pli :", scores)
print("Score moyen :", scores.mean())
print("Écart-type :", scores.std())