import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
print(df.head())

print("-------df.isnull().sum()---------")
print(df.isnull().sum())

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

#----------------------
print("-------Transformez la colonne species (et island) en variables numériques avec get_dummies---------")

print ("Avant dummies :\n" , df.head())

df = pd.get_dummies(df, columns=["species","island"])
print ("Après dummies : \n", df.head())

#----------------------
print("-------Normalisez les colonnes numériques avec StandardScaler.---------")
scaler = StandardScaler()
colonnes = ["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
df[colonnes] = scaler.fit_transform(df[colonnes])
print ("Après normalisation : \n", df.head())