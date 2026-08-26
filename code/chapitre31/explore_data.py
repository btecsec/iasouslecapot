import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

print("-------df.shape---------")
print(df.shape)

print("-------df.info()---------")
print(df.info())

print("-------df.describe()---------")
print(df.describe())

print("-------df.isnull().sum()---------")
print(df.isnull().sum())

print("-------df['species'].value_counts()---------")
print(df['species'].value_counts())

print("-------df.corr(numeric_only=true)---------")
print(df.corr(numeric_only=True))

# Distribution d'une variable numérique
sns.histplot(df["body_mass_g"])
plt.show()

# Relation entre deux variables, colorée par espèce
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g", hue="species")
plt.show()