import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

# 1. Chargez, nettoyez et encodez les données
print("--- Phase 1: Préparation des données ---")
df = sns.load_dataset('penguins')

# Nettoyage : suppression des valeurs manquantes
df = df.dropna()

# Encodage des variables catégorielles
# Cible : species
le_species = LabelEncoder()
y = le_species.fit_transform(df['species'])

# Features : island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
X = df.drop('species', axis=1)

# On encode les autres colonnes catégorielles (island, sex) avec get_dummies
X = pd.get_dummies(X, columns=['island', 'sex'], drop_first=True)

print(f"Données chargées : {X.shape[0]} échantillons, {X.shape[1]} features.")

# 2. Découpez en 80/20 stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Découpage 80/20 stratifié terminé.")

# 3. Entraînez une régression logistique et affichez le score de test
print("\n--- Phase 2: Modèle sans mise à l'échelle ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
score_simple = accuracy_score(y_test, y_pred)
print(f"Score de test (Simple) : {score_simple:.4f}")

# 4. Affichez le classification_report et repérez la classe la moins bien prédite
report = classification_report(y_test, y_pred, target_names=le_species.classes_)
print("\nClassification Report :")
print(report)

# 5. Reconstruisez le tout avec un make_pipeline incluant un StandardScaler
print("\n--- Phase 3: Modèle avec Pipeline et StandardScaler ---")
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
score_pipe = accuracy_score(y_test, y_pred_pipe)
print(f"Score de test (Pipeline) : {score_pipe:.4f}")

print(f"\nLe score a-t-il changé ? {'Oui' if score_simple != score_pipe else 'Non'}")
print(f"Différence : {score_pipe - score_simple:.4f}")
