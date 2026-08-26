from prepare_data import prepare_penguins_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Préparation des données
    print("Préparation des données en cours...")
    X_train, X_test, y_train, y_test, scaler = prepare_penguins_data()
    print("Données prêtes.\n")

    # Dictionnaire pour stocker les résultats
    results = {}

    # --- Modèle 1 : Régression Logistique ---
    print("Entraînement de la Régression Logistique...")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    lr_score = accuracy_score(y_test, lr.predict(X_test))
    results["Régression Logistique"] = lr_score
    print(f"Score : {lr_score:.4f}\n")

    # --- Modèle 2 : Arbre de Décision ---
    print("Entraînement de l'Arbre de Décision (max_depth=3)...")
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_score = accuracy_score(y_test, dt.predict(X_test))
    results["Arbre de Décision"] = dt_score
    print(f"Score : {dt_score:.4f}\n")

    # --- Modèle 3 : Forêt Aléatoire ---
    print("Entraînement de la Forêt Aléatoire (100 arbres)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = accuracy_score(y_test, rf.predict(X_test))
    results["Forêt Aléatoire"] = rf_score
    print(f"Score : {rf_score:.4f}\n")

    # --- Comparaison finale ---
    print("-" * 30)
    print("RÉSULTATS FINAUX :")
    for model, score in results.items():
        print(f"{model} : {score:.4f}")
    print("-" * 30)

    # Réponses aux questions de l'exercice
    best_model = max(results, key=results.get)
    print(f"\nLe meilleur modèle est : {best_model}")

    # Le plus simple suffit-il ? (Dépend du résultat, mais généralement la logistique est très proche pour ce dataset)
    simplest_score = results["Régression Logistique"]
    best_score = results[best_model]
    diff = best_score - simplest_score

    if diff < 0.02:
        print("Le modèle le plus simple (Régression Logistique) suffit largement car la différence de performance est négligeable.")
    else:
        print(f"Le modèle le plus simple a un écart de {diff:.4f} avec le meilleur. Un modèle plus complexe apporte un gain visible.")

    print("\nPourquoi une baseline simple est-elle utile ?")
    print("1. Elle donne un point de référence : savoir si un modèle complexe (comme la Forêt Aléatoire) vaut vraiment la peine pour un gain marginal.")
    print("2. Elle est plus rapide à entraîner et plus facile à expliquer (interprétabilité).")
    print("3. Elle permet de détecter rapidement si les données sont linéairement séparables.")

if __name__ == "__main__":
    main()
