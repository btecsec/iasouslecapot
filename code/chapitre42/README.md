# Chapitre 42 — Sauvegarder et recharger son modèle

## L'énoncé

1. Sauvegardez un modèle Scikit-learn avec joblib, rechargez-le dans un
   nouveau script, vérifiez qu'il prédit pareil.
2. Faites de même avec un modèle Keras (`.keras`).
3. En PyTorch, sauvegardez le `state_dict`, recréez le réseau, rechargez les
   poids, appelez `eval()`.
4. Sauvegardez aussi le scaler utilisé pour préparer les données.
5. Comparez la taille des fichiers « modèle complet » et « poids seuls ».

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `save_model_scikitlearn.py`, `save_model_keras.py`, `save_model_torch.py` | Les scripts du chapitre, un par framework. |
| `exercice_sauvegarde.py` | Le corrigé, avec la démonstration du scaler oublié. |
| `test_exercice_sauvegarde.py` | Les tests des trois frameworks (aller-retour complet). |

```bash
pip install -r requirements.txt
python exercice_sauvegarde.py     # écrit dans modeles/
pytest -q
```

## Corrigé

### 1. Scikit-learn : joblib

```python
import joblib
joblib.dump(modele, "foret.joblib")
modele = joblib.load("foret.joblib")
```

Le critère de réussite n'est pas « le fichier existe » mais **« les mêmes
entrées donnent exactement les mêmes sorties »**. C'est ce que vérifie
`test_le_modele_recharge_predit_exactement_pareil`, et c'est le test que vous
devriez écrire dans tous vos projets.

### 2. Keras : le format `.keras`

```python
modele.save("modele.keras")
modele = keras.models.load_model("modele.keras")
```

Un seul fichier qui contient **tout** : architecture, poids, configuration de
compilation, état de l'optimiseur. Vous pouvez reprendre l'entraînement là où
vous l'aviez laissé.

### 3. PyTorch : le `state_dict`

```python
torch.save(reseau.state_dict(), "poids.pt")

reseau = ReseauManchots()                  # 1. recréer l'architecture
reseau.load_state_dict(torch.load("poids.pt", weights_only=True))   # 2. les poids
reseau.eval()                              # 3. mode évaluation
```

Trois étapes, contre une seule en Keras — c'est la philosophie PyTorch : le
fichier ne contient que **les nombres appris**. Vous devez avoir le code de la
classe pour reconstruire la coquille. En pratique, on sauvegarde donc aussi la
configuration (dimensions, nombre de couches) à côté des poids : sans elle,
personne ne peut reconstruire la coquille.

Et `eval()` n'est pas décoratif : sans lui, un réseau contenant du dropout
continue d'éteindre des neurones au hasard **pendant les prédictions**.

### 4. Le scaler — le vrai sujet du chapitre

Voici la mesure, sur 69 manchots de test :

```text
sans le scaler : 55 predictions sur 69 changent
                 — et aucune erreur ne s'affiche
```

**80 % des prédictions deviennent fausses, en silence.** C'est le pire type de
bug : le programme tourne, renvoie des espèces plausibles, et se trompe.

Trois solutions, de la moins à la plus sûre :

```python
# 1. Deux fichiers — ça marche, mais rien n'empêche d'en oublier un
joblib.dump(modele, "modele.joblib")
joblib.dump(scaler, "scaler.joblib")

# 2. Un paquet — mieux : tout voyage ensemble
joblib.dump({"modele": modele, "scaler": scaler, "colonnes": list(X.columns)},
            "complet.joblib")

# 3. Un pipeline — le meilleur : le scaler EST le modèle
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
joblib.dump(pipeline, "pipeline.joblib")
```

**Pourquoi sauvegarder aussi la liste des colonnes ?** Parce que c'est le
troisième oubli classique. Une API qui reçoit un JSON n'a aucune garantie sur
l'ordre des champs ; si `bill_length` et `flipper_length` sont intervertis, le
modèle prédit sans broncher. Le test
`test_le_paquet_remet_les_colonnes_dans_le_bon_ordre` verrouille ce point.

**Règle générale** : on sauvegarde *tout ce qui a été appris sur les données
d'entraînement* — scaler, encodeur de labels, vocabulaire, liste et ordre des
colonnes. Le modèle n'est qu'une partie de l'objet à livrer.

### 5. Les tailles

```text
foret.joblib                 261.3 Ko
manchots_complet.joblib      262.0 Ko
pipeline.joblib              262.1 Ko
```

**0,7 Ko de plus** pour ne plus jamais avoir le bug ci-dessus. Le scaler ne
contient que quatre moyennes et quatre écarts-types.

La comparaison « modèle complet contre poids seuls » se voit mieux en deep
learning : un `.keras` complet embarque en plus l'état de l'optimiseur Adam,
qui garde **deux valeurs par paramètre**. Un checkpoint d'entraînement pèse
donc environ trois fois le poids du modèle seul — d'où la distinction entre le
checkpoint (pour reprendre l'entraînement) et l'export (pour servir en
production, chapitre 45).

### ⚠️ Sécurité : ne chargez jamais un modèle inconnu

`joblib.load` et `pickle.load` **exécutent du code** en désérialisant.
Télécharger le « modèle entraîné » d'un inconnu et l'ouvrir revient à exécuter
son script sur votre machine.

- Vos propres modèles, dans votre dépôt : aucun problème.
- Un modèle tiers : préférez un format qui ne transporte que des nombres
  (ONNX, safetensors), ou `torch.load(..., weights_only=True)`.

Le réflexe tient en une phrase : un fichier de modèle est du code, pas une
donnée.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Pour ne pas devoir réentraîner à chaque exécution — un entraînement peut prendre des heures. |
| 2 | **b** | `joblib.dump` est le format standard pour Scikit-learn. |
| 3 | **b** | Le `state_dict` contient les poids appris, pas l'architecture. |
| 4 | **a** | Avoir oublié de sauvegarder le scaler ou l'encodeur : les prédictions deviennent fausses, sans aucune erreur. |

## Ce qu'il faut retenir

Un modèle livré, c'est un modèle **plus** tout ce qui a été appris avec lui.
Le pipeline règle la question d'un coup — et le test à écrire tient en une
ligne : mêmes entrées, mêmes sorties, avant et après rechargement.
