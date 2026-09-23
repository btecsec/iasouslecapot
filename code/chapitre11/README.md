# Chapitre 11 — Visualiser ses données avec Matplotlib et Seaborn

## L'énoncé

1. Créez `jours = ["Lun", "Mar", "Mer", "Jeu", "Ven"]` et
   `visiteurs = [200, 240, 180, 300, 260]`.
2. Tracez une courbe de la fréquentation, avec titre et axes légendés.
3. Refaites la même chose en barres.
4. Bonus : un histogramme d'une liste d'âges.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_visualisation.py` | Le corrigé : une fonction par graphique, qui **renvoie** la figure. |
| `test_exercice_visualisation.py` | Les tests unitaires (aucune fenêtre ne s'ouvre). |
| `requirements.txt` | matplotlib, seaborn, pytest. |

```bash
pip install -r requirements.txt
python exercice_visualisation.py        # ouvre les fenêtres
python exercice_visualisation.py --png  # écrit les .png à la place
pytest -q
```

## Corrigé commenté

```python
import matplotlib.pyplot as plt

jours = ["Lun", "Mar", "Mer", "Jeu", "Ven"]
visiteurs = [200, 240, 180, 300, 260]

figure, axes = plt.subplots()
axes.plot(jours, visiteurs, marker="o")
axes.set_title("Fréquentation de la semaine")
axes.set_xlabel("Jour")
axes.set_ylabel("Nombre de visiteurs")
plt.show()
```

### Pourquoi `plt.subplots()` plutôt que `plt.plot()` ?

`plt.plot()` dessine dans « la figure courante », une variable globale cachée.
Ça marche pour un graphique jeté, ça devient ingérable dès qu'il y en a deux.
La forme `figure, axes = plt.subplots()` vous donne deux objets nommés :

- la **figure** = la feuille de papier ;
- les **axes** = le graphique dessiné dessus.

C'est aussi ce qui rend le code **testable** : une fonction qui renvoie sa
figure peut être vérifiée sans jamais ouvrir de fenêtre, comme dans
`test_exercice_visualisation.py`.

### Les trois graphiques, et quand les utiliser

| Fonction | Question à laquelle il répond | Exemple ici |
|---|---|---|
| `axes.plot` | comment ça évolue **dans le temps** ? | fréquentation jour par jour |
| `axes.bar` | comment se **comparent des catégories** ? | les mêmes jours, vus comme 5 boîtes |
| `axes.hist` | comment se **répartissent** mes valeurs ? | les âges, par tranches |

L'erreur fréquente est de tracer une courbe entre des catégories sans ordre
(« Paris, Lyon, Marseille ») : la ligne suggère une progression qui n'existe
pas. Dans notre exercice la courbe est légitime, parce que les jours *sont*
ordonnés.

### Réponses aux observations attendues

- Le mercredi est le creux (180), le jeudi le pic (300).
- Sur l'histogramme des âges, `bins=5` et `bins=20` racontent deux histoires
  différentes avec les mêmes données : le nombre de tranches est un choix, et
  ce choix se justifie.
- `plt.show()` est **bloquant** : le script s'arrête tant que la fenêtre est
  ouverte. Dans un script automatisé, on lui préfère `figure.savefig(...)`.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Pour repérer tendances, anomalies et formes que les statistiques seules cachent. Le quartet d'Anscombe : quatre jeux de données de moyennes et écarts-types identiques, et quatre dessins radicalement différents. |
| 2 | **c** | Le graphique en barres compare des catégories. |
| 3 | **b** | Le nuage de points montre le lien entre deux variables (corrélation). |
| 4 | **b** | Seaborn ajoute des graphiques statistiques élégants, branchés directement sur les DataFrames pandas. |

## Ce qu'il faut retenir

Un graphique n'est pas une décoration, c'est un diagnostic. Avant d'entraîner
quoi que ce soit (partie IV), regardez vos données : dix minutes de dessin
évitent souvent trois heures de débogage de modèle.
