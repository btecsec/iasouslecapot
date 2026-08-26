# Chapitre 49 — Automatiser le cycle de vie avec le CI/CD

## L'énoncé

1. Écrivez deux tests simples pour votre API (une prédiction valide ; une
   entrée mal formée rejetée).
2. Créez le fichier de chaîne CI/CD dans votre dépôt.
3. Poussez une modification et observez la chaîne. Que se passe-t-il si un
   test échoue ?
4. Dessinez la boucle du MLOps et placez-y chaque chapitre de la partie V.
5. **Défi** : décrivez comment un modèle « en dérive » revient automatiquement
   en production grâce au CI/CD.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `api.py` | Une API autonome (le modèle s'entraîne au premier démarrage). |
| `test_api_ci.py` | Les deux tests demandés, plus le filet de sécurité. |
| `exemple_ci.yml` | La chaîne GitHub Actions, commentée ligne à ligne. |
| `donnees.py` | Le chargement des manchots, avec repli hors ligne. |

```bash
pip install -r requirements.txt
pytest -q
uvicorn api:app --reload
```

## Corrigé

### 1. Les deux tests

```python
def test_une_prediction_valide_aboutit():
    reponse = client.post("/predire", json=MANCHOT)
    assert reponse.status_code == 200
    assert reponse.json()["espece"] in {"Adelie", "Chinstrap", "Gentoo"}


def test_une_entree_mal_formee_est_rejetee():
    reponse = client.post("/predire", json=MANCHOT | {"masse": "beaucoup"})
    assert reponse.status_code == 422
```

Ces deux-là couvrent déjà l'essentiel : **le chemin nominal** et **le chemin
d'erreur**. Le second est le plus important des deux — un service qui plante
sur une entrée bizarre est un service qui tombe.

Les tests suivants du fichier sont ceux qu'on ajoute après avoir été surpris
en production :

| Test | Le bug qu'il empêche de revenir |
|---|---|
| masse négative refusée | des entrées absurdes acceptées par le modèle |
| réponse sérialisable en JSON | `numpy.str_` sorti de scikit-learn casse la sérialisation |
| ordre des champs indifférent | un client réordonne son JSON, les prédictions changent |
| **non-régression du modèle** | un réentraînement dégrade la performance sans que personne ne le voie |

Ce dernier mérite qu'on s'y arrête :

```python
def test_le_modele_ne_regresse_pas_sous_le_seuil():
    scores = cross_val_score(api.modele, X, y, cv=3)
    assert scores.mean() > 0.90, f"performance tombee a {scores.mean():.3f}"
```

C'est le test spécifique aux projets d'IA (chapitre 9). Un test classique
vérifie que le code fait ce qu'on attend ; celui-ci vérifie que **le modèle
reste assez bon**. Sans lui, une chaîne CI/CD toute verte peut déployer un
modèle devenu médiocre.

### 2-3. La chaîne, et ce qui se passe en cas d'échec

Le fichier `exemple_ci.yml` contient deux jobs, et une seule ligne fait tout le
travail :

```yaml
deploiement:
  needs: tests        # ← le déploiement n'existe que si les tests passent
```

**Que se passe-t-il si un test échoue ?**

1. Le job `tests` s'arrête en rouge.
2. Le job `deploiement` **ne démarre jamais** — pas « échoue », *ne démarre
   pas*.
3. GitHub notifie l'auteur du commit, et affiche une croix rouge sur la pull
   request. Avec la protection de branche activée, la fusion est bloquée.
4. **La production continue de tourner avec la version précédente.**

C'est tout l'intérêt : le code fautif n'atteint jamais les utilisateurs. Une
chaîne CI/CD n'est pas un outil de vitesse, c'est un **cliquet** — elle empêche
de revenir en arrière sans le savoir.

Quelques choix du fichier qui méritent une phrase :

| Ligne | Pourquoi |
|---|---|
| `matrix: python-version: ["3.11", "3.12"]` | savoir si un bug touche une version ou toutes |
| `fail-fast: false` | voir *tous* les échecs, pas seulement le premier |
| `cache: pip` | quelques minutes gagnées à chaque exécution |
| `ruff check .` | le style avant les tests : c'est plus rapide à corriger |
| `-m "not integration"` | écarter les tests lents ou payants (chapitre 9) |
| `if: github.ref == 'refs/heads/master'` | ne déployer que la branche principale, jamais une PR |
| `docker build -t api:${{ github.sha }}` | une image par commit : traçable, et un retour arrière = redéployer l'étiquette d'avant |
| `secrets.CLE_DEPLOIEMENT` | les secrets viennent de GitHub, jamais du dépôt (chapitres 46-46) |

### 4. La boucle du MLOps, chapitre par chapitre

```text
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   ↓                                                              │
[développer]  →  [empaqueter]  →  [déployer]  →  [surveiller]  ───┘
 ch. 26-39         ch. 42          ch. 43         ch. 44
    ↑                  ↑               ↑              │
    │              ch. 41 : exposer via une API       │
    │                                                 │
    └──────── ch. 45 : le CI/CD automatise ───────────┘
              (tester, construire, redéployer)
```

Le chapitre 44 décrit la boucle entière ; le 45 est ce qui la fait **tourner
toute seule**.

### 5. Défi : de la dérive détectée au redéploiement

Cinq phrases, une par étape :

1. **La surveillance déclenche** (chapitre 48) : la moyenne des entrées
   s'écarte de la référence au-delà du seuil, une alerte part vers l'équipe et
   ouvre automatiquement un ticket.
2. **Le réentraînement se lance** : une tâche planifiée reconstruit le modèle
   sur des données fraîches — les mêmes six étapes qu'au chapitre 43, mais
   exécutées par un script, pas par un humain.
3. **La chaîne CI valide** : tests de l'API, tests de données (schéma, plages),
   et surtout **test de non-régression** — le nouveau modèle doit faire au
   moins aussi bien que celui en production sur un jeu de référence figé.
4. **Le CD construit et déploie** : nouvelle image étiquetée par le SHA du
   commit, poussée dans le registre, service mis à jour progressivement
   (*canary* : 5 % du trafic d'abord).
5. **La surveillance reprend** sur la nouvelle version, et le retour arrière
   reste à une commande — redéployer l'étiquette précédente.

Le garde-fou indispensable : **un humain valide l'étape 4** tant que la
confiance n'est pas établie. Un réentraînement automatique sur des données
elles-mêmes dérivées peut apprendre la panne au lieu de la corriger.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le travail manuel est lent, ennuyeux et source d'erreurs — et il est toujours sauté un vendredi soir. |
| 2 | **b** | CI = tester et vérifier automatiquement le code à chaque changement. |
| 3 | **b** | La chaîne s'arrête et prévient ; le code fautif n'atteint pas la production. |
| 4 | **b** | La surveillance détecte une dérive, on ré-entraîne, et le CI/CD redéploie. |

## Ce qu'il faut retenir

Le CI/CD transforme une suite de gestes manuels en un cliquet automatique :
tests verts → image construite → service mis à jour. Ajoutez-y le test de
non-régression du modèle, sans lequel une chaîne toute verte peut déployer un
modèle devenu mauvais.
