# Solution de l'exercice — Chapitre 9 : Tester son code, et tester son IA

> Tout le code est dans ce dossier, et il tourne :
> `pytest -q -m "not integration"` → **23 passés, 1 ignoré, 1 xfailed**.

---

## 1. La fonction `normaliser`

```python
def normaliser(valeurs: list[float]) -> list[float]:
    if not valeurs:
        raise ValueError("La liste ne peut pas être vide")

    minimum = min(valeurs)
    etendue = max(valeurs) - minimum
    if etendue == 0:
        return [0.0] * len(valeurs)

    return [(v - minimum) / etendue for v in valeurs]
```

Trois décisions, et c'est justement pour cela que l'exercice existe :

- **Liste vide → `ValueError`**, comme `moyenne`. La cohérence à l'intérieur d'un
  module compte autant que la justesse : un appelant ne devrait pas avoir à
  deviner quelle fonction lève quoi.
- **Le résultat est une nouvelle liste.** Une fonction qui modifie la liste
  qu'on lui passe est une source de bugs redoutable, parce que le dégât apparaît
  très loin de sa cause.
- **Étendue nulle → à décider** : c'est le point 2.

## 2. Les trois tests demandés

```python
def test_normaliser_cas_normal():
    assert normaliser([0, 5, 10]) == [0.0, 0.5, 1.0]

def test_normaliser_liste_vide_leve_erreur():
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        normaliser([])

def test_normaliser_valeurs_identiques():
    assert normaliser([7, 7, 7]) == [0.0, 0.0, 0.0]
```

### Le cas limite : que faire quand toutes les valeurs sont identiques ?

`max - min` vaut zéro, donc la division est impossible. Il n'y a pas de réponse
mathématiquement « juste ». Trois options défendables :

| Option | Pour | Contre |
|---|---|---|
| Lever une `ValueError` | Force l'appelant à traiter le cas | Casse un pipeline pour une colonne constante, ce qui arrive souvent |
| Renvoyer des `0.0` | Ne casse rien, et une colonne constante n'apporte de toute façon aucune information | Masque discrètement un problème de données |
| Renvoyer des `0.5` | « Le milieu » paraît neutre | Personne d'autre ne fait ça ; effet de surprise garanti |

**Choix retenu : des zéros**, parce que c'est ce que fait le `MinMaxScaler` de
scikit-learn, que le lecteur croisera au chapitre 30. S'aligner sur l'outil
standard évite une mauvaise surprise le jour du basculement.

Ce qui compte n'est pas l'option choisie, mais qu'elle soit **explicite,
documentée et testée**. Un comportement non testé n'est pas un comportement :
c'est un accident qui n'a pas encore eu lieu.

### Deux tests en plus, qui valent le détour

```python
def test_normaliser_ne_modifie_pas_l_entree():
    original = [0, 5, 10]
    copie = list(original)
    normaliser(original)
    assert original == copie

@pytest.mark.parametrize("valeurs,attendu", [
    ([0, 5, 10], [0.0, 0.5, 1.0]),
    ([10, 5, 0], [1.0, 0.5, 0.0]),      # ordre inversé
    ([-10, 0, 10], [0.0, 0.5, 1.0]),    # valeurs négatives
    ([1, 2], [0.0, 1.0]),
    ([42], [0.0]),                      # un seul élément
])
def test_normaliser_plusieurs_cas(valeurs, attendu):
    assert normaliser(valeurs) == attendu
```

`parametrize` compte comme **cinq** tests : si le cas négatif casse, le rapport
nomme précisément ce cas-là plutôt qu'un vague « le test des valeurs a échoué ».

## 3. Lire un rapport d'échec

Le fichier contient un test volontairement faux, neutralisé par `xfail` pour que
la suite reste verte. Retirez le décorateur et lancez `pytest -q` :

```text
E       assert [0.0, 0.5, 1.0] == [0.0, 0.4, 1.0]
E         At index 1 diff: 0.5 != 0.4
tests/test_calculs.py:79: AssertionError
```

pytest donne la ligne, la valeur obtenue, la valeur attendue **et l'index exact**
de la différence. C'est précisément ce qu'on perd en écrivant `assert
resultat == attendu, "erreur"` : le message personnalisé remplace ce rapport.
Laissez pytest parler.

## 4. La fixture et le test de schéma

```python
# tests/conftest.py — pytest le découvre tout seul
@pytest.fixture
def donnees_client():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({
        "age": [25, 40, 33],
        "revenu": [1800.0, 3200.0, 2500.0],
        "label": [0, 1, 1],
    })
```

```python
# tests/test_donnees.py
def valider(df) -> None:
    assert COLONNES_ATTENDUES.issubset(df.columns), "colonne manquante"
    assert df["age"].between(0, 120).all(), "âge hors plage"
    assert df["label"].isin([0, 1]).all(), "label non binaire"
    assert df.isnull().sum().sum() == 0, "valeurs manquantes"

def test_schema_donnees_propres(donnees_client):
    valider(donnees_client)

def test_schema_detecte_les_donnees_abimees(donnees_client_abimees):
    with pytest.raises(AssertionError):
        valider(donnees_client_abimees)
```

Trois choses à retenir ici :

- **La fixture est rejouée pour chaque test.** Un test qui modifie le DataFrame
  ne peut pas polluer le suivant.
- **La validation vit dans une fonction**, pas dans le test. On peut alors la
  réutiliser en production, au moment où les données arrivent.
- **Le second test vérifie que le garde-fou se déclenche.** Un test de validation
  qui ne sait pas échouer ne prouve rien du tout — c'est l'erreur la plus
  fréquente sur ce sujet.

**Le piège rencontré en écrivant ce corrigé :** un `pytest.importorskip("pandas")`
placé en haut de `conftest.py` fait échouer la collecte de **toute** la suite si
pandas manque. Placé à l'intérieur des fixtures, seuls les tests de données sont
ignorés. C'est exactement ce que montre la sortie : `1 skipped`, et les 23 autres
tests passent quand même.

## 5. Le marqueur `integration`

```python
# tests/test_integration.py
pytestmark = pytest.mark.integration      # marque TOUT le fichier
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "integration: test lent, appel réseau/LLM/base réel",
]
```

Vérification :

```bash
pytest -m "not integration" -q   # → « 2 deselected » : le filtrage marche
pytest -m integration -q         # → seuls les tests lents
```

Déclarer le marqueur dans `pyproject.toml` n'est pas cosmétique : sans cela,
pytest émet un avertissement, et une faute de frappe (`@pytest.mark.integraton`)
passerait inaperçue en désactivant silencieusement le test.

Notez aussi le `skipif` sur `OPENAI_API_KEY` : une clé absente n'est pas un bug.
Un test **ignoré** est honnête ; un test **rouge** pour une mauvaise raison finit
par être ignoré par l'équipe, et c'est bien pire.

## Bonus — la couverture

```bash
pip install pytest-cov
pytest -m "not integration" --cov=. --cov-report=term-missing
```

La colonne `Missing` liste les lignes jamais exécutées. Utilisez-la comme une
**liste de suspects**, pas comme une note : les lignes non couvertes de ce projet
sont celles de `ClientReel.generer`, qui appellerait la vraie API — et c'est
exactement ce que l'on veut ne jamais exécuter en test.

Ne visez pas 100 %. Viser 80 % sur la partie déterministe et l'assumer vaut mieux
que 100 % obtenus avec des tests qui ne vérifient rien.

---

## En prime : ce que le dossier contient au-delà de l'énoncé

`agent.py` et `tests/test_llm.py` mettent en pratique la partie LLM du chapitre,
sans jamais appeler un vrai modèle :

- l'interface `LLMClient` qui isole l'appel, et son implémentation réelle jamais
  exécutée en test ;
- `parser_sortie` testée sur six formes de réponses : JSON pur, JSON précédé de
  bavardage, JSON dans un bloc ```` ```json ````, JSON tronqué, réponse vide,
  réponse sans aucun JSON ;
- l'agent testé sur ses trois issues : outil correctement déclenché, outil
  inventé par le modèle, réponse incompréhensible.

C'est le cœur du message du chapitre : le modèle est imprévisible, mais **tout ce
qui l'entoure est du Python ordinaire**, et se teste en quelques millisecondes
pour zéro euro.
