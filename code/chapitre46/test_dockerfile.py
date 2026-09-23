"""Tests du chapitre 46 : vérifier l'emballage sans lancer Docker.

Construire une image prend plusieurs minutes et exige un démon Docker : ce
n'est pas un test unitaire. En revanche, les erreurs les plus fréquentes d'un
Dockerfile se lisent dans le fichier — ordre des couches, port exposé, secrets
copiés par mégarde — et cela, on peut le vérifier en quelques millisecondes.

Un test d'intégration qui construit vraiment l'image est fourni à la fin,
marqué `docker` et ignoré par défaut.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

DOSSIER = Path(__file__).parent
DOCKERFILE = DOSSIER / "Dockerfile"
DOCKERIGNORE = DOSSIER / ".dockerignore"


@pytest.fixture(scope="module")
def lignes():
    return [
        ligne.strip()
        for ligne in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if ligne.strip() and not ligne.strip().startswith("#")
    ]


# ------------------------------------------------------- le Dockerfile
def test_le_dockerfile_existe():
    assert DOCKERFILE.exists()


def test_il_part_dune_image_de_base_python(lignes):
    assert lignes[0].startswith("FROM python:")


def test_limage_de_base_est_epinglee(lignes):
    """`python:3.11-slim` et non `python:latest` : sinon l'image change sous
    vos pieds au prochain build, et « ça marchait la semaine dernière »."""
    base = lignes[0].split()[1]
    assert ":" in base
    assert not base.endswith(":latest")


def test_limage_de_base_est_legere(lignes):
    """`-slim` divise la taille par cinq environ."""
    assert "slim" in lignes[0] or "alpine" in lignes[0]


def test_un_dossier_de_travail_est_defini(lignes):
    assert any(ligne.startswith("WORKDIR") for ligne in lignes)


def test_les_dependances_sont_copiees_avant_le_code(lignes):
    """L'optimisation la plus rentable du fichier.

    Docker met chaque instruction en cache. Si `COPY . .` venait avant le
    `pip install`, la moindre modification d'une ligne de code invaliderait
    le cache et relancerait l'installation complète des dépendances.
    """
    copie_requirements = next(
        i for i, ligne in enumerate(lignes) if ligne.startswith("COPY requirements")
    )
    installation = next(
        i for i, ligne in enumerate(lignes) if ligne.startswith("RUN pip install")
    )
    copie_code = next(
        i for i, ligne in enumerate(lignes) if ligne.startswith("COPY . ")
    )
    assert copie_requirements < installation < copie_code


def test_pip_nutilise_pas_de_cache(lignes):
    """`--no-cache-dir` : le cache de pip ne sert à rien dans une image et
    pèse plusieurs dizaines de mégaoctets."""
    installation = next(ligne for ligne in lignes if ligne.startswith("RUN pip install"))
    assert "--no-cache-dir" in installation


def test_le_serveur_ecoute_sur_toutes_les_interfaces(lignes):
    """`--host 0.0.0.0` et non `127.0.0.1`.

    Le piège classique : avec 127.0.0.1, le serveur n'écoute que l'intérieur
    du conteneur. `docker run -p 8000:8000` semble marcher, mais rien ne
    répond, et aucune erreur ne s'affiche.
    """
    commande = " ".join(lignes)
    assert "0.0.0.0" in commande


def test_le_conteneur_lance_bien_lapi(lignes):
    assert any("uvicorn" in ligne and "mon_api:app" in ligne for ligne in lignes)


def test_aucun_secret_nest_ecrit_dans_limage(lignes):
    """Question 4 du quiz : une image se télécharge et s'inspecte couche par
    couche. Une clé copiée dedans est une clé publiée."""
    contenu = " ".join(lignes).lower()
    for motif in ("api_key=", "password=", "secret=", "token="):
        assert motif not in contenu


# ------------------------------------------------------ le .dockerignore
def test_le_dockerignore_existe():
    """Le défi de l'exercice."""
    assert DOCKERIGNORE.exists()


@pytest.mark.parametrize("entree", ["__pycache__", ".git", ".env", "venv"])
def test_le_dockerignore_exclut_lessentiel(entree):
    contenu = DOCKERIGNORE.read_text(encoding="utf-8")
    assert entree in contenu


def test_le_dockerignore_exclut_les_donnees_brutes():
    assert "data/" in DOCKERIGNORE.read_text(encoding="utf-8")


def test_lenvironnement_virtuel_est_exclu():
    """Copier un venv dans une image est une double faute : il pèse des
    centaines de Mo, et il contient des binaires compilés pour VOTRE système,
    pas pour celui de l'image."""
    assert "venv" in DOCKERIGNORE.read_text(encoding="utf-8")


# ---------------------------------------------------- les fichiers requis
@pytest.mark.parametrize(
    "fichier", ["requirements.txt", "mon_api.py", "modele_manchots.joblib"]
)
def test_les_fichiers_necessaires_sont_presents(fichier):
    assert (DOSSIER / fichier).exists()


def test_requirements_liste_les_dependances_de_lapi():
    contenu = (DOSSIER / "requirements.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    # Les fichiers générés par `pip freeze` sous Windows peuvent être en
    # UTF-16 : on retire les octets nuls avant de chercher.
    contenu = contenu.replace("\x00", "").lower()
    for paquet in ("fastapi", "uvicorn", "scikit-learn", "joblib", "pandas"):
        assert paquet in contenu


# ------------------------------------------- le vrai build (lent, optionnel)
@pytest.mark.docker
def test_limage_se_construit_vraiment():
    """Lancez-le explicitement : pytest -m docker

    Comptez plusieurs minutes la première fois.
    """
    if shutil.which("docker") is None:
        pytest.skip("Docker n'est pas installé")

    resultat = subprocess.run(
        ["docker", "build", "-t", "api-manchots:test", "."],
        cwd=DOSSIER,
        capture_output=True,
        text=True,
    )
    assert resultat.returncode == 0, resultat.stderr[-2000:]
