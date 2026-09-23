"""Tests des données de fine-tuning du chapitre 52.

Un fine-tuning rate presque toujours à cause des **données**, pas du code :
un JSONL mal formé, une ligne vide, un exemple dupliqué entre train et test.
Ces vérifications coûtent 50 ms et évitent des heures de GPU perdues.

Elles n'exigent ni torch, ni transformers, ni GPU.
"""

import json
from pathlib import Path

import pytest

DOSSIER = Path(__file__).parent
TRAIN = DOSSIER / "data" / "train.jsonl"
TEST = DOSSIER / "data" / "test.jsonl"


def lire(chemin: Path) -> list[dict]:
    """Lit un JSONL : une ligne = un objet JSON. Pas de virgule finale."""
    return [
        json.loads(ligne)
        for ligne in chemin.read_text(encoding="utf-8").splitlines()
        if ligne.strip()
    ]


@pytest.fixture(scope="module")
def train():
    return lire(TRAIN)


@pytest.fixture(scope="module")
def test():
    return lire(TEST)


# ------------------------------------------------------------- le format
@pytest.mark.parametrize("chemin", [TRAIN, TEST])
def test_les_fichiers_existent(chemin):
    assert chemin.exists()


@pytest.mark.parametrize("chemin", [TRAIN, TEST])
def test_chaque_ligne_est_un_json_valide(chemin):
    """L'erreur numéro un : un JSONL rédigé comme un tableau JSON."""
    for numero, ligne in enumerate(
        chemin.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if ligne.strip():
            try:
                json.loads(ligne)
            except json.JSONDecodeError as erreur:
                pytest.fail(f"{chemin.name} ligne {numero} : {erreur}")


def test_chaque_exemple_a_une_instruction_et_une_sortie(train, test):
    for exemple in train + test:
        assert set(exemple) == {"instruction", "output"}


def test_aucun_champ_nest_vide(train, test):
    for exemple in train + test:
        assert exemple["instruction"].strip()
        assert exemple["output"].strip()


# ------------------------------------------------------------ la qualité
def test_il_y_a_assez_dexemples(train):
    """Cinq exemples suffisent à démontrer, pas à obtenir un style stable.
    Comptez plutôt quelques centaines pour un vrai projet."""
    assert len(train) >= 5


def test_aucun_doublon_dans_lentrainement(train):
    instructions = [exemple["instruction"] for exemple in train]
    assert len(instructions) == len(set(instructions))


def test_aucune_fuite_entre_entrainement_et_test(train, test):
    """La fuite du chapitre 33, version LLM : un exemple de test déjà vu à
    l'entraînement donne une évaluation flatteuse et fausse."""
    instructions_train = {exemple["instruction"] for exemple in train}
    instructions_test = {exemple["instruction"] for exemple in test}
    assert instructions_train.isdisjoint(instructions_test)


def test_les_reponses_sont_substantielles(train):
    """Des sorties d'un mot n'apprennent aucun style au modèle."""
    for exemple in train:
        assert len(exemple["output"].split()) >= 5


def test_le_style_est_coherent(train):
    """Le fine-tuning apprend un comportement : encore faut-il que tous les
    exemples le partagent. Ici, un registre soutenu et le vouvoiement."""
    fautifs = [
        exemple["instruction"]
        for exemple in train
        if " tu " in f" {exemple['output'].lower()} "
    ]
    assert fautifs == [], f"tutoiement dans : {fautifs}"


def test_les_exemples_sont_encodes_en_utf8_lisible(train):
    """Un « Ã© » à la place d'un « é » se retrouverait tel quel dans les poids."""
    texte = " ".join(e["instruction"] + e["output"] for e in train)
    for motif in ("Ã©", "Ã¨", "â€™"):
        assert motif not in texte


# ------------------------------------------------ l'adaptateur LoRA produit
def test_ladaptateur_lora_est_leger_par_rapport_a_un_modele_complet():
    """L'argument de LoRA, vérifié sur le fichier réellement produit :
    quelques mégaoctets au lieu de plusieurs gigaoctets."""
    adaptateur = DOSSIER / "concierge_model" / "adapter_model.safetensors"
    if not adaptateur.exists():
        pytest.skip("adaptateur absent : lancez finetune_lora.py")
    assert adaptateur.stat().st_size < 100 * 1024 * 1024  # < 100 Mo


def test_la_configuration_de_ladaptateur_est_lisible():
    config = DOSSIER / "concierge_model" / "adapter_config.json"
    if not config.exists():
        pytest.skip("adaptateur absent : lancez finetune_lora.py")
    contenu = json.loads(config.read_text(encoding="utf-8"))
    # `r` est le rang de LoRA : le nombre de dimensions du raccourci appris.
    assert "r" in contenu
    assert contenu["r"] > 0
