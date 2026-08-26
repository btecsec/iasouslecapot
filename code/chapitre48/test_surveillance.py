"""Tests du corrigé du chapitre 48."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import surveillance as sv  # noqa: E402


@pytest.fixture(scope="module")
def reference():
    generateur = np.random.default_rng(42)
    entrainement = pd.DataFrame({"masse": generateur.normal(4200, 800, size=1000)})
    return sv.calculer_reference(entrainement, "masse")


def lot(moyenne: float, n: int = 200, graine: int = 0) -> pd.DataFrame:
    generateur = np.random.default_rng(graine)
    return pd.DataFrame({"masse": generateur.normal(moyenne, 800, size=n)})


# --------------------------------------------------------------- question 2
def test_la_reference_retrouve_la_moyenne_dentrainement(reference):
    assert reference.moyenne == pytest.approx(4200, abs=100)
    assert reference.ecart_type == pytest.approx(800, abs=100)


def test_la_reference_est_figee(reference):
    """Elle décrit l'entraînement, pas la production — sinon elle suivrait la
    dérive et ne la signalerait jamais."""
    with pytest.raises(Exception):
        reference.moyenne = 0  # dataclass(frozen=True)


def test_un_ecart_type_nul_ne_fait_pas_diviser_par_zero():
    plate = sv.Reference("masse", moyenne=10.0, ecart_type=0.0)
    assert plate.zscore(50.0) == 0.0
    assert plate.zscore_moyenne(50.0, 10) == 0.0


# --------------------------------------------------------------- question 3
def test_des_donnees_normales_ne_declenchent_rien(reference):
    assert sv.verifier_derive(lot(4200), reference).declenchee is False


def test_une_derive_franche_declenche_lalerte(reference):
    """Des jeunes manchots : 1 300 g de moins en moyenne."""
    alerte = sv.verifier_derive(lot(2900), reference)
    assert alerte.declenchee is True
    assert alerte.zscore < 0


def test_une_derive_vers_le_haut_declenche_aussi(reference):
    assert sv.verifier_derive(lot(5500), reference).declenchee is True


def test_un_capteur_en_panne_est_detecte(reference):
    """Tout à zéro : le cas le plus fréquent en vrai, et le plus facile à voir."""
    panne = pd.DataFrame({"masse": np.zeros(200)})
    assert sv.verifier_derive(panne, reference).declenchee is True


def test_le_message_dalerte_est_explicite(reference):
    message = sv.verifier_derive(lot(2900), reference).message
    assert "masse" in message
    assert "sigma" in message


def test_un_seuil_plus_severe_declenche_plus_souvent(reference):
    """Le seuil est un curseur : plus il est bas, plus on alerte (et plus on
    risque de crier au loup)."""
    donnees = lot(4400)
    tolerant = sv.verifier_derive(donnees, reference, seuil_sigma=10)
    severe = sv.verifier_derive(donnees, reference, seuil_sigma=1)
    assert severe.declenchee and not tolerant.declenchee


# ------------------------------------- la taille de l'échantillon compte
def test_la_formule_naive_rate_une_derive_massive(reference):
    """1 300 g d'écart sur 200 manchots : invisible pour la formule naïve."""
    donnees = lot(2900)
    naive = sv.verifier_derive(donnees, reference, tenir_compte_de_la_taille=False)
    correcte = sv.verifier_derive(donnees, reference)
    assert naive.declenchee is False
    assert correcte.declenchee is True


def test_une_moyenne_est_plus_stable_quun_individu(reference):
    """C'est tout l'argument : l'erreur type vaut sigma / racine(n)."""
    individuel = abs(reference.zscore(4400))
    sur_200 = abs(reference.zscore_moyenne(4400, 200))
    assert sur_200 == pytest.approx(individuel * np.sqrt(200), rel=1e-6)


def test_plus_le_lot_est_grand_plus_la_detection_est_fine(reference):
    petit = abs(sv.verifier_derive(lot(4400, n=10), reference).zscore)
    grand = abs(sv.verifier_derive(lot(4400, n=500), reference).zscore)
    assert grand > petit


# --------------------------------------------------------------- question 1
def test_les_indicateurs_comptent_les_requetes():
    indicateurs = sv.Indicateurs()
    for _ in range(10):
        indicateurs.enregistrer(50.0)
    assert indicateurs.requetes == 10


def test_le_taux_derreur():
    indicateurs = sv.Indicateurs()
    for _ in range(8):
        indicateurs.enregistrer(50.0)
    for _ in range(2):
        indicateurs.enregistrer(50.0, en_erreur=True)
    assert indicateurs.taux_erreur == pytest.approx(0.2)


def test_le_p95_revele_ce_que_la_moyenne_cache():
    """90 requêtes à 50 ms, 10 à 900 ms : la moyenne rassure, le p95 alerte."""
    indicateurs = sv.Indicateurs()
    for _ in range(90):
        indicateurs.enregistrer(50.0)
    for _ in range(10):
        indicateurs.enregistrer(900.0)
    assert indicateurs.latence_moyenne < 200
    assert indicateurs.latence_p95 > 500


def test_les_indicateurs_vides_ne_plantent_pas():
    vide = sv.Indicateurs()
    assert vide.latence_moyenne == 0.0
    assert vide.latence_p95 == 0.0
    assert vide.taux_erreur == 0.0


def test_le_resume_contient_les_quatre_chiffres():
    indicateurs = sv.Indicateurs()
    indicateurs.enregistrer(42.0)
    assert set(indicateurs.resume()) == {
        "requetes",
        "latence_moyenne_ms",
        "latence_p95_ms",
        "taux_erreur",
    }
