"""Tests du vérificateur d'installation du chapitre 3."""

import verifier_installation as vi


def test_version_recente_acceptee():
    assert vi.version_python_ok((3, 12)) is True


def test_version_ancienne_refusee():
    assert vi.version_python_ok((3, 8)) is False


def test_version_pile_a_la_limite_acceptee():
    # Cas limite : 3.10 est le minimum, il doit passer.
    assert vi.version_python_ok((3, 10)) is True


def test_pip_est_importable():
    # pip est livré avec tout Python installé normalement.
    assert vi.pip_disponible() is True


def test_python_est_dans_le_path():
    assert vi.python_dans_le_path() is True


def test_diagnostic_renvoie_quatre_verifications():
    resultats = vi.diagnostic()
    assert len(resultats) == 4
    for reussi, message in resultats:
        assert isinstance(reussi, bool)
        assert isinstance(message, str) and message
