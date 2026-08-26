"""Chapitre 53 — le découpage en morceaux, sans dépendance.

`exercice_rag.py` utilise LangChain et un vrai modèle : c'est la version
complète, mais elle exige plusieurs gigaoctets de téléchargement. Le découpage,
lui, est de la manipulation de chaînes de caractères — donc testable en
quelques millisecondes, et c'est là que se jouent la plupart des mauvaises
réponses d'un RAG.

Usage :
    python decoupage.py
"""

from __future__ import annotations

import re
from pathlib import Path

SEPARATEURS = ["\n\n", "\n", ". ", " ", ""]


def decouper(texte: str, taille: int = 500, chevauchement: int = 50) -> list[str]:
    """Découpe un texte en morceaux d'environ `taille` caractères.

    Deux principes, repris du `RecursiveCharacterTextSplitter` de LangChain :

    * on coupe **de préférence aux frontières naturelles** (paragraphe, phrase,
      mot) plutôt qu'au milieu d'un mot ;
    * on fait **chevaucher** les morceaux : une phrase coupée en deux reste
      lisible dans au moins l'un des deux morceaux.
    """
    if chevauchement >= taille:
        raise ValueError("Le chevauchement doit être plus petit que la taille.")

    texte = texte.strip()
    if not texte:
        return []
    if len(texte) <= taille:
        return [texte]

    morceaux: list[str] = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))

        if fin < len(texte):
            fin = _meilleure_coupure(texte, debut, fin)

        morceau = texte[debut:fin].strip()
        if morceau:
            morceaux.append(morceau)

        if fin >= len(texte):
            break
        debut = max(fin - chevauchement, debut + 1)

    return morceaux


def _meilleure_coupure(texte: str, debut: int, fin: int) -> int:
    """Recule jusqu'à la frontière naturelle la plus proche avant `fin`."""
    fenetre = texte[debut:fin]
    for separateur in SEPARATEURS:
        if not separateur:
            break
        position = fenetre.rfind(separateur)
        # On refuse de reculer trop loin : mieux vaut couper net qu'obtenir
        # un morceau deux fois trop petit.
        if position > len(fenetre) // 2:
            return debut + position + len(separateur)
    return fin


def statistiques(morceaux: list[str]) -> dict[str, float]:
    """De quoi répondre à la question « quel effet a chunk_size ? »."""
    if not morceaux:
        return {"nombre": 0, "taille_moyenne": 0.0, "taille_max": 0, "taille_min": 0}
    tailles = [len(m) for m in morceaux]
    return {
        "nombre": len(morceaux),
        "taille_moyenne": round(sum(tailles) / len(tailles), 1),
        "taille_max": max(tailles),
        "taille_min": min(tailles),
    }


def mots_cles(texte: str) -> set[str]:
    """Les mots significatifs d'un texte, en minuscules et sans ponctuation.

    Sert de « recherche par mots-clés », pour montrer par contraste ce que la
    recherche par embeddings apporte : deux phrases de sens proche mais sans
    mot commun ont une intersection **vide** ici, alors qu'un embedding les
    rapprocherait (question 3 de l'exercice).
    """
    vides = {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "a", "à",
        "est", "sont", "pour", "dans", "en", "sur", "par", "au", "aux", "ce",
        "que", "qui", "se", "sa", "son", "ses", "il", "elle", "on", "avec",
    }
    mots = re.findall(r"[a-zàâäéèêëîïôöùûüç]+", texte.lower())
    return {mot for mot in mots if mot not in vides and len(mot) > 2}


def recouvrement_mots(phrase_a: str, phrase_b: str) -> int:
    """Combien de mots significatifs les deux phrases partagent-elles ?"""
    return len(mots_cles(phrase_a) & mots_cles(phrase_b))


def main() -> None:
    document = Path(__file__).parent / "mon_document.txt"
    texte = (
        document.read_text(encoding="utf-8")
        if document.exists()
        else "Texte d'exemple. " * 100
    )

    print("--- effet de chunk_size (question 2) ---")
    for taille in (100, 300, 500, 1000):
        stats = statistiques(decouper(texte, taille=taille, chevauchement=taille // 10))
        print(
            f"  taille={taille:<5} -> {stats['nombre']:>3} morceaux, "
            f"moyenne {stats['taille_moyenne']:>6} caracteres"
        )

    print("\n  Petits morceaux : recherche precise, mais contexte tronque.")
    print("  Gros morceaux   : contexte complet, mais du bruit autour de la reponse.")

    print("\n--- question 3 : deux phrases proches sans mot commun ---")
    a = "Quelle est la duree de vie de la batterie ?"
    b = "Combien de temps tient l'accumulateur avant recharge ?"
    print(f"  A : {a}")
    print(f"  B : {b}")
    print(f"  mots significatifs partages : {recouvrement_mots(a, b)}")
    print(
        "  Une recherche par mots-cles ne trouve rien.\n"
        "  Un embedding, lui, place ces deux phrases cote a cote :\n"
        "  c'est exactement ce que le RAG apporte."
    )


if __name__ == "__main__":
    main()
