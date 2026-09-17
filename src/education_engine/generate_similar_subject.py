from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import re
import sys
from typing import Any

from openai import OpenAI
from mistralai.client import Mistral
from supabase import create_client

from ai_providers import (
    assert_provider_available,
    choose_provider,
    provider_status,
)


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GENERATION_MODEL = "gpt-4.1-mini"

TABLE_GENERATED_SUBJECTS = "generated_subjects"

EVALUATION_TYPES = [
    "Devoir",
    "Composition",
    "Interrogation",
    "BEPC",
    "BEPC Blanc",
    "BAC",
    "BAC Blanc",
    "Corrigé",
    "Format d’épreuve",
    "Format d'épreuve",
]

PROGRAMME_TYPES = [
    "Programme",
]

MAX_CONTEXT_CHUNKS_EVALUATION = 8
MAX_CONTEXT_CHUNKS_PROGRAMME = 5
MAX_CHARS_PER_CHUNK = 1200

MIN_ACCEPTED_QUALITY_SCORE = 85
MAX_QUALITY_REPAIR_ATTEMPTS = 2


def configure_utf8_output() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(f"Variable d'environnement absente : {name}")

    return value


def clean_text(value: str | None) -> str:
    if not value:
        return ""

    value = str(value)
    value = value.replace("\x00", " ")
    value = re.sub(r"[\ud800-\udfff]", " ", value)
    value = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)

    return value.strip()


def slugify(value: str) -> str:
    value = clean_text(value).lower()

    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "’": "_",
        "'": "_",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    return value or "sujet"


def normalize_for_match(value: str | None) -> str:
    value = clean_text(value).lower()

    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "’": "'",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_type_document(value: str | None) -> str | None:
    if not value:
        return None

    value = clean_text(value)

    mapping = {
        "devoir": "Devoir",
        "composition": "Composition",
        "interrogation": "Interrogation",
        "bepc": "BEPC",
        "bepc blanc": "BEPC Blanc",
        "bac": "BAC",
        "bac blanc": "BAC Blanc",
        "corrige": "Corrigé",
        "corrigé": "Corrigé",
        "programme": "Programme",
        "format": "Format d’épreuve",
        "format d'epreuve": "Format d’épreuve",
        "format d’épreuve": "Format d’épreuve",
    }

    key = value.lower()
    key = key.replace("é", "e").replace("è", "e").replace("ê", "e")
    key = key.replace("’", "'")

    return mapping.get(key, value)


def is_formal_assessment(type_demande: str) -> bool:
    value = normalize_for_match(type_demande)

    keywords = [
        "devoir",
        "composition",
        "evaluation",
        "interrogation",
        "bepc",
        "bac",
        "examen",
    ]

    return any(keyword in value for keyword in keywords)


def is_school_exam_type(type_demande: str) -> bool:
    value = normalize_for_match(type_demande)

    keywords = [
        "composition",
        "bepc",
        "bac",
        "examen",
        "epreuve",
        "sujet bepc",
        "sujet bac",
    ]

    return any(keyword in value for keyword in keywords)


PHYSICS_FORBIDDEN_PATTERNS = [
    "R = 0",
    "R=0",
    "0 Ω",
    "0 ohm",
    "0 ohms",
    "/ 0",
    "/0",
    "diviser par zéro",
    "division par zéro",
    "I = U / 0",
    "I=U/0",
    "I tend vers l’infini",
    "I tend vers l'infini",
    "courant infini",
    "intensité infinie",
    "intensite infinie",
    "résistance nulle",
    "resistance nulle",
    "résistance est nulle",
    "resistance est nulle",
    "résistance quasiment nulle",
    "resistance quasiment nulle",
    "résistance presque nulle",
    "resistance presque nulle",
]

PHYSICS_REQUIRED_SHORT_CIRCUIT_TERMS = [
    "résistance très faible",
    "resistance tres faible",
    "courant très élevé",
    "courant tres eleve",
    "intensité très élevée",
    "intensite tres elevee",
    "échauffement",
    "echauffement",
    "fusible",
    "disjoncteur",
]


def is_physics_short_circuit_subject(
    matiere: str,
    theme: str,
    generated_text: str = "",
) -> bool:
    matiere_text = normalize_for_match(matiere)
    theme_text = normalize_for_match(theme)
    full_text = normalize_for_match(f"{theme_text}\n{generated_text or ''}")

    return (
        "physique" in matiere_text
        and (
            "court circuit" in full_text
            or "courtcircuit" in full_text
        )
    )


def build_physics_short_circuit_rules(matiere: str, theme: str) -> str:
    if not is_physics_short_circuit_subject(matiere, theme):
        return ""

    return """
Règle spéciale Physique-Chimie — court-circuit :
- Ne jamais écrire que la résistance est exactement nulle.
- Ne jamais écrire "R = 0 Ω", "R = 0", "0 Ω", "I = U / 0", "division par zéro", "courant infini" ou "intensité infinie".
- Ne jamais présenter une division par zéro comme un calcul.
- Écrire plutôt : le chemin de court-circuit possède une résistance très faible mais non nulle.
- Expliquer que l’intensité devient très élevée, ce qui peut provoquer un échauffement, endommager le matériel et déclencher un fusible ou un disjoncteur.
- Pour les calculs, comparer le circuit normal avec la situation de court-circuit qualitativement si la résistance exacte du court-circuit n’est pas donnée.
""".strip()


def validate_physics_short_circuit_quality(
    matiere: str,
    theme: str,
    subject_text: str,
    correction_text: str = "",
) -> tuple[bool, list[str]]:
    """
    Vérifie les formulations dangereuses ou scientifiquement imprécises
    dans les sujets de court-circuit niveau collège.
    """
    combined_text = clean_text(f"{subject_text or ''}\n{correction_text or ''}")
    combined_lower = combined_text.lower()

    if not is_physics_short_circuit_subject(matiere, theme, combined_text):
        return True, []

    errors = []

    for pattern in PHYSICS_FORBIDDEN_PATTERNS:
        if pattern.lower() in combined_lower:
            errors.append(
                f"Formulation interdite en court-circuit : « {pattern} ». "
                "Utiliser plutôt : résistance très faible mais non nulle, courant très élevé, "
                "échauffement et protection par fusible/disjoncteur."
            )

    required_hits = [
        term
        for term in PHYSICS_REQUIRED_SHORT_CIRCUIT_TERMS
        if term.lower() in combined_lower
    ]

    if len(required_hits) < 2:
        errors.append(
            "Le sujet/corrigé de court-circuit doit mentionner clairement au moins deux idées : "
            "résistance très faible, courant très élevé / intensité très élevée, "
            "échauffement, fusible ou disjoncteur."
        )

    return len(errors) == 0, errors


def apply_physics_short_circuit_quality_guard(
    qa_report: dict[str, Any],
    generated_subject: str,
    matiere: str,
    theme: str,
    include_correction: bool,
) -> dict[str, Any]:
    """
    Renforce localement le contrôle qualité OpenAI pour les courts-circuits.
    Ce garde-fou évite qu'un sujet soit accepté si le texte contient des
    formulations scientifiquement imprécises comme I = U / 0 ou R = 0 Ω.
    """
    subject_text, correction_text = split_subject_and_correction(generated_subject)

    if not include_correction:
        correction_text = ""

    physics_ok, physics_errors = validate_physics_short_circuit_quality(
        matiere=matiere,
        theme=theme,
        subject_text=subject_text,
        correction_text=correction_text or "",
    )

    if physics_ok:
        return qa_report

    updated_report = dict(qa_report or {})
    current_problems = list(updated_report.get("problemes") or [])
    current_recommendations = list(updated_report.get("corrections_recommandees") or [])

    current_problems.extend(physics_errors)
    current_recommendations.append(
        "Réécrire la partie court-circuit en parlant de résistance très faible mais non nulle, "
        "d’intensité très élevée, d’échauffement et de protection par fusible/disjoncteur. "
        "Ne pas utiliser de division par zéro."
    )

    updated_report["valide"] = False
    updated_report["decision"] = "REGENERER"
    updated_report["score_sur_100"] = min(int(updated_report.get("score_sur_100", 0) or 0), 70)
    updated_report["problemes"] = current_problems
    updated_report["corrections_recommandees"] = current_recommendations

    return updated_report


def build_question(
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
) -> str:
    correction_part = "avec corrigé" if include_correction else "sans corrigé"

    return (
        f"Génère-moi un {type_demande} de {matiere} "
        f"niveau {niveau}, classe {classe}, sur le thème : {theme}, "
        f"{correction_part}"
    )


def extract_theme(generated_text: str, fallback_theme: str) -> str:
    text = clean_text(generated_text)

    patterns = [
        r"##\s*Thème\s*\n(.+?)(?:\n##|\Z)",
        r"##\s*Theme\s*\n(.+?)(?:\n##|\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

        if match:
            theme = clean_text(match.group(1))
            theme = re.sub(r"^\s*[:\-]\s*", "", theme).strip()

            if len(theme) > 250:
                theme = theme[:250].strip()

            return theme

    return fallback_theme


def split_subject_and_correction(generated_text: str) -> tuple[str, str | None]:
    text = clean_text(generated_text)

    correction_patterns = [
        r"\n##\s*Corrigé\s*\n",
        r"\n##\s*Correction\s*\n",
        r"\n#\s*Corrigé\s*\n",
        r"\n#\s*Correction\s*\n",
    ]

    for pattern in correction_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            subject_text = clean_text(text[:match.start()])
            correction_text = clean_text(text[match.start():])
            return subject_text, correction_text

    return text, None


def create_query_embedding(client: OpenAI, model: str, query: str) -> list[float]:
    response = client.embeddings.create(
        model=model,
        input=query,
        encoding_format="float",
    )

    return response.data[0].embedding


def rpc_match_chunks(
    supabase,
    query_embedding: list[float],
    match_count: int,
    matiere: str | None = None,
    niveau: str | None = None,
    classe: str | None = None,
    type_document: str | None = None,
) -> list[dict[str, Any]]:
    response = (
        supabase
        .rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter_matiere": matiere,
                "filter_niveau": niveau,
                "filter_classe": classe,
                "filter_type_document": type_document,
            },
        )
        .execute()
    )

    return response.data or []


def dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []

    for item in results:
        chunk_id = item.get("id")

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        deduped.append(item)

    return deduped


def sort_by_similarity(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda row: float(row.get("similarity") or 0),
        reverse=True,
    )


def search_by_types(
    supabase,
    query_embedding: list[float],
    matiere: str,
    niveau: str,
    classe: str,
    types: list[str],
    per_type_count: int,
) -> list[dict[str, Any]]:
    all_results = []

    for type_document in types:
        results = rpc_match_chunks(
            supabase=supabase,
            query_embedding=query_embedding,
            match_count=per_type_count,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            type_document=type_document,
        )

        all_results.extend(results)

    return sort_by_similarity(dedupe_results(all_results))


def search_without_type(
    supabase,
    query_embedding: list[float],
    matiere: str,
    niveau: str,
    classe: str,
    match_count: int,
) -> list[dict[str, Any]]:
    return rpc_match_chunks(
        supabase=supabase,
        query_embedding=query_embedding,
        match_count=match_count,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        type_document=None,
    )


def search_fallback_same_subject_level(
    supabase,
    query_embedding: list[float],
    matiere: str,
    niveau: str,
    match_count: int,
) -> list[dict[str, Any]]:
    return rpc_match_chunks(
        supabase=supabase,
        query_embedding=query_embedding,
        match_count=match_count,
        matiere=matiere,
        niveau=niveau,
        classe=None,
        type_document=None,
    )


def format_context_block(title: str, results: list[dict[str, Any]], limit: int) -> str:
    lines = []
    lines.append(f"\n## {title}\n")

    if not results:
        lines.append("Aucune source disponible pour ce groupe.")
        return "\n".join(lines)

    for index, item in enumerate(results[:limit], start=1):
        texte = clean_text(item.get("texte") or "")
        extrait = texte[:MAX_CHARS_PER_CHUNK]

        lines.append(f"\n### Source {index}")
        lines.append(f"Chunk ID    : {item.get('id')}")
        lines.append(f"Document ID : {item.get('document_id')}")
        lines.append(f"Titre       : {item.get('titre')}")
        lines.append(f"Matière     : {item.get('matiere')}")
        lines.append(f"Niveau      : {item.get('niveau')}")
        lines.append(f"Classe      : {item.get('classe')}")
        lines.append(f"Type        : {item.get('type_document')}")
        lines.append(f"Score       : {item.get('similarity')}")
        lines.append("Extrait :")
        lines.append(extrait)

    return "\n".join(lines)


def build_expected_format(
    matiere: str,
    classe: str,
    include_correction: bool,
    type_demande: str,
) -> str:
    formal = is_formal_assessment(type_demande)

    if include_correction:
        if formal:
            return f"""
Format attendu :

# Sujet généré — {matiere} {classe}

## Thème
...

## Compétence visée
...

## Durée indicative
...

## Barème indicatif
...

## Sujet / Évaluation

### Exercice 1 : ...
Énoncé :
...
Questions :
1. ...
2. ...

### Exercice 2 : ...
Énoncé :
...
Questions :
1. ...
2. ...

## Corrigé

### Exercice 1
1. ...
2. ...

### Exercice 2
1. ...
2. ...

## Vérification de cohérence
- Données utilisées :
- Données non utilisées :
- Données ajoutées dans le corrigé : aucune
- Contradictions détectées : aucune

## Remarque pédagogique
...

Sources utilisées :
- citer seulement les titres et types des sources utilisées.
""".strip()

        return f"""
Format attendu :

# Sujet généré — {matiere} {classe}

## Thème
...

## Compétence visée
...

## Sujet / Exercice

Énoncé :
...

Questions :
1. ...
2. ...
3. ...
4. ...

## Corrigé

1. ...
2. ...
3. ...
4. ...

## Vérification de cohérence
- Données utilisées :
- Données non utilisées :
- Données ajoutées dans le corrigé : aucune
- Contradictions détectées : aucune

## Remarque pédagogique
...

Sources utilisées :
- citer seulement les titres et types des sources utilisées.
""".strip()

    if formal:
        return f"""
Format attendu :

# Sujet généré — {matiere} {classe}

## Thème
...

## Compétence visée
...

## Durée indicative
...

## Barème indicatif
...

## Sujet / Évaluation

### Exercice 1 : ...
Énoncé :
...
Questions :
1. ...
2. ...

### Exercice 2 : ...
Énoncé :
...
Questions :
1. ...
2. ...

## Remarque pédagogique
...

Sources utilisées :
- citer seulement les titres et types des sources utilisées.

Important :
- Ne pas afficher de corrigé.
- Ne pas afficher de solution.
- Ne pas afficher de réponse aux questions.
- Ne pas inclure une section intitulée "Vérification de cohérence" si elle révèle des éléments de correction.
""".strip()

    return f"""
Format attendu :

# Sujet généré — {matiere} {classe}

## Thème
...

## Compétence visée
...

## Sujet / Exercice

Énoncé :
...

Questions :
1. ...
2. ...
3. ...
4. ...

## Remarque pédagogique
...

Sources utilisées :
- citer seulement les titres et types des sources utilisées.

Important :
- Ne pas afficher de corrigé.
- Ne pas afficher de solution.
- Ne pas afficher de réponse aux questions.
""".strip()


def build_formal_assessment_rules(type_demande: str, include_correction: bool) -> str:
    if not is_formal_assessment(type_demande):
        return """
Règles de style :
- Employer un ton scolaire, clair et neutre.
- Éviter toute expression familière ou imprécise.
""".strip()

    correction_line = (
        "- Le corrigé peut contenir les calculs détaillés, mais il doit rester séparé du sujet."
        if include_correction
        else "- Le sujet ne doit contenir aucune réponse, aucun indice trop direct et aucun corrigé."
    )

    return f"""
Règles strictes pour devoir / composition / examen :
- Employer uniquement un style académique : "Exercice 1", "Exercice 2", "Situation d’évaluation", "Questions".
- Interdire les formulations familières ou maladroites comme : "Un autre souci", "petit problème", "truc", "on va voir", "voici un autre souci".
- Ne jamais mélanger deux situations sans transition scolaire claire.
- Pour une composition ou un examen, proposer un sujet structuré avec durée indicative et barème indicatif.
- Ne pas construire une contradiction pédagogique.
- Si une question demande de vérifier qu’une situation est proportionnelle et que les données montrent qu’elle ne l’est pas, les questions suivantes ne doivent pas utiliser cette même relation comme si elle était proportionnelle.
- Éviter les phrases du type "en supposant que c’est proportionnel" après avoir fourni des données qui montrent le contraire.
- Les données numériques doivent être cohérentes du début à la fin.
- Chaque exercice doit être indépendant ou clairement lié au précédent.
- Les questions doivent être progressives : compréhension, calcul, justification, application.
- Les unités doivent être cohérentes et précisées.
{correction_line}
""".strip()


def build_generation_prompt(
    question: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
    programme_results: list[dict[str, Any]],
    evaluation_results: list[dict[str, Any]],
    fallback_results: list[dict[str, Any]],
) -> str:
    programme_context = format_context_block(
        title="CONTEXTE PROGRAMME OFFICIEL",
        results=programme_results,
        limit=MAX_CONTEXT_CHUNKS_PROGRAMME,
    )

    evaluation_context = format_context_block(
        title="CONTEXTE SUJETS / MODÈLES D'ÉVALUATION",
        results=evaluation_results,
        limit=MAX_CONTEXT_CHUNKS_EVALUATION,
    )

    fallback_context = format_context_block(
        title="CONTEXTE COMPLÉMENTAIRE",
        results=fallback_results,
        limit=4,
    )

    correction_rule = (
        "- Ajouter un corrigé complet, séparé dans une section ## Corrigé."
        if include_correction
        else "- Ne pas générer de corrigé. Ne pas afficher de solution. Ne produire que le sujet."
    )

    expected_format = build_expected_format(
        matiere=matiere,
        classe=classe,
        include_correction=include_correction,
        type_demande=type_demande,
    )

    formal_rules = build_formal_assessment_rules(
        type_demande=type_demande,
        include_correction=include_correction,
    )

    physics_short_circuit_rules = build_physics_short_circuit_rules(
        matiere=matiere,
        theme=theme,
    )

    return f"""
Tu es AskCI Éducation, assistant pédagogique pour le système scolaire ivoirien.

Demande utilisateur :
{question}

Objectif :
Générer un NOUVEAU contenu pédagogique original, similaire aux modèles disponibles, sans recopier mot à mot les sources.

Paramètres :
- Matière : {matiere}
- Niveau : {niveau}
- Classe : {classe}
- Thème : {theme}
- Type demandé : {type_demande}
- Corrigé demandé : {"Oui" if include_correction else "Non"}

Règles pédagogiques obligatoires :
- Répondre entièrement en français.
- Respecter strictement la matière, le niveau, la classe et le thème demandés.
- Utiliser le programme officiel comme cadre pédagogique.
- Utiliser les sujets, formats, devoirs, compositions, BEPC/BAC ou corrigés comme inspiration de structure.
- Si les modèles exacts sont insuffisants, utiliser le contexte complémentaire sans changer le niveau demandé.
- Ne jamais prétendre que le sujet généré est un sujet officiel.
- Ne jamais recopier un énoncé existant.
- Générer un contenu directement exploitable par un professeur.
{correction_rule}

Règles de cohérence obligatoires :
- Toutes les données nécessaires aux calculs ou réponses doivent être dans l’énoncé.
- Aucune hypothèse cachée n’est autorisée.
- Chaque question doit être résoluble avec les seules données de l’énoncé.
- Les nombres, exemples et consignes doivent être adaptés à la classe.
- Ne jamais utiliser d’expression anglaise.
- En mathématiques, dire “On effectue le produit en croix”, jamais “cross-multiplier”.
- Si le corrigé n’est pas demandé, ne donne aucune réponse directe.
- Le sujet ne doit contenir aucune contradiction interne.
- Une donnée fausse ou non proportionnelle ne doit pas être utilisée ensuite comme base d’un calcul proportionnel.

{formal_rules}

{physics_short_circuit_rules}

{expected_format}

{programme_context}

{evaluation_context}

{fallback_context}
""".strip()


def generate_with_openai(
    client: OpenAI,
    model: str,
    prompt: str,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Tu génères des contenus pédagogiques originaux, cohérents, "
                    "adaptés au niveau scolaire et entièrement en français. "
                    "Tu respectes strictement la consigne concernant la présence ou non du corrigé. "
                    "Pour les devoirs, compositions et examens, tu adoptes un style académique strict "
                    "et tu évites toute contradiction pédagogique."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    output_text = getattr(response, "output_text", None)

    if output_text:
        return output_text.strip()

    try:
        parts = []

        for item in response.output:
            for content in item.content:
                if hasattr(content, "text"):
                    parts.append(content.text)

        return "\n".join(parts).strip()

    except Exception:
        return str(response)


def generate_with_mistral(
    client: Mistral,
    model: str,
    prompt: str,
) -> str:
    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu génères des contenus pédagogiques originaux, cohérents, "
                    "adaptés au niveau scolaire et entièrement en français. "
                    "Tu respectes strictement la consigne concernant la présence ou non du corrigé. "
                    "Pour les devoirs, compositions et examens, tu adoptes un style académique strict "
                    "et tu évites toute contradiction pédagogique."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    if not response.choices:
        raise RuntimeError("Mistral n'a retourné aucune réponse.")

    content = response.choices[0].message.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()

def generate_with_claude_placeholder(
    prompt: str,
    task: str,
) -> str:
    """
    Claude est prévu dans l'architecture, mais désactivé pour la V1.
    Cette fonction évite les appels accidentels tant que l'intégration Anthropic
    n'est pas réellement branchée.
    """
    raise RuntimeError(
        f"Claude est configuré comme provider pour {task}, "
        "mais l'intégration Claude n'est pas encore activée dans cette version."
    )


def generate_text_with_provider(
    openai_client: OpenAI,
    model: str,
    prompt: str,
) -> str:
    """
    Point d'entrée unique pour la génération.
    V1 : OpenAI actif.
    V2 : Claude pourra être activé ici sans changer le reste du moteur.
    """
    provider = assert_provider_available("generation")

    if provider == "openai":
        return generate_with_openai(
            client=openai_client,
            model=model,
            prompt=prompt,
        )

    if provider == "claude":
        return generate_with_claude_placeholder(
            prompt=prompt,
            task="generation",
        )

    raise RuntimeError("Aucun provider IA disponible pour la génération.")


def run_quality_check_with_provider(
    openai_client: OpenAI,
    model: str,
    generated_subject: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
) -> dict[str, Any]:
    """
    Point d'entrée unique pour le contrôle qualité.
    V1 : OpenAI actif.
    V2 : Claude pourra devenir relecteur qualité secondaire.
    """
    provider = assert_provider_available("quality")

    if provider == "openai":
        return run_quality_check(
            client=openai_client,
            model=model,
            generated_subject=generated_subject,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            theme=theme,
            type_demande=type_demande,
            include_correction=include_correction,
        )

    if provider == "claude":
        generate_with_claude_placeholder(
            prompt=generated_subject,
            task="quality",
        )

    raise RuntimeError("Aucun provider IA disponible pour le contrôle qualité.")


def build_quality_check_prompt(
    generated_subject: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
) -> str:
    correction_checks = (
        """
Contrôles spécifiques au corrigé :
- Le corrigé est-il présent ?
- Le corrigé est-il séparé du sujet ?
- Le corrigé ajoute-t-il une donnée absente de l’énoncé ?
- Les réponses/corrections sont-elles cohérentes ?
"""
        if include_correction
        else """
Contrôles spécifiques à l’absence de corrigé :
- Le contenu ne doit pas contenir de section Corrigé.
- Le contenu ne doit pas donner les réponses aux questions.
- Le contenu ne doit pas afficher de solution détaillée.
"""
    )

    formal_checks = (
        """
Contrôles spécifiques devoir/composition/examen :
- Le style est-il académique et adapté à une évaluation ?
- Le texte évite-t-il les expressions familières ou maladroites comme "Un autre souci" ?
- Le sujet est-il structuré en exercices ou situations d’évaluation claires ?
- Le barème et la durée sont-ils cohérents si présents ?
- Les données numériques sont-elles cohérentes entre elles ?
- Le sujet évite-t-il une contradiction du type : montrer qu’une situation n’est pas proportionnelle puis l’utiliser comme proportionnelle ?
- Le sujet évite-t-il les formulations "en supposant que..." lorsqu’elles contredisent les données précédentes ?
"""
        if is_formal_assessment(type_demande)
        else ""
    )

    physics_short_circuit_checks = build_physics_short_circuit_rules(
        matiere=matiere,
        theme=theme,
    )

    return f"""
Tu es un contrôleur qualité pédagogique strict pour AskCI Éducation.

Analyse le contenu généré ci-dessous.

Objectif :
Vérifier si le contenu est exploitable directement par un professeur ou un élève et s’il respecte la demande.

Contrôles obligatoires :
1. Le contenu respecte-t-il la matière : {matiere} ?
2. Le contenu respecte-t-il le niveau : {niveau} ?
3. Le contenu respecte-t-il la classe : {classe} ?
4. Le contenu traite-t-il clairement le thème : {theme} ?
5. Toutes les questions sont-elles résolubles avec les données présentes dans l’énoncé ?
6. Le niveau est-il adapté à la classe ?
7. Le contenu évite-t-il la copie mot à mot des sources ?
8. Le texte est-il entièrement en français ?
9. Le contenu respecte-t-il la consigne corrigé demandé = {"Oui" if include_correction else "Non"} ?
10. Le contenu ne contient-il aucune contradiction pédagogique ou numérique ?

{correction_checks}

{formal_checks}

{physics_short_circuit_checks}

Réponds uniquement en JSON valide, sans markdown, avec cette structure :

{{
  "valide": true,
  "score_sur_100": 0,
  "problemes": [],
  "corrections_recommandees": [],
  "decision": "ACCEPTER ou REGENERER"
}}

Règle de décision :
- Si une contradiction pédagogique ou numérique est détectée, decision doit être "REGENERER".
- Si le style est familier dans un devoir/composition/examen, decision doit être "REGENERER".
- Si le corrigé est présent alors qu’il n’est pas demandé, decision doit être "REGENERER".
- Si le corrigé est absent alors qu’il est demandé, decision doit être "REGENERER".

Contenu à contrôler :

{generated_subject}
""".strip()


def run_quality_check(
    client: OpenAI,
    model: str,
    generated_subject: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
) -> dict[str, Any]:
    prompt = build_quality_check_prompt(
        generated_subject=generated_subject,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        type_demande=type_demande,
        include_correction=include_correction,
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Tu es un contrôleur qualité strict. "
                    "Tu réponds uniquement avec du JSON valide."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw = getattr(response, "output_text", None)

    if not raw:
        raw = str(response)

    raw = raw.strip()
    raw = re.sub(r"^```json", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"^```", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except Exception:
        return {
            "valide": False,
            "score_sur_100": 0,
            "problemes": [
                "La réponse du contrôle qualité n'est pas un JSON valide."
            ],
            "corrections_recommandees": [
                raw[:1000]
            ],
            "decision": "REGENERER",
        }


def is_quality_accepted(qa_report: dict[str, Any]) -> bool:
    decision = str(qa_report.get("decision", "")).upper()
    score = int(qa_report.get("score_sur_100", 0) or 0)
    valide = bool(qa_report.get("valide", False))

    return decision == "ACCEPTER" and score >= MIN_ACCEPTED_QUALITY_SCORE and valide


def quality_label(qa_report: dict[str, Any]) -> str:
    decision = str(qa_report.get("decision", "")).upper()
    score = int(qa_report.get("score_sur_100", 0) or 0)
    valide = bool(qa_report.get("valide", False))

    return f"{decision} / score {score}/100 / valide={valide}"


def build_repair_prompt(
    original_prompt: str,
    generated_subject: str,
    quality_report: dict[str, Any],
    type_demande: str,
    include_correction: bool,
) -> str:
    correction_rule = (
        "- Le corrigé est demandé : produire un corrigé complet dans une section ## Corrigé."
        if include_correction
        else "- Le corrigé n’est pas demandé : ne produire aucune réponse, aucune solution, aucun corrigé."
    )

    formal_rule = (
        """
- Le contenu est un devoir/composition/examen : utiliser un style académique strict.
- Remplacer toute formulation maladroite par "Exercice", "Situation d’évaluation" ou "Deuxième situation".
- Supprimer toute contradiction numérique ou pédagogique.
- Ne pas utiliser une relation non proportionnelle comme si elle était proportionnelle.
"""
        if is_formal_assessment(type_demande)
        else ""
    )

    physics_rule = build_physics_short_circuit_rules(
        matiere="Physique-Chimie",
        theme=generated_subject,
    )

    return f"""
Le contenu généré ci-dessous a été contrôlé et doit être corrigé.

Rapport qualité :
{json.dumps(quality_report, ensure_ascii=False, indent=2)}

Contenu à corriger :
{generated_subject}

Consigne :
Réécris entièrement le contenu en corrigeant tous les problèmes signalés.

Règles non négociables :
- Répondre entièrement en français.
- Toutes les questions doivent être résolubles.
- Respecter la matière, le niveau, la classe et le thème demandés.
- Produire un contenu original.
- Garder un style directement exploitable par un professeur.
- Ne jamais utiliser d’expression anglaise.
- Ne jamais écrire "Un autre souci".
- Ne jamais conserver une contradiction entre l’énoncé et les questions.
{correction_rule}
{formal_rule}
{physics_rule}

Contexte initial :
{original_prompt}
""".strip()


def extract_sources_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []

    for item in results:
        summary.append({
            "chunk_id": item.get("id"),
            "document_id": item.get("document_id"),
            "titre": item.get("titre"),
            "matiere": item.get("matiere"),
            "niveau": item.get("niveau"),
            "classe": item.get("classe"),
            "type_document": item.get("type_document"),
            "similarity": item.get("similarity"),
        })

    return summary


def save_generated_subject_to_supabase(
    supabase,
    question: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    generated_text: str,
    qa_report: dict[str, Any],
    programme_results: list[dict[str, Any]],
    evaluation_results: list[dict[str, Any]],
    fallback_results: list[dict[str, Any]],
    generation_model: str,
    embedding_model: str,
    include_correction: bool,
) -> dict[str, Any]:
    if not is_quality_accepted(qa_report):
        raise RuntimeError(
            "Sujet rejeté par le contrôle qualité final. "
            f"État final : {quality_label(qa_report)}. "
            "Le sujet n'a pas été sauvegardé dans Supabase."
        )

    subject_text, correction_text = split_subject_and_correction(generated_text)

    if not include_correction:
        correction_text = None

    correction_available = bool(correction_text)

    payload = {
        "question": question,
        "matiere": matiere,
        "niveau": niveau,
        "classe": classe,
        "theme": extract_theme(generated_text, theme),
        "generated_text": generated_text,
        "subject_text": subject_text,
        "correction_text": correction_text,
        "include_correction": include_correction,
        "correction_available": correction_available,
        "correction_requested_at": datetime.now(timezone.utc).isoformat() if include_correction else None,
        "quality_score": int(qa_report.get("score_sur_100", 0) or 0),
        "quality_decision": str(qa_report.get("decision", "")),
        "quality_report": qa_report,
        "sources_programme": extract_sources_summary(programme_results),
        "sources_modeles": extract_sources_summary(evaluation_results + fallback_results),
        "generation_model": generation_model,
        "embedding_model": embedding_model,
    }

    response = (
        supabase
        .table(TABLE_GENERATED_SUBJECTS)
        .insert(payload)
        .execute()
    )

    data = response.data or []

    if not data:
        raise RuntimeError("Insertion Supabase effectuée sans retour de donnée.")

    return data[0]


def build_output_paths(
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    include_correction: bool,
    rejected: bool = False,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix = "askci_sujet_rejete" if rejected else "askci_sujet_genere"
    correction_suffix = "avec_corrige" if include_correction else "sans_corrige"

    base_name = "_".join([
        prefix,
        slugify(matiere),
        slugify(niveau),
        slugify(classe),
        slugify(theme),
        correction_suffix,
        timestamp,
    ])

    output_path = BASE_DIR / f"{base_name}.txt"
    qa_path = BASE_DIR / f"{base_name}_qa.json"

    return output_path, qa_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Générer un sujet similaire AskCI Éducation avec RAG + contrôle qualité."
    )

    parser.add_argument(
        "--matiere",
        required=True,
        help='Exemple : "Mathématiques", "Français", "Physique-Chimie", "Philosophie"',
    )

    parser.add_argument(
        "--niveau",
        required=True,
        help='Exemple : "Collège" ou "Lycée"',
    )

    parser.add_argument(
        "--classe",
        required=True,
        help='Exemple : "3e", "5e", "Seconde A", "Terminale D"',
    )

    parser.add_argument(
        "--theme",
        required=True,
        help='Exemple : "proportionnalité", "expression écrite", "électricité"',
    )

    parser.add_argument(
        "--type",
        default="exercice",
        help='Exemple : "exercice", "devoir", "composition", "sujet BEPC"',
    )

    parser.add_argument(
        "--type-document",
        default=None,
        help='Filtre optionnel : "Devoir", "BEPC", "Programme", etc.',
    )

    parser.add_argument(
        "--include-correction",
        action="store_true",
        help="Inclure le corrigé dans le sujet généré.",
    )

    return parser.parse_args()


def main() -> None:
    configure_utf8_output()

    args = parse_args()

    matiere = clean_text(args.matiere)
    niveau = clean_text(args.niveau)
    classe = clean_text(args.classe)
    theme = clean_text(args.theme)
    type_demande = clean_text(args.type)
    requested_type_document = normalize_type_document(args.type_document)
    include_correction = bool(args.include_correction)

    load_env_file(ENV_PATH)

    supabase_url = require_env("SUPABASE_URL")
    supabase_key = require_env("SUPABASE_SERVICE_ROLE_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")

    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    ).strip() or DEFAULT_EMBEDDING_MODEL

    generation_model = os.getenv(
        "OPENAI_GENERATION_MODEL",
        DEFAULT_GENERATION_MODEL,
    ).strip() or DEFAULT_GENERATION_MODEL

    question = build_question(
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        type_demande=type_demande,
        include_correction=include_correction,
    )

    openai_client = OpenAI(api_key=openai_api_key)
    supabase = create_client(supabase_url, supabase_key)

    print("ASKCI ÉDUCATION — GÉNÉRATION DYNAMIQUE DE SUJET SIMILAIRE")
    print("=" * 80)
    print(f"Question          : {question}")
    print(f"Matière           : {matiere}")
    print(f"Niveau            : {niveau}")
    print(f"Classe            : {classe}")
    print(f"Thème             : {theme}")
    print(f"Type demandé      : {type_demande}")
    print(f"Corrigé demandé   : {'Oui' if include_correction else 'Non'}")
    print(f"Type doc forcé    : {requested_type_document or 'Aucun'}")
    ai_status = provider_status()

    print(f"Modèle embedding  : {embedding_model}")
    print(f"Modèle génération : {generation_model}")
    print(f"IA génération     : {ai_status.get('selected_generation_provider')}")
    print(f"IA qualité        : {ai_status.get('selected_quality_provider')}")
    print(f"OpenAI actif      : {ai_status.get('openai_generation_available')}")
    print(f"Claude actif      : {ai_status.get('claude_generation_available')}")
    print("=" * 80)

    query_embedding = create_query_embedding(
        client=openai_client,
        model=embedding_model,
        query=question,
    )

    if requested_type_document:
        evaluation_results = search_by_types(
            supabase=supabase,
            query_embedding=query_embedding,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            types=[requested_type_document],
            per_type_count=10,
        )
    else:
        evaluation_results = search_by_types(
            supabase=supabase,
            query_embedding=query_embedding,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            types=EVALUATION_TYPES,
            per_type_count=4,
        )

    programme_results = search_by_types(
        supabase=supabase,
        query_embedding=query_embedding,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        types=PROGRAMME_TYPES,
        per_type_count=6,
    )

    fallback_results = []

    if len(evaluation_results) < 3:
        fallback_results = search_without_type(
            supabase=supabase,
            query_embedding=query_embedding,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            match_count=8,
        )

    if len(evaluation_results) + len(programme_results) + len(fallback_results) < 3:
        fallback_results.extend(
            search_fallback_same_subject_level(
                supabase=supabase,
                query_embedding=query_embedding,
                matiere=matiere,
                niveau=niveau,
                match_count=8,
            )
        )

    fallback_results = sort_by_similarity(dedupe_results(fallback_results))

    print(f"Sources programme trouvées     : {len(programme_results)}")
    print(f"Sources modèles trouvées       : {len(evaluation_results)}")
    print(f"Sources complémentaires        : {len(fallback_results)}")
    print("=" * 80)

    generation_prompt = build_generation_prompt(
        question=question,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        type_demande=type_demande,
        include_correction=include_correction,
        programme_results=programme_results,
        evaluation_results=evaluation_results,
        fallback_results=fallback_results,
    )

    generated = generate_text_with_provider(
        openai_client=openai_client,
        model=generation_model,
        prompt=generation_prompt,
    )

    qa_report = run_quality_check_with_provider(
        openai_client=openai_client,
        model=generation_model,
        generated_subject=generated,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        type_demande=type_demande,
        include_correction=include_correction,
    )

    qa_report = apply_physics_short_circuit_quality_guard(
        qa_report=qa_report,
        generated_subject=generated,
        matiere=matiere,
        theme=theme,
        include_correction=include_correction,
    )

    print(f"Contrôle qualité initial : {quality_label(qa_report)}")

    repair_attempt = 0

    while not is_quality_accepted(qa_report) and repair_attempt < MAX_QUALITY_REPAIR_ATTEMPTS:
        repair_attempt += 1
        print(f"Correction automatique du sujet... tentative {repair_attempt}/{MAX_QUALITY_REPAIR_ATTEMPTS}")

        repair_prompt = build_repair_prompt(
            original_prompt=generation_prompt,
            generated_subject=generated,
            quality_report=qa_report,
            type_demande=type_demande,
            include_correction=include_correction,
        )

        generated = generate_text_with_provider(
            openai_client=openai_client,
            model=generation_model,
            prompt=repair_prompt,
        )

        qa_report = run_quality_check_with_provider(
            openai_client=openai_client,
            model=generation_model,
            generated_subject=generated,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            theme=theme,
            type_demande=type_demande,
            include_correction=include_correction,
        )

        qa_report = apply_physics_short_circuit_quality_guard(
            qa_report=qa_report,
            generated_subject=generated,
            matiere=matiere,
            theme=theme,
            include_correction=include_correction,
        )

        print(f"Contrôle qualité après correction {repair_attempt} : {quality_label(qa_report)}")

    if not is_quality_accepted(qa_report):
        output_path, qa_path = build_output_paths(
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            theme=theme,
            include_correction=include_correction,
            rejected=True,
        )

        output_path.write_text(generated, encoding="utf-8")

        qa_payload = {
            "question": question,
            "filtre": {
                "matiere": matiere,
                "niveau": niveau,
                "classe": classe,
                "theme": theme,
                "type_demande": type_demande,
                "type_document_force": requested_type_document,
                "include_correction": include_correction,
            },
            "models": {
                "embedding": embedding_model,
                "generation": generation_model,
            },
            "quality_report": qa_report,
            "sources_programme": extract_sources_summary(programme_results),
            "sources_modeles": extract_sources_summary(evaluation_results),
            "sources_complementaires": extract_sources_summary(fallback_results),
            "output_file": str(output_path),
            "saved_to_supabase": False,
            "rejected": True,
        }

        qa_path.write_text(
            json.dumps(qa_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Sujet rejeté après contrôle qualité final. "
            f"État final : {quality_label(qa_report)}. "
            f"Fichier rejeté : {output_path}. "
            f"Rapport rejeté : {qa_path}. "
            "Aucune sauvegarde Supabase n'a été faite."
        )

    output_path, qa_path = build_output_paths(
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        include_correction=include_correction,
        rejected=False,
    )

    output_path.write_text(generated, encoding="utf-8")

    subject_text, correction_text = split_subject_and_correction(generated)

    if not include_correction:
        correction_text = None

    qa_payload = {
        "question": question,
        "filtre": {
            "matiere": matiere,
            "niveau": niveau,
            "classe": classe,
            "theme": theme,
            "type_demande": type_demande,
            "type_document_force": requested_type_document,
            "include_correction": include_correction,
        },
        "models": {
            "embedding": embedding_model,
            "generation": generation_model,
        },
        "quality_report": qa_report,
        "sources_programme": extract_sources_summary(programme_results),
        "sources_modeles": extract_sources_summary(evaluation_results),
        "sources_complementaires": extract_sources_summary(fallback_results),
        "output_file": str(output_path),
        "saved_to_supabase": True,
        "rejected": False,
        "subject_text_preview": subject_text[:1000],
        "correction_available": bool(correction_text),
    }

    qa_path.write_text(
        json.dumps(qa_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    saved_row = save_generated_subject_to_supabase(
        supabase=supabase,
        question=question,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        generated_text=generated,
        qa_report=qa_report,
        programme_results=programme_results,
        evaluation_results=evaluation_results,
        fallback_results=fallback_results,
        generation_model=generation_model,
        embedding_model=embedding_model,
        include_correction=include_correction,
    )

    print("\n" + generated)
    print("\n" + "=" * 80)
    print(f"Sujet généré enregistré ici : {output_path}")
    print(f"Rapport qualité enregistré ici : {qa_path}")
    print(f"Sujet sauvegardé dans Supabase avec ID : {saved_row.get('id')}")
    print(f"Corrigé inclus : {'Oui' if include_correction else 'Non'}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        configure_utf8_output()
        main()
    except Exception as exc:
        print(f"ERREUR FATALE : {type(exc).__name__}: {exc}")
        sys.exit(1)