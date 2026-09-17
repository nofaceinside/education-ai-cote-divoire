from pathlib import Path
from datetime import datetime, timezone
import argparse
import os
import re
import subprocess
import sys
from typing import Any

from supabase import create_client


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

GENERATOR_SCRIPT = BASE_DIR / "generate_similar_subject.py"

TABLE_GENERATED_SUBJECTS = "generated_subjects"
TABLE_DELIVERY_HISTORY = "subject_delivery_history"
TABLE_SCHOOL_RESERVATIONS = "school_subject_reservations"

MIN_ACCEPTED_QUALITY_SCORE = 85

MODE_STUDENT_PRACTICE = "student_practice"
MODE_TEACHER_ASSIGNMENT = "teacher_assignment"
MODE_SCHOOL_EXAM = "school_exam"

VALID_MODES = {
    MODE_STUDENT_PRACTICE,
    MODE_TEACHER_ASSIGNMENT,
    MODE_SCHOOL_EXAM,
}


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
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)

    return value.strip()


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_subject_id_from_output(output: str) -> int:
    patterns = [
        r"Sujet sauvegardé dans Supabase avec ID\s*:\s*(\d+)",
        r"Sujet sauvegarde dans Supabase avec ID\s*:\s*(\d+)",
        r"ID\s*:\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)

        if match:
            return int(match.group(1))

    raise RuntimeError(
        "Impossible de récupérer l'ID du sujet généré dans la sortie du générateur."
    )


def get_supabase_client():
    load_env_file(ENV_PATH)

    supabase_url = require_env("SUPABASE_URL")
    supabase_key = require_env("SUPABASE_SERVICE_ROLE_KEY")

    return create_client(supabase_url, supabase_key)


def fetch_subject_by_id(supabase, subject_id: int) -> dict[str, Any]:
    response = (
        supabase
        .table(TABLE_GENERATED_SUBJECTS)
        .select("*")
        .eq("id", subject_id)
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        raise RuntimeError(f"Sujet introuvable dans Supabase : ID {subject_id}")

    return rows[0]


def theme_matches(row_theme: str | None, requested_theme: str) -> bool:
    row_theme_norm = normalize_for_match(row_theme)
    requested_theme_norm = normalize_for_match(requested_theme)

    if not requested_theme_norm:
        return True

    if requested_theme_norm in row_theme_norm:
        return True

    if row_theme_norm in requested_theme_norm:
        return True

    return False


def get_user_delivered_subject_ids(
    supabase,
    user_id: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
) -> set[int]:
    response = (
        supabase
        .table(TABLE_DELIVERY_HISTORY)
        .select("subject_id, theme")
        .eq("user_id", user_id)
        .eq("matiere", matiere)
        .eq("niveau", niveau)
        .eq("classe", classe)
        .execute()
    )

    rows = response.data or []

    delivered_ids = set()

    for row in rows:
        if row.get("subject_id") is None:
            continue

        if theme_matches(row.get("theme"), theme):
            delivered_ids.add(int(row["subject_id"]))

    return delivered_ids


def search_existing_student_subjects(
    supabase,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    include_correction: bool,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Règle importante :
    - Si l'élève ne demande PAS le corrigé, on peut servir un sujet qui possède un corrigé,
      mais on masque le corrigé dans la réponse.
    - Si l'élève demande le corrigé, il faut choisir un sujet où correction_available = true.
    """

    query = (
        supabase
        .table(TABLE_GENERATED_SUBJECTS)
        .select("*")
        .eq("matiere", matiere)
        .eq("niveau", niveau)
        .eq("classe", classe)
        .ilike("theme", f"%{theme}%")
        .gte("quality_score", MIN_ACCEPTED_QUALITY_SCORE)
        .eq("quality_decision", "ACCEPTER")
        .eq("usage_mode", MODE_STUDENT_PRACTICE)
        .eq("is_reusable", True)
        .order("created_at", desc=True)
        .limit(limit)
    )

    if include_correction:
        query = query.eq("correction_available", True)

    response = query.execute()

    return response.data or []


def select_unseen_subject(
    subjects: list[dict[str, Any]],
    delivered_ids: set[int],
) -> dict[str, Any] | None:
    for subject in subjects:
        subject_id = int(subject["id"])

        if subject_id not in delivered_ids:
            return subject

    return None


def select_least_recent_seen_subject(
    subjects: list[dict[str, Any]],
    delivered_ids: set[int],
) -> dict[str, Any] | None:
    """
    Fallback : si tous les sujets existants ont déjà été vus par l'élève,
    on peut retourner le plus ancien de la liste plutôt que de générer immédiatement.
    Pour l’instant, on ne l’utilise pas comme premier fallback, car on préfère générer
    une nouvelle variante quand tous les sujets sont épuisés.
    """

    for subject in reversed(subjects):
        subject_id = int(subject["id"])

        if subject_id in delivered_ids:
            return subject

    return None


def record_subject_delivery(
    supabase,
    user_id: str,
    subject: dict[str, Any],
    usage_mode: str,
) -> None:
    payload = {
        "user_id": user_id,
        "subject_id": int(subject["id"]),
        "matiere": subject.get("matiere"),
        "niveau": subject.get("niveau"),
        "classe": subject.get("classe"),
        "theme": subject.get("theme"),
        "usage_mode": usage_mode,
        "delivered_at": now_iso(),
    }

    (
        supabase
        .table(TABLE_DELIVERY_HISTORY)
        .insert(payload)
        .execute()
    )


def reserve_subject_for_school(
    supabase,
    subject: dict[str, Any],
    school_id: str,
    teacher_id: str | None,
    usage_mode: str,
) -> None:
    payload = {
        "subject_id": int(subject["id"]),
        "school_id": school_id,
        "teacher_id": teacher_id,
        "matiere": subject.get("matiere"),
        "niveau": subject.get("niveau"),
        "classe": subject.get("classe"),
        "theme": subject.get("theme"),
        "usage_mode": usage_mode,
        "reserved_at": now_iso(),
    }

    (
        supabase
        .table(TABLE_SCHOOL_RESERVATIONS)
        .insert(payload)
        .execute()
    )

    update_payload = {
        "usage_mode": usage_mode,
        "reserved_school_id": school_id,
        "reserved_teacher_id": teacher_id,
        "is_reusable": False,
    }

    (
        supabase
        .table(TABLE_GENERATED_SUBJECTS)
        .update(update_payload)
        .eq("id", int(subject["id"]))
        .execute()
    )


def update_generated_subject_usage(
    supabase,
    subject_id: int,
    usage_mode: str,
    is_reusable: bool,
    school_id: str | None = None,
    teacher_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "usage_mode": usage_mode,
        "is_reusable": is_reusable,
        "reserved_school_id": school_id,
        "reserved_teacher_id": teacher_id,
    }

    response = (
        supabase
        .table(TABLE_GENERATED_SUBJECTS)
        .update(payload)
        .eq("id", subject_id)
        .execute()
    )

    rows = response.data or []

    if rows:
        return rows[0]

    return fetch_subject_by_id(supabase, subject_id)


def run_generator(
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
    type_document: str | None = None,
) -> dict[str, Any]:
    if not GENERATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Script générateur introuvable : {GENERATOR_SCRIPT}")

    command = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--matiere",
        matiere,
        "--niveau",
        niveau,
        "--classe",
        classe,
        "--theme",
        theme,
        "--type",
        type_demande,
    ]

    if include_correction:
        command.append("--include-correction")

    if type_document:
        command.extend(["--type-document", type_document])

    process = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined_output = (process.stdout or "") + "\n" + (process.stderr or "")

    if process.returncode != 0:
        raise RuntimeError(
            "La génération du sujet a échoué.\n"
            f"Code retour : {process.returncode}\n"
            f"Sortie :\n{combined_output}"
        )

    subject_id = parse_subject_id_from_output(combined_output)

    supabase = get_supabase_client()
    subject = fetch_subject_by_id(supabase, subject_id)

    return subject


def build_subject_response(
    subject: dict[str, Any],
    include_correction: bool,
    source: str,
    usage_mode: str,
) -> dict[str, Any]:
    subject_text = clean_text(subject.get("subject_text") or subject.get("generated_text") or "")
    correction_text = clean_text(subject.get("correction_text") or "")

    response = {
        "source": source,
        "usage_mode": usage_mode,
        "subject_id": int(subject["id"]),
        "matiere": subject.get("matiere"),
        "niveau": subject.get("niveau"),
        "classe": subject.get("classe"),
        "theme": subject.get("theme"),
        "quality_score": subject.get("quality_score"),
        "quality_decision": subject.get("quality_decision"),
        "include_correction": bool(include_correction),
        "correction_available": bool(subject.get("correction_available")),
        "subject_text": subject_text,
        "correction_text": correction_text if include_correction else None,
    }

    return response


def get_subject_for_student_practice(
    supabase,
    user_id: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
) -> dict[str, Any]:
    delivered_ids = get_user_delivered_subject_ids(
        supabase=supabase,
        user_id=user_id,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
    )

    existing_subjects = search_existing_student_subjects(
        supabase=supabase,
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        include_correction=include_correction,
        limit=50,
    )

    selected_subject = select_unseen_subject(
        subjects=existing_subjects,
        delivered_ids=delivered_ids,
    )

    if selected_subject:
        record_subject_delivery(
            supabase=supabase,
            user_id=user_id,
            subject=selected_subject,
            usage_mode=MODE_STUDENT_PRACTICE,
        )

        return build_subject_response(
            subject=selected_subject,
            include_correction=include_correction,
            source="existing_unseen",
            usage_mode=MODE_STUDENT_PRACTICE,
        )

    generated_subject = run_generator(
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        type_demande=type_demande,
        include_correction=include_correction,
    )

    generated_subject = update_generated_subject_usage(
        supabase=supabase,
        subject_id=int(generated_subject["id"]),
        usage_mode=MODE_STUDENT_PRACTICE,
        is_reusable=True,
    )

    record_subject_delivery(
        supabase=supabase,
        user_id=user_id,
        subject=generated_subject,
        usage_mode=MODE_STUDENT_PRACTICE,
    )

    return build_subject_response(
        subject=generated_subject,
        include_correction=include_correction,
        source="generated_new",
        usage_mode=MODE_STUDENT_PRACTICE,
    )


def get_subject_for_teacher_assignment(
    supabase,
    user_id: str,
    teacher_id: str,
    school_id: str | None,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
) -> dict[str, Any]:
    generated_subject = run_generator(
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        type_demande=type_demande,
        include_correction=include_correction,
    )

    generated_subject = update_generated_subject_usage(
        supabase=supabase,
        subject_id=int(generated_subject["id"]),
        usage_mode=MODE_TEACHER_ASSIGNMENT,
        is_reusable=False,
        school_id=school_id,
        teacher_id=teacher_id,
    )

    if school_id:
        reserve_subject_for_school(
            supabase=supabase,
            subject=generated_subject,
            school_id=school_id,
            teacher_id=teacher_id,
            usage_mode=MODE_TEACHER_ASSIGNMENT,
        )

    record_subject_delivery(
        supabase=supabase,
        user_id=user_id,
        subject=generated_subject,
        usage_mode=MODE_TEACHER_ASSIGNMENT,
    )

    return build_subject_response(
        subject=generated_subject,
        include_correction=include_correction,
        source="generated_new_reserved_teacher",
        usage_mode=MODE_TEACHER_ASSIGNMENT,
    )


def get_subject_for_school_exam(
    supabase,
    user_id: str,
    school_id: str,
    teacher_id: str | None,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
) -> dict[str, Any]:
    generated_subject = run_generator(
        matiere=matiere,
        niveau=niveau,
        classe=classe,
        theme=theme,
        type_demande=type_demande,
        include_correction=include_correction,
    )

    generated_subject = update_generated_subject_usage(
        supabase=supabase,
        subject_id=int(generated_subject["id"]),
        usage_mode=MODE_SCHOOL_EXAM,
        is_reusable=False,
        school_id=school_id,
        teacher_id=teacher_id,
    )

    reserve_subject_for_school(
        supabase=supabase,
        subject=generated_subject,
        school_id=school_id,
        teacher_id=teacher_id,
        usage_mode=MODE_SCHOOL_EXAM,
    )

    record_subject_delivery(
        supabase=supabase,
        user_id=user_id,
        subject=generated_subject,
        usage_mode=MODE_SCHOOL_EXAM,
    )

    return build_subject_response(
        subject=generated_subject,
        include_correction=include_correction,
        source="generated_new_reserved_school",
        usage_mode=MODE_SCHOOL_EXAM,
    )


def get_or_generate_subject(
    mode: str,
    user_id: str,
    matiere: str,
    niveau: str,
    classe: str,
    theme: str,
    type_demande: str,
    include_correction: bool,
    teacher_id: str | None = None,
    school_id: str | None = None,
) -> dict[str, Any]:
    mode = clean_text(mode)

    if mode not in VALID_MODES:
        raise RuntimeError(
            f"Mode invalide : {mode}. Modes valides : {', '.join(sorted(VALID_MODES))}"
        )

    if not user_id:
        raise RuntimeError("user_id est obligatoire.")

    if mode == MODE_TEACHER_ASSIGNMENT and not teacher_id:
        raise RuntimeError("teacher_id est obligatoire en mode teacher_assignment.")

    if mode == MODE_SCHOOL_EXAM and not school_id:
        raise RuntimeError("school_id est obligatoire en mode school_exam.")

    supabase = get_supabase_client()

    if mode == MODE_STUDENT_PRACTICE:
        return get_subject_for_student_practice(
            supabase=supabase,
            user_id=user_id,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            theme=theme,
            type_demande=type_demande,
            include_correction=include_correction,
        )

    if mode == MODE_TEACHER_ASSIGNMENT:
        return get_subject_for_teacher_assignment(
            supabase=supabase,
            user_id=user_id,
            teacher_id=teacher_id or user_id,
            school_id=school_id,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            theme=theme,
            type_demande=type_demande,
            include_correction=include_correction,
        )

    if mode == MODE_SCHOOL_EXAM:
        return get_subject_for_school_exam(
            supabase=supabase,
            user_id=user_id,
            school_id=school_id or "",
            teacher_id=teacher_id,
            matiere=matiere,
            niveau=niveau,
            classe=classe,
            theme=theme,
            type_demande=type_demande,
            include_correction=include_correction,
        )

    raise RuntimeError(f"Mode non géré : {mode}")


def print_response(result: dict[str, Any], full: bool) -> None:
    print("ASKCI SUBJECT SERVICE — RÉSULTAT")
    print("=" * 90)
    print(f"Source              : {result.get('source')}")
    print(f"Mode                : {result.get('usage_mode')}")
    print(f"Subject ID          : {result.get('subject_id')}")
    print(f"Matière             : {result.get('matiere')}")
    print(f"Niveau              : {result.get('niveau')}")
    print(f"Classe              : {result.get('classe')}")
    print(f"Thème               : {result.get('theme')}")
    print(f"Score qualité       : {result.get('quality_score')}")
    print(f"Décision qualité    : {result.get('quality_decision')}")
    print(f"Corrigé demandé     : {'Oui' if result.get('include_correction') else 'Non'}")
    print(f"Corrigé disponible  : {'Oui' if result.get('correction_available') else 'Non'}")
    print("=" * 90)

    subject_text = clean_text(result.get("subject_text") or "")
    correction_text = clean_text(result.get("correction_text") or "")

    if not full and len(subject_text) > 1800:
        subject_text = subject_text[:1800] + "\n..."

    print("\nSUJET")
    print("-" * 90)
    print(subject_text)

    if result.get("include_correction") and correction_text:
        if not full and len(correction_text) > 1800:
            correction_text = correction_text[:1800] + "\n..."

        print("\nCORRIGÉ")
        print("-" * 90)
        print(correction_text)

    print("\n" + "=" * 90)
    print("FIN")
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Service métier AskCI : servir ou générer un sujet selon le mode utilisateur."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(VALID_MODES),
        help="student_practice, teacher_assignment ou school_exam",
    )

    parser.add_argument(
        "--user-id",
        required=True,
        help="Identifiant utilisateur.",
    )

    parser.add_argument(
        "--teacher-id",
        default=None,
        help="Identifiant professeur, requis pour teacher_assignment.",
    )

    parser.add_argument(
        "--school-id",
        default=None,
        help="Identifiant école, requis pour school_exam.",
    )

    parser.add_argument(
        "--matiere",
        required=True,
        help='Exemple : "Mathématiques"',
    )

    parser.add_argument(
        "--niveau",
        required=True,
        help='Exemple : "Collège"',
    )

    parser.add_argument(
        "--classe",
        required=True,
        help='Exemple : "3e"',
    )

    parser.add_argument(
        "--theme",
        required=True,
        help='Exemple : "proportionnalité"',
    )

    parser.add_argument(
        "--type",
        default="exercice",
        help='Exemple : "exercice", "devoir", "composition", "sujet BEPC"',
    )

    parser.add_argument(
        "--include-correction",
        action="store_true",
        help="Inclure ou demander le corrigé.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Afficher le sujet complet.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = get_or_generate_subject(
        mode=clean_text(args.mode),
        user_id=clean_text(args.user_id),
        teacher_id=clean_text(args.teacher_id),
        school_id=clean_text(args.school_id),
        matiere=clean_text(args.matiere),
        niveau=clean_text(args.niveau),
        classe=clean_text(args.classe),
        theme=clean_text(args.theme),
        type_demande=clean_text(args.type),
        include_correction=bool(args.include_correction),
    )

    print_response(
        result=result,
        full=bool(args.full),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR FATALE : {type(exc).__name__}: {exc}")
        sys.exit(1)