from pathlib import Path
import json
import os
import sys
import traceback
from typing import Optional, Literal, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from supabase import create_client

from askci_subject_service import get_or_generate_subject


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

APP_NAME = "AskCI Éducation Subjects API"
APP_VERSION = "1.0.2"

TABLE_GENERATED_SUBJECTS = "generated_subjects"

VALID_MODES = {
    "student_practice",
    "teacher_assignment",
    "school_exam",
}


class UTF8JSONResponse(Response):
    media_type = "application/json; charset=utf-8"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


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


def get_supabase_client():
    load_env_file(ENV_PATH)

    supabase_url = require_env("SUPABASE_URL")
    supabase_key = require_env("SUPABASE_SERVICE_ROLE_KEY")

    return create_client(supabase_url, supabase_key)


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""

    return str(value).strip()


class SubjectRequest(BaseModel):
    mode: Literal["student_practice", "teacher_assignment", "school_exam"] = Field(
        ...,
        description="Mode de génération : élève, professeur ou école.",
    )

    user_id: str = Field(
        ...,
        min_length=1,
        description="Identifiant de l'utilisateur demandeur.",
    )

    teacher_id: Optional[str] = Field(
        default=None,
        description="Identifiant professeur. Requis pour teacher_assignment.",
    )

    school_id: Optional[str] = Field(
        default=None,
        description="Identifiant école. Requis pour school_exam, optionnel pour teacher_assignment.",
    )

    matiere: str = Field(
        ...,
        min_length=1,
        description='Exemple : "Mathématiques", "Français", "Physique-Chimie".',
    )

    niveau: str = Field(
        ...,
        min_length=1,
        description='Exemple : "Collège".',
    )

    classe: str = Field(
        ...,
        min_length=1,
        description='Exemple : "3e", "4e", "5e", "6e".',
    )

    theme: str = Field(
        ...,
        min_length=1,
        description='Exemple : "proportionnalité".',
    )

    type: str = Field(
        default="exercice",
        description='Exemple : "exercice", "devoir", "composition", "sujet BEPC".',
    )

    include_correction: bool = Field(
        default=False,
        description="Inclure ou demander le corrigé.",
    )


class SubjectResponse(BaseModel):
    success: bool
    source: str
    usage_mode: str
    subject_id: int
    matiere: Optional[str]
    niveau: Optional[str]
    classe: Optional[str]
    theme: Optional[str]
    quality_score: Optional[int]
    quality_decision: Optional[str]
    include_correction: bool
    correction_available: bool
    subject_text: str
    correction_text: Optional[str] = None


class CorrectionResponse(BaseModel):
    success: bool
    subject_id: int
    correction_available: bool
    correction_text: Optional[str] = None
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool
    error: str
    detail: Optional[str] = None


configure_utf8_output()
load_env_file(ENV_PATH)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API AskCI Éducation pour servir ou générer des sujets selon le mode utilisateur.",
    default_response_class=UTF8JSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=UTF8JSONResponse)
def root() -> dict[str, Any]:
    return {
        "success": True,
        "app": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "request_subject": "POST /api/education/subjects/request",
            "get_correction": "GET /api/education/subjects/{subject_id}/correction",
        },
    }


@app.get("/health", response_class=UTF8JSONResponse)
def health() -> dict[str, Any]:
    return {
        "success": True,
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


def validate_request(payload: SubjectRequest) -> None:
    mode = clean_text(payload.mode)

    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Mode invalide : {mode}",
        )

    if not clean_text(payload.user_id):
        raise HTTPException(
            status_code=400,
            detail="user_id est obligatoire.",
        )

    if mode == "teacher_assignment" and not clean_text(payload.teacher_id):
        raise HTTPException(
            status_code=400,
            detail="teacher_id est obligatoire pour teacher_assignment.",
        )

    if mode == "school_exam" and not clean_text(payload.school_id):
        raise HTTPException(
            status_code=400,
            detail="school_id est obligatoire pour school_exam.",
        )

    required_fields = {
        "matiere": payload.matiere,
        "niveau": payload.niveau,
        "classe": payload.classe,
        "theme": payload.theme,
    }

    for field_name, field_value in required_fields.items():
        if not clean_text(field_value):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} est obligatoire.",
            )


@app.post(
    "/api/education/subjects/request",
    response_model=SubjectResponse,
    response_class=UTF8JSONResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def request_subject(payload: SubjectRequest) -> SubjectResponse:
    validate_request(payload)

    try:
        result = get_or_generate_subject(
            mode=clean_text(payload.mode),
            user_id=clean_text(payload.user_id),
            teacher_id=clean_text(payload.teacher_id),
            school_id=clean_text(payload.school_id),
            matiere=clean_text(payload.matiere),
            niveau=clean_text(payload.niveau),
            classe=clean_text(payload.classe),
            theme=clean_text(payload.theme),
            type_demande=clean_text(payload.type) or "exercice",
            include_correction=bool(payload.include_correction),
        )

        return SubjectResponse(
            success=True,
            source=str(result.get("source") or ""),
            usage_mode=str(result.get("usage_mode") or ""),
            subject_id=int(result.get("subject_id")),
            matiere=result.get("matiere"),
            niveau=result.get("niveau"),
            classe=result.get("classe"),
            theme=result.get("theme"),
            quality_score=result.get("quality_score"),
            quality_decision=result.get("quality_decision"),
            include_correction=bool(result.get("include_correction")),
            correction_available=bool(result.get("correction_available")),
            subject_text=str(result.get("subject_text") or ""),
            correction_text=result.get("correction_text"),
        )

    except HTTPException:
        raise

    except Exception as exc:
        error_detail = traceback.format_exc()

        print("ERREUR API /api/education/subjects/request")
        print(error_detail)

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )


@app.get(
    "/api/education/subjects/{subject_id}/correction",
    response_model=CorrectionResponse,
    response_class=UTF8JSONResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def get_subject_correction(subject_id: int) -> CorrectionResponse:
    try:
        supabase = get_supabase_client()

        response = (
            supabase
            .table(TABLE_GENERATED_SUBJECTS)
            .select("id, correction_available, correction_text")
            .eq("id", subject_id)
            .limit(1)
            .execute()
        )

        rows = response.data or []

        if not rows:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "NOT_FOUND",
                    "message": f"Sujet introuvable : {subject_id}",
                },
            )

        row = rows[0]
        correction_available = bool(row.get("correction_available"))
        correction_text = clean_text(row.get("correction_text"))

        if not correction_available or not correction_text:
            return CorrectionResponse(
                success=True,
                subject_id=subject_id,
                correction_available=False,
                correction_text=None,
                message="Aucun corrigé disponible pour ce sujet.",
            )

        return CorrectionResponse(
            success=True,
            subject_id=subject_id,
            correction_available=True,
            correction_text=correction_text,
            message="Corrigé disponible.",
        )

    except HTTPException:
        raise

    except Exception as exc:
        error_detail = traceback.format_exc()

        print("ERREUR API /api/education/subjects/{subject_id}/correction")
        print(error_detail)

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )




@app.get("/ai-status")
def get_ai_status():
    """
    Statut IA interne pour AskCI Éducation.
    V1 : OpenAI actif, Claude préparé mais désactivé.
    """
    try:
        from ai_providers import provider_status

        return {
            "success": True,
            "providers": provider_status(),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "providers": {
                "selected_generation_provider": "none",
                "selected_quality_provider": "none",
                "openai_generation_available": False,
                "openai_quality_available": False,
                "claude_generation_available": False,
                "claude_quality_available": False,
            },
        }




@app.get("/subjects/history")
def get_subjects_history(
    mode: str | None = None,
    matiere: str | None = None,
    niveau: str | None = None,
    classe: str | None = None,
    theme: str | None = None,
    limit: int = 30,
):
    """
    Historique des sujets générés AskCI Éducation.
    Sert à construire plus tard : Mes sujets / Historique professeur / Historique école.
    """
    try:
        import os
        from pathlib import Path
        from supabase import create_client

        env_path = Path(__file__).resolve().parent / ".env"

        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value

        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

        if not supabase_url or not supabase_key:
            return {
                "success": False,
                "error": "Supabase non configuré.",
                "items": [],
            }

        client = create_client(supabase_url, supabase_key)

        safe_limit = max(1, min(int(limit or 30), 100))

        query = (
            client
            .table("generated_subjects")
            .select("*")
            .order("created_at", desc=True)
            .limit(safe_limit)
        )

        if mode:
            query = query.eq("usage_mode", mode)

        if matiere:
            query = query.ilike("matiere", f"%{matiere}%")

        if niveau:
            query = query.ilike("niveau", f"%{niveau}%")

        if classe:
            query = query.ilike("classe", f"%{classe}%")

        if theme:
            query = query.ilike("theme", f"%{theme}%")

        result = query.execute()
        rows = result.data or []

        items = []

        for row in rows:
            subject_text = (
                row.get("subject_text")
                or row.get("generated_text")
                or row.get("content")
                or ""
            )

            preview = subject_text.replace("\n", " ").strip()
            if len(preview) > 220:
                preview = preview[:220] + "..."

            items.append({
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "usage_mode": row.get("usage_mode"),
                "matiere": row.get("matiere"),
                "niveau": row.get("niveau"),
                "classe": row.get("classe"),
                "theme": row.get("theme"),
                "type": row.get("type") or row.get("type_demande"),
                "include_correction": row.get("include_correction"),
                "correction_available": bool(row.get("correction_text") or row.get("correction_available")),
                "quality_score": row.get("quality_score"),
                "quality_decision": row.get("quality_decision"),
                "is_reusable": row.get("is_reusable"),
                "preview": preview,
            })

        return {
            "success": True,
            "count": len(items),
            "items": items,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "items": [],
        }


@app.get("/subjects/{subject_id}")
def get_subject_detail(subject_id: int):
    """
    Détail complet d'un sujet généré.
    """
    try:
        import os
        from pathlib import Path
        from supabase import create_client

        env_path = Path(__file__).resolve().parent / ".env"

        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value

        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

        if not supabase_url or not supabase_key:
            return {
                "success": False,
                "error": "Supabase non configuré.",
                "subject": None,
            }

        client = create_client(supabase_url, supabase_key)

        result = (
            client
            .table("generated_subjects")
            .select("*")
            .eq("id", subject_id)
            .single()
            .execute()
        )

        row = result.data

        if not row:
            return {
                "success": False,
                "error": "Sujet introuvable.",
                "subject": None,
            }

        subject_text = (
            row.get("subject_text")
            or row.get("generated_text")
            or row.get("content")
            or ""
        )

        return {
            "success": True,
            "subject": {
                **row,
                "subject_text": subject_text,
                "correction_available": bool(row.get("correction_text") or row.get("correction_available")),
            },
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "subject": None,
        }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ASKCI_SUBJECT_API_HOST", "127.0.0.1")
    port = int(os.getenv("ASKCI_SUBJECT_API_PORT", "8010"))

    uvicorn.run(
        "education_subjects_api:app",
        host=host,
        port=port,
        reload=True,
    )