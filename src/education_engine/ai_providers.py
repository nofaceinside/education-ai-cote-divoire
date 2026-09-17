import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


ProviderName = Literal["mistral", "openai", "claude", "none"]

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    """
    Charge .env sans dépendre obligatoirement de python-dotenv.
    Respecte les variables déjà présentes dans os.environ.
    """
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


load_env_file()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return str(value).strip().lower() in ["1", "true", "yes", "on"]


def env_provider(name: str, default: ProviderName = "none") -> ProviderName:
    value = str(os.getenv(name, default)).strip().lower()

    if value in ["mistral", "openai", "claude", "none"]:
        return value  # type: ignore

    return default


@dataclass
class AIProviderConfig:
    mistral_enabled: bool
    openai_enabled: bool
    claude_enabled: bool
    default_provider: ProviderName
    fallback_provider: ProviderName
    openai_generation_enabled: bool
    openai_quality_enabled: bool
    claude_generation_enabled: bool
    claude_quality_enabled: bool
    mistral_key_present: bool
    openai_key_present: bool
    anthropic_key_present: bool


def get_ai_provider_config() -> AIProviderConfig:
    return AIProviderConfig(
        mistral_enabled=env_bool("EDUCATION_AI_MISTRAL_ENABLED", True),
        openai_enabled=env_bool("ASKCI_OPENAI_ENABLED", True),
        claude_enabled=env_bool("ASKCI_CLAUDE_ENABLED", False),
        default_provider=env_provider("EDUCATION_AI_DEFAULT_PROVIDER", "mistral"),
        fallback_provider=env_provider("EDUCATION_AI_FALLBACK_PROVIDER", "openai"),
        openai_generation_enabled=env_bool("ASKCI_OPENAI_GENERATION_ENABLED", True),
        openai_quality_enabled=env_bool("ASKCI_OPENAI_QUALITY_ENABLED", True),
        claude_generation_enabled=env_bool("ASKCI_CLAUDE_GENERATION_ENABLED", False),
        claude_quality_enabled=env_bool("ASKCI_CLAUDE_QUALITY_ENABLED", False),
        mistral_key_present=bool(os.getenv("MISTRAL_API_KEY", "").strip()),
        openai_key_present=bool(os.getenv("OPENAI_API_KEY", "").strip()),
        anthropic_key_present=bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
    )


def is_provider_available(provider: ProviderName, task: Literal["generation", "quality"]) -> bool:
    config = get_ai_provider_config()

    if provider == "none":
        return False

    if provider == "mistral":
        return config.mistral_enabled and config.mistral_key_present

    if provider == "openai":
        if not config.openai_enabled or not config.openai_key_present:
            return False

        if task == "generation":
            return config.openai_generation_enabled

        if task == "quality":
            return config.openai_quality_enabled

    if provider == "claude":
        if not config.claude_enabled or not config.anthropic_key_present:
            return False

        if task == "generation":
            return config.claude_generation_enabled

        if task == "quality":
            return config.claude_quality_enabled

    return False


def choose_provider(task: Literal["generation", "quality"]) -> ProviderName:
    config = get_ai_provider_config()

    if is_provider_available(config.default_provider, task):
        return config.default_provider

    if is_provider_available(config.fallback_provider, task):
        return config.fallback_provider

    if is_provider_available("mistral", task):
        return "mistral"

    if is_provider_available("openai", task):
        return "openai"

    if is_provider_available("claude", task):
        return "claude"

    return "none"


def provider_status() -> dict:
    config = get_ai_provider_config()

    return {
        "openai_enabled": config.openai_enabled,
        "claude_enabled": config.claude_enabled,
        "default_provider": config.default_provider,
        "fallback_provider": config.fallback_provider,
        "openai_key_present": config.openai_key_present,
        "anthropic_key_present": config.anthropic_key_present,
        "openai_generation_available": is_provider_available("openai", "generation"),
        "openai_quality_available": is_provider_available("openai", "quality"),
        "claude_generation_available": is_provider_available("claude", "generation"),
        "claude_quality_available": is_provider_available("claude", "quality"),
        "selected_generation_provider": choose_provider("generation"),
        "selected_quality_provider": choose_provider("quality"),
    }


def assert_provider_available(task: Literal["generation", "quality"]) -> ProviderName:
    provider = choose_provider(task)

    if provider == "none":
        raise RuntimeError(
            f"Aucun fournisseur IA disponible pour la tâche : {task}. "
            "Vérifie ASKCI_OPENAI_ENABLED, ASKCI_CLAUDE_ENABLED et les clés API."
        )

    return provider
