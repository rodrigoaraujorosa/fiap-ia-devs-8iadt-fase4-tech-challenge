"""
Configuração compartilhada do projeto.

Centraliza caminhos e o carregamento de credenciais (Azure) a partir do .env,
para que os módulos das 3 entregas não repitam essa lógica.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # carrega .env da raiz, se existir
except ImportError:  # python-dotenv é opcional para os módulos que não usam Azure
    pass

# ----- Caminhos do projeto -----
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_VIDEO_DIR = DATA_DIR / "video"
DATA_AUDIO_DIR = DATA_DIR / "audio"
DATA_ANOMALY_DIR = DATA_DIR / "anomaly"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ----- Credenciais Azure (Entrega 2) -----
def get_azure_speech_config() -> dict[str, str | None]:
    """Retorna key/region do Azure Speech-to-Text a partir do ambiente."""
    return {
        "key": os.getenv("AZURE_SPEECH_KEY"),
        "region": os.getenv("AZURE_SPEECH_REGION"),
    }


def get_azure_language_config() -> dict[str, str | None]:
    """Retorna key/endpoint do Azure Text Analytics a partir do ambiente."""
    return {
        "key": os.getenv("AZURE_LANGUAGE_KEY"),
        "endpoint": os.getenv("AZURE_LANGUAGE_ENDPOINT"),
    }
