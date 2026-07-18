"""
Configuração compartilhada do projeto.

Centraliza caminhos e o carregamento de credenciais (AWS) a partir do .env,
para que os módulos das 3 entregas não repitam essa lógica.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # carrega .env da raiz, se existir
except ImportError:  # python-dotenv é opcional para os módulos que não usam a nuvem
    pass

# ----- Caminhos do projeto -----
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_VIDEO_DIR = DATA_DIR / "video"
DATA_AUDIO_DIR = DATA_DIR / "audio"
DATA_ANOMALY_DIR = DATA_DIR / "anomaly"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ----- Credenciais AWS (Entrega 2) -----
# O boto3 já resolve credenciais por conta própria (variáveis de ambiente, ~/.aws/credentials,
# perfil, role). Estas funções existem para o .env da raiz funcionar como as demais entregas
# e para o código poder avisar cedo quando falta configuração, em vez de estourar na chamada.
def get_aws_config() -> dict[str, str | None]:
    """Retorna região e perfil da AWS a partir do ambiente."""
    return {
        "region": os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        "profile": os.getenv("AWS_PROFILE"),
        "s3_bucket": os.getenv("AWS_S3_BUCKET"),
    }


def aws_is_configured() -> bool:
    """True se há indício de credencial da AWS disponível (env, perfil ou arquivo)."""
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True
    if os.getenv("AWS_PROFILE"):
        return True
    return (Path.home() / ".aws" / "credentials").exists()
