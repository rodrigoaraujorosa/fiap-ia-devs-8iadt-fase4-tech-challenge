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


def check_aws(verbose: bool = True) -> bool:
    """
    Verifica se o ambiente AWS está pronto para a Entrega 2.

    Confere, nesta ordem: variáveis do ``.env``, credencial resolvível, autenticação
    (STS ``get_caller_identity``), acesso ao bucket S3 e existência dos endpoints do
    Transcribe e do Comprehend Medical na região escolhida.

    Existe para que o erro apareça **antes** de um job de transcrição, e não no meio
    dele — e para que quem for reproduzir o trabalho saiba em qual passo parou. Nunca
    imprime credenciais; do identificador da conta mostra apenas os 4 últimos dígitos.
    """
    def _diz(msg: str) -> None:
        if verbose:
            print(msg)

    cfg = get_aws_config()
    ok = True

    _diz("Verificação do ambiente AWS")
    _diz(f"  região             : {cfg['region'] or '(ausente — defina AWS_REGION no .env)'}")
    _diz(f"  bucket             : {cfg['s3_bucket'] or '(ausente — defina AWS_S3_BUCKET no .env)'}")
    if not cfg["region"] or not cfg["s3_bucket"]:
        _diz("  -> faltam variáveis no .env (veja .env.example)")
        return False

    if not aws_is_configured():
        _diz("  credencial         : NÃO encontrada")
        _diz("  -> rode `aws configure` ou defina AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
        return False
    _diz("  credencial         : encontrada")

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        _diz("  boto3              : NÃO instalado (pip install -r requirements.txt)")
        return False

    try:
        ident = boto3.client("sts", region_name=cfg["region"]).get_caller_identity()
        _diz(f"  autenticação (STS) : OK (conta ...{ident['Account'][-4:]})")
    except (ClientError, BotoCoreError) as e:
        _diz(f"  autenticação (STS) : FALHOU — {type(e).__name__}")
        _diz("  -> credenciais inválidas ou expiradas; rode `aws configure` de novo")
        return False

    try:
        boto3.client("s3", region_name=cfg["region"]).head_bucket(Bucket=cfg["s3_bucket"])
        _diz("  bucket S3          : acessível")
    except ClientError as e:
        codigo = e.response["Error"]["Code"]
        _diz(f"  bucket S3          : FALHOU — {codigo}")
        if codigo in ("404", "NoSuchBucket"):
            _diz("  -> o bucket não existe. Atenção: o console da AWS costuma acrescentar")
            _diz("     um sufixo (id da conta + região) ao nome digitado na criação —")
            _diz("     use o nome completo, como aparece em `aws s3 ls`.")
        elif codigo == "403":
            _diz("  -> bucket existe mas o usuário não tem permissão (falta política de S3)")
        ok = False

    for servico, rotulo in (("transcribe", "Transcribe"), ("comprehendmedical", "Comprehend Medical")):
        try:
            boto3.client(servico, region_name=cfg["region"])
            _diz(f"  {rotulo:19s}: disponível em {cfg['region']}")
        except Exception:  # noqa: BLE001 — serviço indisponível na região escolhida
            _diz(f"  {rotulo:19s}: INDISPONÍVEL em {cfg['region']}")
            ok = False

    _diz("\nambiente pronto." if ok else "\nambiente incompleto — veja os pontos acima.")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if check_aws() else 1)
