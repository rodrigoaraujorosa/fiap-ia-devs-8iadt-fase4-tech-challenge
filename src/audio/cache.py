"""
Cache em disco dos resultados da nuvem (Entrega 2 — Análise de Áudio).

Transcribe, Comprehend Medical, Comprehend e Translate cobram por volume processado.
Todo resultado é gravado em JSON e reaproveitado nas execuções seguintes, de modo que
reprocessar um caso exija um pedido explícito (``force=True``).

Este módulo existe porque o padrão "se há cache, lê; senão, chama a nuvem e grava"
aparecia repetido em três lugares — transcrição, entidades e sentimento — e a versão
duplicada é onde um esquecimento vira cobrança indevida.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..common.config import ROOT_DIR

CACHE_ROOT = ROOT_DIR / "reports"


def cached_json(path: Path, compute: Callable[[], object], force: bool = False,
                indent: int | None = None) -> object:
    """
    Devolve o conteúdo de ``path``; se não existir (ou ``force``), chama ``compute``.

    ``compute`` só é executada quando necessário — é ela que faz a chamada paga. O
    resultado é gravado antes de retornar, para que uma interrupção posterior não
    obrigue a pagar de novo.
    """
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=indent),
                    encoding="utf-8")
    return result


def is_cached(path: Path) -> bool:
    """True se o resultado já está em disco (nenhuma chamada paga seria feita)."""
    return path.exists()


def require_aws(cfg: dict, need_bucket: bool = False) -> str | None:
    """
    Valida a configuração da AWS e devolve a mensagem de erro, ou ``None`` se estiver ok.

    Centraliza a checagem que estava repetida nos módulos, para a mensagem de orientação
    ser sempre a mesma.
    """
    if not cfg.get("region"):
        return "AWS não configurada (região ausente). Rode: python -m src.common.config"
    if need_bucket and not cfg.get("s3_bucket"):
        return "Bucket S3 não configurado. Rode: python -m src.common.config"
    return None
