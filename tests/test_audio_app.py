"""
Testes da app web da Entrega 2 (`src/audio/app.py`), sem dataset e sem tocar na AWS.

O foco não é o visual: é a **trava de custo**. Vídeo e anomalias rodam local, e um clique
indevido custa tempo de CPU; aqui um clique num caso ainda não processado dispara chamadas
cobradas por volume. Os testes fixam as decisões que, se revertidas, produzem cobrança sem
que nada quebre — que é a falha silenciosa desta entrega.

O cache é redirecionado para um diretório temporário (`monkeypatch`), de modo que os
testes não dependem dos casos já processados no repositório nem os alteram.
"""
import json

import pytest

from src.audio import app, cli, comprehend, transcribe


@pytest.fixture
def cache_vazio(tmp_path, monkeypatch):
    """Aponta os caches de transcrição e de entidades para um diretório limpo."""
    transcricoes = tmp_path / "transcriptions"
    entidades = tmp_path / "entities"
    transcricoes.mkdir()
    entidades.mkdir()
    monkeypatch.setattr(transcribe, "CACHE_DIR", transcricoes)
    monkeypatch.setattr(comprehend, "CACHE_DIR", entidades)
    monkeypatch.setattr(app, "TRANSCRIPT_CACHE_DIR", transcricoes)
    return tmp_path


def _povoar(cache_vazio, case: str) -> None:
    """Cria o cache completo de um caso — nada mais seria cobrado por ele."""
    transcribe.cache_path(case).write_text("{}", encoding="utf-8")
    for sufixo in ("human", "aws"):
        comprehend.cache_path(case, sufixo).write_text("[]", encoding="utf-8")
    comprehend.sentiment_cache_path(case).write_text("{}", encoding="utf-8")


# ----- A trava de custo -----

def test_caso_sem_cache_e_bloqueado_sem_autorizacao(cache_vazio):
    """
    Processar um caso não cacheado sem marcar a caixa deve falhar **antes** de qualquer
    chamada. É a proteção principal da app: sem ela, um clique distraído no seletor de
    272 casos vira cobrança.
    """
    import gradio as gr

    with pytest.raises(gr.Error) as exc:
        app.process("RES9999", allow_paid=False, progress=lambda *a, **k: None)

    # a mensagem precisa dizer o que seria cobrado, senão não orienta a decisão
    assert "Transcribe" in str(exc.value)


def test_caso_em_cache_nao_exige_autorizacao(cache_vazio):
    """Com tudo em cache não há o que cobrar — a caixa não deve ser exigida."""
    _povoar(cache_vazio, "RES0001")
    assert cli._pending("RES0001", compare=True) == []


def test_aviso_de_custo_distingue_os_tres_estados(cache_vazio):
    """O aviso na tela é o `--dry-run` do CLI: precisa dizer o estado antes do clique."""
    _povoar(cache_vazio, "RES0001")
    assert "não" in app.cost_notice("RES0001", allow_paid=False)

    bloqueado = app.cost_notice("RES9999", allow_paid=False)
    liberado = app.cost_notice("RES9999", allow_paid=True)
    assert "Bloqueado" in bloqueado and "Transcribe" in bloqueado
    assert "cobradas" in liberado


def test_aviso_de_custo_usa_a_mesma_regra_do_cli(cache_vazio):
    """
    A app não pode ter a sua própria noção de "o que custa".

    Duplicar a regra é o que faria a tela dizer "sem custo" enquanto a AWS cobra — por
    isso a app importa `_pending` do CLI em vez de reimplementá-la.
    """
    assert app._pending is cli._pending


# ----- Ordenação e rotulagem do seletor -----

def test_seletor_poe_os_casos_em_cache_primeiro(cache_vazio, monkeypatch):
    """
    O caminho barato tem de ser também o caminho óbvio.

    Sem esta ordenação, o primeiro caso do seletor (e portanto o valor inicial, que é o
    que o usuário processa ao clicar sem escolher nada) seria um caso pago.
    """
    import pandas as pd

    monkeypatch.setattr(app, "list_cases", lambda root: pd.DataFrame([
        {"case": "AAA0001", "category": "cardíaco", "has_audio": True, "has_transcript": True},
        {"case": "ZZZ0009", "category": "respiratório", "has_audio": True, "has_transcript": True},
    ]))
    _povoar(cache_vazio, "ZZZ0009")

    escolhas = app._case_choices("qualquer")
    assert [valor for _, valor in escolhas] == ["ZZZ0009", "AAA0001"]
    assert "sem custo" in escolhas[0][0]
    assert "custa" in escolhas[1][0]


# ----- Coerência com o relatório -----

def test_negacao_conflitante_nao_aparece_como_negada(cache_vazio):
    """
    SEGURANÇA CLÍNICA: um termo afirmado numa fala e negado em outra deve permanecer
    entre os achados, com ressalva — nunca na lista de negados.

    Listar a queixa principal como "negada" faria a equipe ler que ela foi descartada. A
    regra é a mesma do `report.py`; se as duas divergirem, a tela e o documento abaixo
    dela passam a dizer coisas diferentes sobre o mesmo paciente.
    """
    entidades = [
        {"text": "chest pain", "category": "MEDICAL_CONDITION", "type": "DX_NAME",
         "score": 0.95, "traits": ["SYMPTOM"], "negated": False, "begin": 0, "end": 10},
        {"text": "chest pain", "category": "MEDICAL_CONDITION", "type": "DX_NAME",
         "score": 0.90, "traits": ["NEGATION"], "negated": True, "begin": 20, "end": 30},
        {"text": "fever", "category": "MEDICAL_CONDITION", "type": "DX_NAME",
         "score": 0.92, "traits": ["NEGATION"], "negated": True, "begin": 40, "end": 45},
    ]
    comprehend.cache_path("RES0001", "human").write_text(
        json.dumps(entidades), encoding="utf-8")

    md = app._findings_md("RES0001", comparacao=None)
    negados = md.split("**Negados pelo paciente** —")[1].split("\n")[0]
    assert "fever" in negados
    assert "chest pain" not in negados
    assert "Afirmados e negados em momentos diferentes" in md


def test_achados_deduplicam_mantendo_a_maior_confianca(cache_vazio):
    """O mesmo termo mencionado várias vezes é um achado, não vários."""
    entidades = [
        {"text": "cough", "category": "MEDICAL_CONDITION", "type": "DX_NAME",
         "score": 0.72, "traits": ["SYMPTOM"], "negated": False, "begin": 0, "end": 5},
        {"text": "Cough", "category": "MEDICAL_CONDITION", "type": "DX_NAME",
         "score": 0.94, "traits": ["SYMPTOM"], "negated": False, "begin": 10, "end": 15},
    ]
    comprehend.cache_path("RES0001", "human").write_text(
        json.dumps(entidades), encoding="utf-8")

    md = app._findings_md("RES0001", comparacao=None)
    assert md.count("| 0.94 |") == 1
    assert "0.72" not in md


def test_entidade_abaixo_do_limiar_fica_fora_dos_achados(cache_vazio):
    """O corte de 0,70 é o mesmo do `clinical_findings()` usado na métrica de recall."""
    entidades = [
        {"text": "arm fracture", "category": "MEDICAL_CONDITION", "type": "DX_NAME",
         "score": 0.60, "traits": ["SYMPTOM"], "negated": False, "begin": 0, "end": 12},
    ]
    comprehend.cache_path("RES0001", "human").write_text(
        json.dumps(entidades), encoding="utf-8")

    md = app._findings_md("RES0001", comparacao=None)
    assert "arm fracture" not in md
    assert "nenhum achado acima do limiar" in md
