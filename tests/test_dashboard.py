"""
Testes do painel unificado (`src/dashboard/app.py`), sem datasets e sem nuvem.

O painel não tem lógica própria: é um invólucro que monta as três telas das entregas em
abas. O que pode quebrar, então, é a **montagem** — e ela quebra em silêncio de duas
maneiras que estes testes fixam:

1. O refactor que criou o painel dividiu `build_demo()` em `build_ui()` + `build_demo()`
   nas três apps. Se alguém voltar a montar componentes dentro de `build_demo()`, a app
   autônoma continua funcionando e **só o painel** perde aquela aba.
2. O painel afirma, no rodapé, que as modalidades não se fundem. Essa afirmação é
   verificável: cada aba tem de existir separadamente, e a fronteira entre as fontes de
   dados precisa estar declarada na tela — não apenas no relatório técnico.
"""
import pytest

gr = pytest.importorskip("gradio")

from src.anomaly import app as anomaly_app          # noqa: E402
from src.audio import app as audio_app              # noqa: E402
from src.dashboard import app as dashboard          # noqa: E402
from src.video import app as video_app              # noqa: E402

TODAS_AS_APPS = (video_app, audio_app, anomaly_app)


def test_painel_monta():
    """O painel inteiro precisa montar; um erro em qualquer aba derruba a tela toda."""
    assert isinstance(dashboard.build_demo(), gr.Blocks)


@pytest.mark.parametrize("modulo", TODAS_AS_APPS,
                         ids=lambda m: m.__name__.split(".")[1])
def test_apps_autonomas_continuam_montando(modulo):
    """
    O painel é uma porta de entrada adicional, não um substituto.

    As três apps individuais continuam documentadas no relatório (3.11, 4.9 e 5.8) e no
    roteiro do vídeo, cada uma na sua porta.
    """
    assert isinstance(modulo.build_demo(), gr.Blocks)


@pytest.mark.parametrize("modulo", TODAS_AS_APPS,
                         ids=lambda m: m.__name__.split(".")[1])
def test_cada_app_expoe_build_ui_montavel_em_contexto_externo(modulo):
    """
    `build_ui()` monta no contexto Blocks ativo, sem criar um Blocks próprio.

    É o contrato de que o painel depende. Se alguma app voltar a abrir o próprio
    `gr.Blocks` aqui dentro, os componentes não entram na aba do painel.
    """
    with gr.Blocks() as fora:
        modulo.build_ui()
    assert fora.children, f"{modulo.__name__}.build_ui() não montou nada no contexto"


def test_portas_nao_colidem():
    """As quatro telas precisam poder ficar abertas ao mesmo tempo na demonstração."""
    portas = [audio_app.PORT, anomaly_app.PORT, dashboard.PORT]
    assert len(set(portas)) == len(portas)
    # a app de vídeo usa o padrão do Gradio (7860) e não declara PORT
    assert 7860 not in portas


def test_rodape_declara_a_fronteira_entre_as_fontes():
    """
    SEGURANÇA DE INTERPRETAÇÃO: uma tela única é o lugar mais fácil de sugerir que as
    três modalidades descrevem o mesmo paciente — o que os datasets não sustentam
    (1.4 e 2.2 do relatório). O rodapé precisa dizer isso na própria tela.
    """
    rodape = dashboard.RODAPE
    assert "não há indivíduo em comum" in rodape
    assert "não combina" in rodape
    # a unidade monitorada muda junto com a fonte: paciente nos leitos, sujeito no HAR
    assert "sujeito" in rodape and "paciente" in rodape
    # e as quatro fontes precisam estar nomeadas, não só aludidas
    for fonte in ("REHAB24-6", "figshare", "Challenge 2019", "UCI HAR"):
        assert fonte in rodape, f"fonte não declarada no rodapé: {fonte}"


def test_uma_aba_por_modalidade():
    """Três abas, uma por entrega — nunca uma visão combinada."""
    demo = dashboard.build_demo()
    rotulos = [c.label for c in demo.children
               if isinstance(c, gr.Tabs)
               for c in c.children if isinstance(c, gr.Tab)]
    assert len(rotulos) == 3
    assert any("Vídeo" in r for r in rotulos)
    assert any("Áudio" in r for r in rotulos)
    assert any("Anomalias" in r for r in rotulos)
