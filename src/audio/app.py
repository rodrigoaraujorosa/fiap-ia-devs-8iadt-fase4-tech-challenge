"""
App web (Gradio) para demonstração local da Entrega 2 — Análise de Áudio.

Expõe o mesmo pipeline do ``cli.py`` — Transcribe, Comprehend Medical, Comprehend e
Translate — numa tela para a **equipe médica**: ouvir a consulta, ver os achados
clínicos extraídos da fala do paciente e ler o relatório bilíngue.

Como nas Entregas 1 e 3, a app **não reimplementa nada**: chama as mesmas funções de
biblioteca que o CLI usa, na mesma ordem.

**A diferença desta app para as outras duas: aqui o processamento custa dinheiro.**
Vídeo e anomalias rodam local, e apertar o botão sem querer custa tempo de CPU. Aqui um
clique num caso ainda não processado dispara chamadas cobradas por volume ao Transcribe,
ao Comprehend Medical e ao Translate. Por isso:

- o seletor mostra, em cada caso, se ele já está **em cache** (sem custo) ou não;
- a chamada paga é **bloqueada por padrão** e exige marcar explicitamente a caixa;
- antes de processar, a tela lista **quais etapas** fariam chamada paga — é o
  ``--dry-run`` do CLI embutido na interface.

Com os casos em cache a app roda **sem credenciais da AWS**, o que permite reproduzir a
demonstração sem conta na nuvem.

Uso:
    python -m src.audio.app        # abre em http://localhost:7862
"""
from __future__ import annotations

import json
import time

import gradio as gr

from ..common.config import ROOT_DIR, get_aws_config
from ..video.report import fmt_dur
from .cache import require_aws
# _pending é importada do CLI de propósito: é a regra que decide o que ainda seria
# cobrado, e mantê-la em um lugar só é o que evita que um esquecimento aqui vire
# cobrança indevida (mesma razão pela qual o cache mora em cache.py).
from .cli import _pending
from .comprehend import SENTIMENT_LABELS_PT, TRAIT_LABELS_PT
from .comprehend import analyze_sentiment, clinical_findings, compare_sources, extract
from .comprehend import cache_path as entities_cache_path
from .consultations import audio_path, list_cases
from .report import build_report
from .report import cache_path_for as report_path_for
from .transcribe import CACHE_DIR as TRANSCRIPT_CACHE_DIR
from .transcribe import cache_path as transcript_cache_path
from .transcribe import process as run_transcribe

DEFAULT_ROOT = "data/audio/consultas"

# 7860 é o app da Entrega 1 e 7861 o da Entrega 3 — os três podem ficar abertos lado a
# lado durante a demonstração.
PORT = 7862

AVISO = """
---
**Uso acadêmico.** Os achados são extraídos automaticamente da fala do paciente e **não
constituem diagnóstico**. O sentimento vem de um modelo de propósito geral: num relato de
sintomas o resultado negativo é o esperado e, isoladamente, informa pouco.
"""


def _is_cached(case: str) -> bool:
    """True se a transcrição do caso já está em disco (nenhuma chamada paga)."""
    return transcript_cache_path(case).exists()


def _case_choices(root: str) -> list[tuple[str, str]]:
    """
    Casos disponíveis, com os já processados no topo e o custo declarado no rótulo.

    A ordem não é estética: são 272 casos e apenas alguns estão em cache. Deixar os
    gratuitos primeiro é o que faz o caminho barato ser também o caminho óbvio.
    """
    try:
        cases = list_cases(root)
    except (FileNotFoundError, OSError):
        cases = None

    if cases is None or cases.empty:
        # Sem o dataset em disco ainda dá para demonstrar o que está em cache.
        return [(f"{c} — em cache (sem custo)", c)
                for c in sorted(p.stem for p in TRANSCRIPT_CACHE_DIR.glob("*.json"))]

    disponiveis = cases[cases["has_audio"] & cases["has_transcript"]]
    rotulos = []
    for row in disponiveis.itertuples():
        cached = _is_cached(row.case)
        marca = "em cache (sem custo)" if cached else "não processado (custa)"
        rotulos.append((f"{row.case} — {row.category} · {marca}", row.case, cached))
    # ordena: em cache primeiro, depois alfabético
    rotulos.sort(key=lambda r: (not r[2], r[1]))
    return [(label, value) for label, value, _ in rotulos]


def cost_notice(case: str | None, allow_paid: bool) -> str:
    """
    Diz o que aconteceria ao processar este caso — o ``--dry-run`` do CLI, na tela.

    Roda a cada troca no seletor, para que a informação de custo apareça **antes** do
    clique, e não depois.
    """
    if not case:
        return ""
    pendentes = _pending(case, compare=True)
    if not pendentes:
        return ("✅ **Tudo em cache.** Processar este caso **não** faz nenhuma chamada "
                "paga à AWS — os resultados são lidos do disco.")

    lista = ", ".join(f"`{p}`" for p in pendentes)
    if allow_paid:
        return (f"💳 **Chamadas pagas liberadas.** Ao processar, estas etapas serão "
                f"cobradas: {lista}. O relatório também paga a tradução dos trechos "
                f"ainda não traduzidos.")
    return (f"🔒 **Bloqueado.** Este caso faria chamada paga em: {lista}. "
            f"Marque *Permitir chamadas pagas à AWS* para prosseguir, ou escolha um "
            f"caso em cache.")


def _metrics_md(m: dict | None) -> str:
    """Qualidade da transcrição — é o contexto que calibra a confiança nos achados."""
    if not m:
        return "_(métricas de transcrição indisponíveis)_"
    return (
        "### 1. Amazon Transcribe — qualidade da transcrição\n\n"
        f"| Métrica | Valor |\n|:--|--:|\n"
        f"| WER (erro de palavra) | **{m['wer']:.2%}** |\n"
        f"| Substituições / inserções / remoções | {m['substitutions']} / "
        f"{m['insertions']} / {m['deletions']} |\n"
        f"| Palavras na referência humana | {m['reference_words']} |\n"
        f"| Turnos (AWS / humano) | {m['aws_turns']} / {m['human_turns']} |\n\n"
        "_O WER é medido contra a transcrição humana do próprio dataset. As hesitações "
        "('um', 'uh') ficam fora dos dois lados: o anotador as transcreveu e o Transcribe "
        "as omite, então contá-las mediria a convenção de anotação, não o reconhecimento._"
    )


def _findings_md(case: str, comparacao: dict | None) -> str:
    """Achados afirmados, negados e de história familiar, com a recall entre origens."""
    path = entities_cache_path(case, "human")
    if not path.exists():
        return "_(entidades indisponíveis)_"
    entidades = json.loads(path.read_text(encoding="utf-8"))

    def _unicos(itens: list[dict]) -> list[dict]:
        """Um item por termo, guardando a maior confiança — como faz o relatório."""
        melhor: dict[str, dict] = {}
        for e in itens:
            chave = e["text"].lower()
            if chave not in melhor or e["score"] > melhor[chave]["score"]:
                melhor[chave] = e
        return sorted(melhor.values(), key=lambda x: -x["score"])

    achados = _unicos(clinical_findings(entidades))
    negados = _unicos([e for e in entidades if e["negated"] and e["score"] >= 0.7])
    familia = _unicos([e for e in entidades if "PERTAINS_TO_FAMILY" in e["traits"]])

    # Mesma regra de segurança clínica do relatório (report.py): um termo pode ser
    # afirmado numa fala e negado em outra ("dor no peito" e, depois, "sem dor de
    # cabeça"). Listá-lo nas duas tabelas faria a equipe ler que a queixa principal foi
    # descartada — o achado afirmado prevalece e a negação vira ressalva.
    afirmados = {e["text"].lower() for e in achados}
    conflitantes = [e for e in negados if e["text"].lower() in afirmados]
    negados = [e for e in negados if e["text"].lower() not in afirmados]

    L = ["### 2. Amazon Comprehend Medical — achados clínicos", ""]
    L.append(f"{len(entidades)} entidades extraídas **apenas da fala do paciente** — as "
             "perguntas do médico contêm termos clínicos, mas são hipóteses sendo "
             "investigadas, não achados dele.")
    L.append("")

    L.append("**Relatados pelo paciente**")
    L.append("")
    if achados:
        L.append("| Achado | Confiança |")
        L.append("|:--|--:|")
        L += [f"| {e['text']} | {e['score']:.2f} |" for e in achados]
    else:
        L.append("_(nenhum achado acima do limiar de confiança de 0,70)_")
    L.append("")
    L.append("_Este quadro lista as **condições médicas** afirmadas. O relatório completo, "
             "abaixo, é mais amplo: inclui também anatomia, exames e medicações._")
    L.append("")

    if negados:
        marcados = ", ".join(f"`{e['text']}`" for e in negados)
        L.append(f"**Negados pelo paciente** — {marcados}")
        L.append("")
    if conflitantes:
        marcados = ", ".join(f"`{e['text']}`" for e in conflitantes)
        L.append(f"⚠️ **Afirmados e negados em momentos diferentes** — {marcados}. "
                 "Mantidos entre os achados; conferir na gravação a que cada menção "
                 "se refere.")
        L.append("")
    if familia:
        marcados = ", ".join(f"`{e['text']}`" for e in familia)
        L.append(f"**História familiar** *(do parente, não do paciente)* — {marcados}")
        L.append("")

    L.append("_Os traços "
             f"({', '.join(f'`{TRAIT_LABELS_PT[t]}`' for t in ('NEGATION', 'PERTAINS_TO_FAMILY', 'HYPOTHETICAL'))})"
             " são o que separa 'tem febre' de 'não tem febre'. Ignorá-los produziria um "
             "relatório listando como sintoma o que o paciente negou._")

    if comparacao and comparacao.get("recall") == comparacao.get("recall"):  # não-NaN
        L.append("")
        L.append(f"**Transcrição humana × transcrição da AWS** — dos "
                 f"{comparacao['human_findings']} achados extraídos da referência humana, "
                 f"{comparacao['recovered']} foram recuperados a partir da transcrição "
                 f"automática (**recall {comparacao['recall']:.1%}**).")
        if comparacao["missed"]:
            L.append("")
            L.append(f"Não recuperados: {', '.join(f'`{t}`' for t in comparacao['missed'])} "
                     "— parte deles está presente na extração, apenas abaixo do corte de "
                     "confiança de 0,70.")
    return "\n".join(L)


def _sentiment_md(s: dict | None) -> str:
    """Tom do relato, geral e por turno — o turno é o que localiza a informação."""
    if not s:
        return "_(sentimento indisponível)_"
    rotulo = SENTIMENT_LABELS_PT.get(s["sentiment"], s["sentiment"])
    L = ["### 3. Amazon Comprehend — sentimento do relato", ""]
    L.append(f"Tom geral: **{s['sentiment']}** ({rotulo}) — "
             f"{s['scores'].get('Negative', 0):.0%} negativo, "
             f"{s['scores'].get('Neutral', 0):.0%} neutro.")
    L.append("")

    piores = sorted(s.get("by_turn", []), key=lambda t: -t["negative"])[:3]
    if piores:
        L.append("**Falas de maior carga negativa** — é aqui que o indicador informa "
                 "alguma coisa, muito mais que no rótulo agregado:")
        L.append("")
        for t in piores:
            trecho = " ".join(t["text"].split())
            if len(trecho) > 160:
                trecho = trecho[:160].rsplit(" ", 1)[0] + "..."
            L.append(f"- *\"{trecho}\"* — {t['negative']:.2f} negativo")
    return "\n".join(L)


def process(case: str, allow_paid: bool, root: str = DEFAULT_ROOT,
            progress=gr.Progress()):
    """
    Roda as quatro etapas e devolve (áudio, métricas, achados, sentimento, relatório).

    A ordem é a mesma do ``cli.py``, porque é a ordem em que as etapas dependem umas das
    outras: sem transcrição não há texto para extrair, sem entidades não há relatório.
    """
    if not case:
        raise gr.Error("Selecione uma consulta.")

    pendentes = _pending(case, compare=True)
    if pendentes and not allow_paid:
        raise gr.Error(
            f"O caso {case} exige chamada paga em: {', '.join(pendentes)}. "
            "Marque 'Permitir chamadas pagas à AWS' ou escolha um caso em cache.")

    # Sem etapa pendente nada é cobrado, e aí nem credencial é necessária: as funções
    # leem do cache. É o que permite reproduzir a demonstração sem conta na AWS.
    region, bucket = "", ""
    if pendentes:
        cfg = get_aws_config()
        erro = require_aws(cfg, need_bucket=True)
        if erro:
            raise gr.Error(erro)
        region, bucket = cfg["region"], cfg["s3_bucket"]

    timings: dict[str, float] = {}

    # 1. Transcrição (Amazon Transcribe)
    progress(0.05, desc="Amazon Transcribe — transcrevendo a consulta...")
    t0 = time.perf_counter()
    try:
        metricas = run_transcribe([case], root, bucket, region, log=lambda *_: None)
    except Exception as e:  # noqa: BLE001 — demo: mostra a causa em vez de quebrar
        raise gr.Error(f"Transcrição indisponível: {type(e).__name__} — {e}") from e
    timings["1. Transcrição"] = time.perf_counter() - t0
    m = metricas[0] if metricas else None

    # 2. Entidades clínicas (Amazon Comprehend Medical), nas duas origens
    progress(0.45, desc="Amazon Comprehend Medical — extraindo achados clínicos...")
    t0 = time.perf_counter()
    try:
        comparacao = compare_sources(case, root, region)
    except Exception:  # noqa: BLE001 — sem a origem AWS ainda dá para mostrar a humana
        extract(case, root, region, source="human")
        comparacao = None
    timings["2. Entidades"] = time.perf_counter() - t0

    # 3. Sentimento (Amazon Comprehend, o serviço geral)
    progress(0.7, desc="Amazon Comprehend — analisando o sentimento do relato...")
    t0 = time.perf_counter()
    try:
        sentimento = analyze_sentiment(case, root, region)
    except Exception:  # noqa: BLE001
        sentimento = None
    timings["3. Sentimento"] = time.perf_counter() - t0

    # 4. Relatório bilíngue (Amazon Translate). Sem liberação de custo, a tradução sai
    # do cache: `enabled=False` devolve o original para os trechos ainda não traduzidos.
    progress(0.85, desc="Amazon Translate — montando o relatório bilíngue...")
    t0 = time.perf_counter()
    md = build_report(case, root, region, translate_enabled=bool(pendentes or allow_paid))
    out = report_path_for(case)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    timings["4. Relatório"] = time.perf_counter() - t0
    timings["Total"] = sum(timings.values())

    tempos = " · ".join(f"{etapa} {fmt_dur(seg)}" for etapa, seg in timings.items())
    cabecalho = (f"## Consulta {case}\n\n"
                 f"_{'Nenhuma chamada paga: tudo veio do cache' if not pendentes else 'Processado na AWS'}. "
                 f"Tempo por etapa — {tempos}._\n\n"
                 f"Relatório salvo em `{out.relative_to(ROOT_DIR).as_posix()}`.")

    return (str(audio_path(root, case)), cabecalho, _metrics_md(m),
            _findings_md(case, comparacao), _sentiment_md(sentimento), md)


def build_ui(root: str = DEFAULT_ROOT) -> None:
    """
    Monta os componentes no contexto ``gr.Blocks`` ativo.

    Separada de :func:`build_demo` para que a mesma tela sirva à app autônoma
    (porta 7862) e à aba de áudio do painel unificado (`src/dashboard/app.py`),
    sem duplicar a montagem — em especial a trava de custo, que uma segunda
    cópia poderia perder de vista.
    """
    escolhas = _case_choices(root)
    inicial = escolhas[0][1] if escolhas else None

    gr.Markdown(
        "# 🎙️ Análise de Áudio de Consultas (AWS)\n"
        "Transcreve a consulta com o **Amazon Transcribe**, extrai os achados "
        "clínicos da fala do paciente com o **Comprehend Medical**, mede o tom do "
        "relato com o **Comprehend** e monta um relatório bilíngue com o "
        "**Translate**."
    )

    with gr.Row():
        case_dd = gr.Dropdown(escolhas, value=inicial, label="Consulta",
                              info="Casos em cache aparecem primeiro e não geram custo")
        allow_paid = gr.Checkbox(
            value=False, label="Permitir chamadas pagas à AWS",
            info="Necessário apenas para casos ainda não processados")
    custo = gr.Markdown(cost_notice(inicial, False))
    run_btn = gr.Button("▶ Processar consulta", variant="primary")

    resumo = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Áudio da consulta** — a gravação original, para conferir "
                        "os achados contra o que foi dito.")
            # interactive=False: sem isso o Gradio monta o componente no modo de
            # entrada, com botão de gravação e pedido de acesso ao microfone — aqui
            # ele só reproduz o arquivo do dataset.
            player = gr.Audio(label="Consulta (MP3)", type="filepath",
                              interactive=False)
            metricas = gr.Markdown()
        with gr.Column():
            achados = gr.Markdown()
            sentimento = gr.Markdown()

    gr.Markdown("---\n### 4. Relatório clínico bilíngue\n"
                "Cada termo aparece **no original em inglês seguido da tradução**: o "
                "áudio-fonte é em inglês, e traduzir sem mostrar o original impediria "
                "a equipe de conferir contra a gravação.")
    relatorio = gr.Markdown()
    gr.Markdown(AVISO)

    case_dd.change(cost_notice, [case_dd, allow_paid], [custo])
    allow_paid.change(cost_notice, [case_dd, allow_paid], [custo])
    # show_progress_on: sem isso o Gradio desenha uma barra em cada output, inclusive
    # nos Markdown, que não têm altura fixa — mesma razão da app da Entrega 1.
    run_btn.click(process, [case_dd, allow_paid],
                  [player, resumo, metricas, achados, sentimento, relatorio],
                  show_progress_on=[player])


def build_demo(root: str = DEFAULT_ROOT) -> gr.Blocks:
    """App autônoma da Entrega 2 (porta 7862)."""
    with gr.Blocks(title="Análise de Áudio de Consultas — Entrega 2") as demo:
        build_ui(root)
    return demo


def main() -> None:
    build_demo().launch(server_port=PORT)


if __name__ == "__main__":
    main()
