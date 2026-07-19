"""
Pipeline fim-a-fim da Entrega 2 (Análise de Áudio).

Roda as quatro etapas em um comando, na ordem em que dependem umas das outras:

    1. Transcribe          áudio -> S3 -> transcrição + diarização (+ WER)
    2. Comprehend Medical  fala do paciente -> entidades clínicas tipadas
    3. Comprehend          fala do paciente -> sentimento (geral e por turno)
    4. Relatório           tudo acima -> documento bilíngue para a equipe médica

Este é o **único ponto de entrada** da entrega: ``transcribe.py``, ``comprehend.py`` e
``report.py`` são módulos de biblioteca, sem CLI própria — mesma organização da Entrega 1,
onde só ``cli.py`` e ``app.py`` são executáveis.

Uso:
    python -m src.audio.cli --case RES0091
    python -m src.audio.cli --cases RES0091 RES0142 RES0094     # lote
    python -m src.audio.cli --case RES0091 --dry-run            # o que seria cobrado

Modos que **não chamam a nuvem** (úteis para iterar sem custo):
    python -m src.audio.cli --report                  # recalcula o WER do que está em cache
    python -m src.audio.cli --show-entities RES0091   # inspeciona as entidades extraídas

**Custo.** As etapas 1 a 3 e a tradução da 4 são pagas por volume. Nenhuma delas
reprocessa um caso já em cache — para isso é preciso ``--force``, e o ``--dry-run``
mostra antes o que seria efetivamente chamado.
"""
from __future__ import annotations

import argparse
import time

from ..common.config import get_aws_config
from ..video.report import fmt_dur
from .cache import is_cached, require_aws
from .comprehend import analyze_sentiment
from .comprehend import cache_path as entities_cache_path
from .comprehend import compare_sources, extract, sentiment_cache_path
from .report import build_report
from .report import cache_path_for as report_path_for
from .transcribe import cache_path as transcript_cache_path
from .transcribe import process as run_transcribe


def _pending(case: str, compare: bool) -> list[str]:
    """Etapas que ainda fariam chamada paga para este caso."""
    todo = []
    if not is_cached(transcript_cache_path(case)):
        todo.append("Transcribe")
    if not is_cached(entities_cache_path(case, "human")):
        todo.append("Comprehend Medical (humano)")
    if compare and not is_cached(entities_cache_path(case, "aws")):
        todo.append("Comprehend Medical (AWS)")
    if not is_cached(sentiment_cache_path(case)):
        todo.append("Comprehend (sentimento)")
    return todo


def run_case(
    case: str,
    root: str,
    region: str,
    bucket: str,
    force: bool = False,
    compare: bool = True,
    translate_enabled: bool = True,
    keep_fillers: bool = False,
) -> dict:
    """Executa as quatro etapas para um caso e devolve um resumo."""
    timings: dict[str, float] = {}
    resumo: dict = {"case": case}

    # 1. Transcrição
    print(f"\n[1/4] {case} — transcrição (Amazon Transcribe)")
    t0 = time.perf_counter()
    metricas = run_transcribe([case], root, bucket, region, force=force,
                              keep_fillers=keep_fillers)
    timings["1. Transcrição"] = time.perf_counter() - t0
    if metricas:
        m = metricas[0]
        resumo["wer"] = m["wer"]
        print(f"      WER {m['wer']:.2%} | {m['reference_words']} palavras de referência"
              f" | {m['aws_turns']} turnos identificados")

    # 2. Entidades clínicas
    print(f"\n[2/4] {case} — entidades clínicas (Comprehend Medical)")
    t0 = time.perf_counter()
    if compare:
        c = compare_sources(case, root, region, force=force)
        resumo.update({"achados": c["human_findings"], "recall": c["recall"]})
        print(f"      {c['human_entities']} entidades | {c['human_findings']} achados"
              f" | recall humano→AWS {c['recall']:.1%}")
        if c["missed"]:
            print(f"      não recuperados pela AWS: {', '.join(c['missed'])}")
    else:
        ents = extract(case, root, region, source="human", force=force)
        resumo["achados"] = len(ents)
        print(f"      {len(ents)} entidades extraídas")
    timings["2. Entidades"] = time.perf_counter() - t0

    # 3. Sentimento
    print(f"\n[3/4] {case} — sentimento (Amazon Comprehend)")
    t0 = time.perf_counter()
    s = analyze_sentiment(case, root, region, force=force)
    resumo["sentimento"] = s["sentiment"]
    timings["3. Sentimento"] = time.perf_counter() - t0
    print(f"      {s['sentiment']} ({s['scores'].get('Negative', 0):.0%} negativo)"
          f" | {len(s.get('by_turn', []))} turnos analisados")

    # 4. Relatório
    print(f"\n[4/4] {case} — relatório clínico bilíngue")
    t0 = time.perf_counter()
    md = build_report(case, root, region, translate_enabled=translate_enabled)
    out = report_path_for(case)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    timings["4. Relatório"] = time.perf_counter() - t0
    resumo["relatorio"] = str(out)
    print(f"      salvo em {out}")

    timings["Total"] = sum(v for k, v in timings.items() if k != "Total")
    print("\n      tempo por etapa:")
    for etapa, seg in timings.items():
        print(f"        {etapa:16s} {fmt_dur(seg)}")
    return resumo


def _report_from_cache(root: str, out: str | None, keep_fillers: bool = False) -> None:
    """
    Recalcula as métricas de tudo o que já foi transcrito, **sem chamar a AWS**.

    Permite iterar sobre a forma de medir (por exemplo, ligar e desligar a contagem de
    hesitações) sem pagar de novo pela transcrição.
    """
    from .transcribe import CACHE_DIR as TRANSCRIPT_DIR

    cases = sorted(p.stem for p in TRANSCRIPT_DIR.glob("*.json"))
    if not cases:
        print("nenhuma transcrição em cache — rode antes: python -m src.audio.cli --case <CASO>")
        return

    import pandas as pd
    metricas = run_transcribe(cases, root, bucket="", region="", keep_fillers=keep_fillers)
    df = pd.DataFrame(metricas)
    print(df.to_string(index=False))
    if len(df) > 1:
        print(f"\nWER: média {df['wer'].mean():.2%} | mediana {df['wer'].median():.2%}"
              f" | desvio {df['wer'].std():.2%}")
    if out:
        df.to_csv(out, index=False)
        print(f"\nsalvo em {out}")


def _show_entities(case: str) -> None:
    """Lista as entidades já extraídas de um caso, agrupadas por categoria."""
    import json

    from .comprehend import CATEGORY_LABELS_PT, TRAIT_LABELS_PT

    path = entities_cache_path(case, "human")
    if not path.exists():
        print(f"sem entidades para {case} — rode: python -m src.audio.cli --case {case}")
        return

    entities = json.loads(path.read_text(encoding="utf-8"))
    print(f"{case}: {len(entities)} entidades\n")
    por_categoria: dict[str, list[dict]] = {}
    for e in entities:
        por_categoria.setdefault(e["category"], []).append(e)

    for cat, itens in sorted(por_categoria.items(), key=lambda kv: -len(kv[1])):
        print(f"{cat} ({CATEGORY_LABELS_PT.get(cat, cat)}) — {len(itens)}")
        vistos = set()
        for e in sorted(itens, key=lambda x: -x["score"]):
            chave = e["text"].lower()
            if chave in vistos:
                continue
            vistos.add(chave)
            marcas = "".join(f" [{TRAIT_LABELS_PT.get(t, t)}]" for t in e["traits"])
            print(f"  {e['score']:.2f}  {e['text']:28s} {e['type']}{marcas}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pipeline fim-a-fim da análise de áudio (transcrição -> relatório).")
    ap.add_argument("--root", default="data/audio/consultas", help="raiz do dataset")
    grupo = ap.add_mutually_exclusive_group()
    grupo.add_argument("--case", help="um caso (ex.: RES0091)")
    grupo.add_argument("--cases", nargs="+", help="vários casos")
    ap.add_argument("--force", action="store_true",
                    help="reprocessa tudo, mesmo com cache (cobra de novo)")
    ap.add_argument("--no-compare", action="store_true",
                    help="não extrai entidades da transcrição da AWS (metade do custo na etapa 2)")
    ap.add_argument("--no-translate", action="store_true",
                    help="gera o relatório sem chamar o Amazon Translate")
    ap.add_argument("--keep-fillers", action="store_true",
                    help="conta hesitações no WER")
    ap.add_argument("--dry-run", action="store_true",
                    help="lista o que seria chamado na nuvem, sem executar nada")
    ap.add_argument("--report", action="store_true",
                    help="recalcula as métricas do que está em cache, sem chamar a AWS")
    ap.add_argument("--show-entities", metavar="CASO",
                    help="lista as entidades já extraídas de um caso (não chama a AWS)")
    ap.add_argument("--out", help="CSV com o resumo dos casos processados")
    args = ap.parse_args()

    if not (args.case or args.cases or args.report or args.show_entities):
        ap.error("informe --case, --cases, --report ou --show-entities")

    cases = args.cases or [args.case]
    cfg = get_aws_config()

    if args.report:
        _report_from_cache(args.root, args.out, keep_fillers=args.keep_fillers)
        return

    if args.show_entities:
        _show_entities(args.show_entities)
        return

    if args.dry_run:
        print("Etapas que fariam chamada paga:\n")
        for case in cases:
            todo = _pending(case, compare=not args.no_compare)
            if args.force:
                todo = ["Transcribe", "Comprehend Medical", "Comprehend (sentimento)"]
            print(f"  {case}: {', '.join(todo) if todo else 'nada — tudo em cache'}")
        print("\n(a tradução do relatório também é cobrada, por trecho ainda não traduzido)")
        return

    erro = require_aws(cfg, need_bucket=True)
    if erro:
        print(erro)
        return

    resumos = []
    for case in cases:
        try:
            resumos.append(run_case(
                case, args.root, cfg["region"], cfg["s3_bucket"],
                force=args.force, compare=not args.no_compare,
                translate_enabled=not args.no_translate,
                keep_fillers=args.keep_fillers,
            ))
        except Exception as e:  # noqa: BLE001 — um caso ruim não deve abortar o lote
            print(f"\n[erro] {case}: {type(e).__name__} — {e}")

    if len(resumos) > 1:
        import pandas as pd
        df = pd.DataFrame(resumos)
        print(f"\n{'=' * 60}\nResumo de {len(df)} casos\n")
        print(df.to_string(index=False))
        if "wer" in df.columns:
            print(f"\nWER: média {df['wer'].mean():.2%} | mediana {df['wer'].median():.2%}"
                  f" | desvio {df['wer'].std():.2%}")
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nresumo salvo em {args.out}")


if __name__ == "__main__":
    main()
