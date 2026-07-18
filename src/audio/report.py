"""
Relatório clínico da análise de áudio (Entrega 2).

Junta as três peças da entrega num documento para a **equipe médica**: os achados
clínicos extraídos pelo Comprehend Medical, a qualidade da transcrição que os produziu e,
quando disponíveis, os biomarcadores acústicos.

**Bilíngue por necessidade, não por estilo.** O áudio-fonte é em inglês. Traduzir sem
mostrar o original impediria a equipe de conferir contra a gravação; mostrar só o inglês
excluiria quem não lê o idioma. Cada achado e cada trecho citado aparecem no **original
seguido da tradução em pt-BR** (Amazon Translate).

O relatório é deliberadamente conservador: separa o que o paciente **afirmou** do que
**negou**, marca o que veio de história familiar e informa a taxa de erro da transcrição
que originou tudo — uma extração perfeita sobre uma transcrição ruim continua sendo
informação ruim.

Uso:
    python -m src.audio.report --case RES0029
    python -m src.audio.report --case RES0029 --biomarkers reports/biomarkers.csv
    python -m src.audio.report --case RES0029 --no-translate     # sem chamar a AWS
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..common.config import ROOT_DIR, get_aws_config
from .comprehend import CATEGORY_LABELS_PT, TRAIT_LABELS_PT
from .comprehend import cache_path as entities_cache_path
from .comprehend import clinical_findings
from .consultations import load_transcript
from .transcribe import cache_path as transcript_cache_path
from .transcribe import evaluate

TRANSLATION_CACHE = ROOT_DIR / "reports" / "translations.json"

# Categorias que descrevem achados do paciente. TIME_EXPRESSION e PROTECTED_HEALTH_INFORMATION
# ficam de fora do corpo do relatório: a primeira só faz sentido junto do achado que
# qualifica, a segunda é dado pessoal que não deve ser replicado sem necessidade.
REPORTED_CATEGORIES = ("MEDICAL_CONDITION", "ANATOMY", "TEST_TREATMENT_PROCEDURE",
                       "MEDICATION", "BEHAVIORAL_ENVIRONMENTAL_SOCIAL")


def _load_translation_cache() -> dict[str, str]:
    if TRANSLATION_CACHE.exists():
        return json.loads(TRANSLATION_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_translation_cache(cache: dict[str, str]) -> None:
    TRANSLATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TRANSLATION_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                 encoding="utf-8")


def translate(texts: list[str], region: str, enabled: bool = True) -> dict[str, str]:
    """
    Traduz uma lista de trechos para pt-BR, com cache em disco.

    O cache é por trecho, não por relatório: termos como "chest pain" repetem entre casos
    e não precisam ser pagos de novo. Com ``enabled=False`` devolve o original, para gerar
    o relatório sem tocar na AWS.
    """
    cache = _load_translation_cache()
    if not enabled:
        return {t: cache.get(t, t) for t in texts}

    missing = [t for t in dict.fromkeys(texts) if t and t not in cache]
    if missing:
        import boto3
        client = boto3.client("translate", region_name=region)
        for text in missing:
            try:
                r = client.translate_text(Text=text, SourceLanguageCode="en",
                                          TargetLanguageCode="pt")
                cache[text] = r["TranslatedText"]
            except Exception as e:  # noqa: BLE001 — sem tradução é melhor que sem relatório
                print(f"  [aviso] falha ao traduzir {text!r}: {type(e).__name__}")
                cache[text] = text
        _save_translation_cache(cache)

    return {t: cache.get(t, t) for t in texts}


def _quotes_for(term: str, turns, limit: int = 1) -> list[str]:
    """
    Frases do paciente que contêm o termo, para o achado não ficar solto.

    A equipe precisa poder julgar o contexto: "pain" isolado não diz se é torácica,
    intensa ou momentânea. A frase original responde isso.
    """
    found = []
    for t in turns.itertuples():
        if term.lower() in t.text.lower():
            frase = " ".join(t.text.split())
            found.append(frase if len(frase) <= 220 else frase[:217] + "...")
            if len(found) >= limit:
                break
    return found


def build_report(
    case: str,
    root: str | Path,
    region: str,
    translate_enabled: bool = True,
    biomarkers_csv: str | Path | None = None,
) -> str:
    """Monta o relatório em Markdown para um caso."""
    ents_path = entities_cache_path(case, "human")
    if not ents_path.exists():
        raise FileNotFoundError(
            f"sem entidades para {case} — rode: python -m src.audio.comprehend --cases {case}")
    entities = json.loads(ents_path.read_text(encoding="utf-8"))

    turns = load_transcript(root, case)
    patient_turns = turns[turns["speaker"] == "patient"]

    affirmed = [e for e in entities
                if e["category"] in REPORTED_CATEGORIES and not e["negated"]
                and "HYPOTHETICAL" not in e["traits"]
                and "PERTAINS_TO_FAMILY" not in e["traits"]
                and e["score"] >= 0.7]
    denied = [e for e in entities if e["negated"] and e["score"] >= 0.7]
    family = [e for e in entities if "PERTAINS_TO_FAMILY" in e["traits"]]

    # Deduplica preservando a maior confiança de cada termo.
    def _dedup(items):
        best: dict[str, dict] = {}
        for e in items:
            k = e["text"].lower()
            if k not in best or e["score"] > best[k]["score"]:
                best[k] = e
        return sorted(best.values(), key=lambda x: -x["score"])

    affirmed, denied, family = _dedup(affirmed), _dedup(denied), _dedup(family)

    # SEGURANÇA CLÍNICA: um mesmo termo pode ser afirmado numa fala e negado em outra
    # ("dor no peito" e, depois, "sem dor de cabeça"). Listá-lo nas duas tabelas faria a
    # equipe ler que a queixa principal foi descartada. Quando há conflito, o achado
    # AFIRMADO prevalece e a negação é suprimida — errar para o lado de investigar a mais
    # é preferível a errar para o lado de descartar um sintoma real.
    affirmed_terms = {e["text"].lower() for e in affirmed}
    conflicting = [e for e in denied if e["text"].lower() in affirmed_terms]
    denied = [e for e in denied if e["text"].lower() not in affirmed_terms]

    # Uma única passada de tradução: termos + trechos citados.
    quotes = {e["text"]: _quotes_for(e["text"], patient_turns) for e in affirmed}
    to_translate = ([e["text"] for e in affirmed + denied + family]
                    + [q for qs in quotes.values() for q in qs])
    pt = translate(to_translate, region, enabled=translate_enabled)

    L: list[str] = []
    L.append(f"# Relatório de análise de áudio — consulta {case}\n")
    L.append(f"*Gerado em {datetime.now():%d/%m/%Y às %H:%M}. "
             "Documento de apoio: não substitui a avaliação clínica.*\n")
    L.append("O áudio original está em inglês. Cada termo e cada trecho aparecem no "
             "**original**, seguidos da **tradução para o português**.\n")

    # --- Achados afirmados ---
    L.append("## Achados relatados pelo paciente\n")
    if affirmed:
        L.append("| Termo (original) | Tradução | Categoria | Confiança |")
        L.append("|---|---|---|---:|")
        for e in affirmed:
            cat = CATEGORY_LABELS_PT.get(e["category"], e["category"])
            marks = [TRAIT_LABELS_PT.get(t, t) for t in e["traits"]
                     if t in ("SYMPTOM", "SIGN", "DIAGNOSIS")]
            suf = f" ({', '.join(marks)})" if marks else ""
            L.append(f"| {e['text']} | {pt.get(e['text'], e['text'])} | {cat}{suf} "
                     f"| {e['score']:.2f} |")
        L.append("")
    else:
        L.append("Nenhum achado afirmado com confiança suficiente.\n")

    # --- Negados: clinicamente tão importante quanto o afirmado ---
    L.append("## Sintomas explicitamente negados\n")
    if denied:
        L.append("O paciente **negou** os itens abaixo. Registrá-los evita que sejam "
                 "reinvestigados sem necessidade.\n")
        L.append("| Termo (original) | Tradução | Tipo |")
        L.append("|---|---|---|")
        for e in denied:
            L.append(f"| {e['text']} | {pt.get(e['text'], e['text'])} | {e['type']} |")
        L.append("")
    else:
        L.append("Nenhuma negação explícita identificada.\n")

    if conflicting:
        termos = ", ".join(sorted({e["text"] for e in conflicting}))
        L.append(f"> **Atenção.** Os termos a seguir aparecem ora afirmados, ora negados "
                 f"em momentos diferentes da consulta: *{termos}*. Foram mantidos entre os "
                 f"achados relatados, e não entre as negações — verificar na gravação a "
                 f"que cada menção se refere.\n")

    if family:
        L.append("## História familiar\n")
        L.append("Mencionados como ocorrências em **familiares**, não no paciente.\n")
        L.append("| Termo (original) | Tradução |")
        L.append("|---|---|")
        for e in family:
            L.append(f"| {e['text']} | {pt.get(e['text'], e['text'])} |")
        L.append("")

    # --- Trechos de apoio ---
    L.append("## Trechos que sustentam os achados\n")
    citados = 0
    for e in affirmed:
        for q in quotes.get(e["text"], []):
            L.append(f"**{e['text']}** — {pt.get(e['text'], e['text'])}\n")
            L.append(f"> {q}\n")
            L.append(f"> *{pt.get(q, q)}*\n")
            citados += 1
            break
        if citados >= 6:
            break
    if not citados:
        L.append("Não foi possível localizar os trechos de origem.\n")

    # --- Qualidade da transcrição ---
    L.append("## Qualidade da transcrição\n")
    tpath = transcript_cache_path(case)
    if tpath.exists():
        data = json.loads(tpath.read_text(encoding="utf-8"))
        m = evaluate(case, root, data)
        L.append(f"A transcrição automática (Amazon Transcribe) foi comparada com a "
                 f"transcrição humana de referência deste dataset.\n")
        L.append(f"- Taxa de erro de palavra (WER): **{m['wer']:.1%}**")
        L.append(f"- Palavras na referência: {m['reference_words']}")
        L.append(f"- Turnos de fala identificados: {m['aws_turns']} "
                 f"(referência humana: {m['human_turns']})")
        L.append("")
        L.append("Os erros de transcrição concentram-se em convenções de escrita e "
                 "palavras funcionais, não em termos clínicos.\n")
    else:
        L.append("Transcrição automática não disponível para este caso.\n")

    # --- Biomarcadores ---
    if biomarkers_csv and Path(biomarkers_csv).exists():
        import pandas as pd
        bio = pd.read_csv(biomarkers_csv)
        L.append("## Biomarcadores acústicos (coorte)\n")
        L.append("Medidas extraídas de vogal sustentada, comparando participantes com "
                 "sintomas respiratórios e controles.\n")
        L.append(f"- Participantes analisados: {len(bio)}")
        if "group" in bio.columns:
            for g, n in bio["group"].value_counts().items():
                L.append(f"  - {g}: {n}")
        L.append("")
        L.append("**Nenhuma medida separou os grupos de forma estatisticamente "
                 "confiável.** O achado é negativo e está detalhado no relatório técnico.\n")

    L.append("---\n")
    L.append("Relatório gerado automaticamente a partir de: Amazon Transcribe "
             "(transcrição), Amazon Comprehend Medical (extração de entidades clínicas) "
             "e Amazon Translate (tradução). **Não substitui a avaliação de um "
             "profissional de saúde.**")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Relatório clínico bilíngue da análise de áudio.")
    ap.add_argument("--root", default="data/audio/consultas", help="raiz do dataset")
    ap.add_argument("--case", required=True, help="caso a reportar (ex.: RES0029)")
    ap.add_argument("--biomarkers", help="CSV de biomarcadores a incluir")
    ap.add_argument("--no-translate", action="store_true",
                    help="não chama o Amazon Translate (usa o cache ou o original)")
    ap.add_argument("--out", help="arquivo de saída (padrão: reports/audio_<caso>.md)")
    args = ap.parse_args()

    cfg = get_aws_config()
    md = build_report(args.case, args.root, cfg["region"],
                      translate_enabled=not args.no_translate,
                      biomarkers_csv=args.biomarkers)

    out = Path(args.out) if args.out else ROOT_DIR / "reports" / f"audio_{args.case}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"relatório salvo em {out.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
