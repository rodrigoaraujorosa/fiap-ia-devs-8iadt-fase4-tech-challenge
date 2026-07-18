"""
Extração de entidades clínicas com Amazon Comprehend Medical (Entrega 2).

O serviço recebe texto livre e devolve entidades já **tipadas** (sintoma, anatomia,
medicação, procedimento...), com pontuação de confiança e, o que mais importa
clinicamente, **traços** como ``NEGATION`` — a diferença entre "tem febre" e "não tem
febre" está nesse traço, não no texto extraído.

    transcrição ──► DetectEntitiesV2 ──► entidades tipadas + traços ──► achados clínicos

**Por que rodar sobre as falas do paciente e não sobre a consulta inteira.** As perguntas
do médico ("any fever?", "do you have a cough?") também contêm termos clínicos, mas não
são achados *do paciente* — são hipóteses sendo investigadas. Misturá-las produziria uma
lista de sintomas que o paciente nunca relatou.

**Validação embutida.** Como temos a transcrição humana **e** a da AWS para a mesma
consulta, o módulo extrai entidades das duas e compara. Isso responde à pergunta que o
WER sozinho não responde: os ~5% de erro de transcrição atrapalham a extração clínica?

**Custo.** O Comprehend Medical cobra por caractere processado. Todo resultado é cacheado
em ``reports/entities/`` e não se reprocessa sem ``--force``.

Uso:
    python -m src.audio.comprehend --cases RES0029            # extrai do texto do paciente
    python -m src.audio.comprehend --cases RES0029 --compare  # humano vs AWS
    python -m src.audio.comprehend --report                   # consolida o que está em cache
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..common.config import ROOT_DIR, get_aws_config
from .consultations import patient_text
from .transcribe import cache_path as transcript_cache_path
from .transcribe import patient_text_from_aws

CACHE_DIR = ROOT_DIR / "reports" / "entities"

# Limite de bytes UTF-8 por chamada do DetectEntitiesV2. O valor real é definido pela AWS;
# usamos uma margem folgada para não depender de um número que pode mudar — e o texto de
# um paciente (até ~1.300 palavras) cabe com sobra numa única chamada.
MAX_BYTES = 18_000

# Categorias devolvidas pelo serviço, com o rótulo em pt-BR para os relatórios.
CATEGORY_LABELS_PT = {
    "MEDICAL_CONDITION": "condição médica",
    "ANATOMY": "anatomia",
    "MEDICATION": "medicação",
    "TEST_TREATMENT_PROCEDURE": "exame/procedimento",
    "TIME_EXPRESSION": "expressão temporal",
    "BEHAVIORAL_ENVIRONMENTAL_SOCIAL": "comportamental/social",
    "PROTECTED_HEALTH_INFORMATION": "dado pessoal",
}

# Traços que qualificam a entidade. NEGATION é o mais importante: inverte o sentido
# clínico do achado, e ignorá-lo produziria relatório com sintomas que o paciente negou.
TRAIT_LABELS_PT = {
    "NEGATION": "negado",
    "DIAGNOSIS": "diagnóstico",
    "SIGN": "sinal",
    "SYMPTOM": "sintoma",
    "HYPOTHETICAL": "hipotético",
    "LOW_CONFIDENCE": "baixa confiança",
    "PERTAINS_TO_FAMILY": "história familiar",
}


def _split_text(text: str, max_bytes: int = MAX_BYTES) -> list[str]:
    """
    Divide o texto em blocos que caibam numa chamada, cortando em fim de frase.

    Cortar no meio de uma frase quebraria a detecção de negação — "no chest pain" só é
    reconhecido como negado se o "no" e o "chest pain" estiverem no mesmo bloco.
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        candidate = f"{current} {sentence}".strip()
        if len(candidate.encode("utf-8")) > max_bytes and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def detect_entities(text: str, region: str) -> list[dict]:
    """
    Chama o DetectEntitiesV2 e devolve as entidades encontradas.

    Os offsets de cada bloco são corrigidos para o texto completo, senão as posições de
    blocos posteriores apontariam para o lugar errado.
    """
    import boto3

    client = boto3.client("comprehendmedical", region_name=region)
    entities: list[dict] = []
    offset = 0

    for chunk in _split_text(text):
        response = client.detect_entities_v2(Text=chunk)
        for e in response.get("Entities", []):
            e["BeginOffset"] += offset
            e["EndOffset"] += offset
            entities.append(e)
        offset += len(chunk) + 1
    return entities


def simplify(entities: list[dict]) -> list[dict]:
    """Reduz a resposta da AWS ao que interessa para o relatório."""
    simple = []
    for e in entities:
        traits = [t["Name"] for t in e.get("Traits", [])]
        simple.append({
            "text": e["Text"],
            "category": e["Category"],
            "type": e["Type"],
            "score": round(e["Score"], 4),
            "traits": traits,
            "negated": "NEGATION" in traits,
            "begin": e["BeginOffset"],
            "end": e["EndOffset"],
        })
    return simple


def clinical_findings(entities: list[dict], min_score: float = 0.7) -> list[dict]:
    """
    Filtra os achados clínicos afirmados pelo paciente.

    Descarta: entidades abaixo do limiar de confiança, as **negadas** (o paciente disse
    que *não* tem), as hipotéticas e as que se referem a familiares — nenhuma delas é um
    achado do paciente no momento da consulta.
    """
    out = []
    for e in entities:
        if e["score"] < min_score:
            continue
        if e["negated"] or "HYPOTHETICAL" in e["traits"] or "PERTAINS_TO_FAMILY" in e["traits"]:
            continue
        if e["category"] != "MEDICAL_CONDITION":
            continue
        out.append(e)
    return out


def cache_path(case: str, source: str) -> Path:
    """``source`` distingue a origem do texto: 'human' ou 'aws'."""
    return CACHE_DIR / f"{case}__{source}.json"


def extract(case: str, root: str | Path, region: str, source: str = "human",
            force: bool = False) -> list[dict]:
    """
    Extrai entidades de um caso, a partir da referência humana ou da transcrição da AWS.

    Nos dois casos usa-se **apenas a fala do paciente**: na origem humana pelos rótulos
    ``D:``/``P:`` do dataset, na origem AWS pela diarização do Transcribe. Comparar fala
    do paciente contra consulta inteira inflaria a origem AWS com as perguntas do médico,
    que não são achados do paciente.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(case, source)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    if source == "human":
        text = patient_text(root, case)
    elif source == "aws":
        tpath = transcript_cache_path(case)
        if not tpath.exists():
            raise FileNotFoundError(
                f"sem transcrição da AWS para {case} — rode: "
                f"python -m src.audio.transcribe --cases {case}")
        text = patient_text_from_aws(json.loads(tpath.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"source inválido: {source}")

    if not text.strip():
        raise ValueError(f"texto vazio para {case} (source={source})")

    entities = simplify(detect_entities(text, region))
    path.write_text(json.dumps(entities, ensure_ascii=False, indent=1), encoding="utf-8")
    return entities


def compare_sources(case: str, root: str | Path, region: str, force: bool = False) -> dict:
    """
    Extrai das duas origens e mede o quanto a transcrição automática preservou os achados.

    A recuperação é medida sobre o **conjunto de termos** (em minúsculas), não sobre a
    contagem: para o relatório clínico interessa se o achado foi detectado, não quantas
    vezes o paciente o mencionou.
    """
    human = extract(case, root, region, source="human", force=force)
    aws = extract(case, root, region, source="aws", force=force)

    h_terms = {e["text"].lower() for e in clinical_findings(human)}
    a_terms = {e["text"].lower() for e in clinical_findings(aws)}

    recovered = h_terms & a_terms
    return {
        "case": case,
        "human_entities": len(human),
        "aws_entities": len(aws),
        "human_findings": len(h_terms),
        "aws_findings": len(a_terms),
        "recovered": len(recovered),
        "recall": round(len(recovered) / len(h_terms), 4) if h_terms else float("nan"),
        "missed": sorted(h_terms - a_terms),
        "extra": sorted(a_terms - h_terms),
    }


def _print_entities(case: str, entities: list[dict]) -> None:
    print(f"\n=== {case}: {len(entities)} entidades ===")
    by_cat: dict[str, list[dict]] = {}
    for e in entities:
        by_cat.setdefault(e["category"], []).append(e)

    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        rotulo = CATEGORY_LABELS_PT.get(cat, cat)
        print(f"\n{cat} ({rotulo}) — {len(items)}")
        vistos = set()
        for e in sorted(items, key=lambda x: -x["score"]):
            chave = e["text"].lower()
            if chave in vistos:
                continue
            vistos.add(chave)
            marcas = "".join(f" [{TRAIT_LABELS_PT.get(t, t)}]" for t in e["traits"])
            print(f"  {e['score']:.2f}  {e['text']:28s} {e['type']}{marcas}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extração de entidades clínicas com Amazon Comprehend Medical.")
    ap.add_argument("--root", default="data/audio/consultas", help="raiz do dataset")
    ap.add_argument("--cases", nargs="+", help="casos a processar (ex.: RES0029)")
    ap.add_argument("--source", choices=("human", "aws"), default="human",
                    help="origem do texto: transcrição humana ou do Transcribe")
    ap.add_argument("--compare", action="store_true",
                    help="extrai das duas origens e mede a recuperação dos achados")
    ap.add_argument("--force", action="store_true",
                    help="reprocessa mesmo com cache (custa dinheiro de novo)")
    ap.add_argument("--report", action="store_true",
                    help="consolida o que já está em cache, sem chamar a AWS")
    ap.add_argument("--out", help="salva o resumo em CSV")
    args = ap.parse_args()

    cfg = get_aws_config()

    if args.report:
        cases = sorted({p.stem.split("__")[0] for p in CACHE_DIR.glob("*.json")})
        if not cases:
            print("nenhuma extração em cache — rode com --cases primeiro")
            return
        for case in cases:
            for source in ("human", "aws"):
                p = cache_path(case, source)
                if p.exists():
                    _print_entities(f"{case} ({source})",
                                    json.loads(p.read_text(encoding="utf-8")))
        return

    if not args.cases:
        ap.print_help()
        return

    if not cfg["region"]:
        print("AWS não configurada. Rode: python -m src.common.config")
        return

    if args.compare:
        rows = [compare_sources(c, args.root, cfg["region"], force=args.force)
                for c in args.cases]
        import pandas as pd
        df = pd.DataFrame(rows)
        print(df.drop(columns=["missed", "extra"]).to_string(index=False))
        for r in rows:
            if r["missed"]:
                print(f"\n[{r['case']}] achados perdidos pela AWS: {', '.join(r['missed'])}")
            if r["extra"]:
                print(f"[{r['case']}] achados só na AWS: {', '.join(r['extra'])}")
        if args.out:
            df.drop(columns=["missed", "extra"]).to_csv(args.out, index=False)
            print(f"\nsalvo em {args.out}")
        return

    for case in args.cases:
        entities = extract(case, args.root, cfg["region"], source=args.source,
                           force=args.force)
        _print_entities(f"{case} ({args.source})", entities)
        findings = clinical_findings(entities)
        print(f"\nachados clínicos afirmados (score>=0.7, não negados): {len(findings)}")
        print("  " + ", ".join(sorted({f['text'].lower() for f in findings})))


if __name__ == "__main__":
    main()
