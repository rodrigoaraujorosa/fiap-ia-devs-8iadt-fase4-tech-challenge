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

Módulo de biblioteca — o ponto de entrada é ``src.audio.cli``:

    python -m src.audio.cli --case RES0091          # pipeline completo
    python -m src.audio.cli --show-entities RES0091 # inspeciona as entidades em cache
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..common.config import ROOT_DIR
from .cache import cached_json
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

    # >>> CHAMADA AWS: cliente do Amazon Comprehend Medical
    client = boto3.client("comprehendmedical", region_name=region)
    entities: list[dict] = []
    offset = 0

    for chunk in _split_text(text):
        # >>> CHAMADA AWS: Comprehend Medical / DetectEntitiesV2 — entidades clínicas
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

    Duas limitações conhecidas deste filtro:

    - **O limiar cria efeito de degrau.** No RES0091, "arm fracture" foi extraída das duas
      transcrições, mas com 0,82 na humana e 0,60 na da AWS; só a primeira passa. A
      comparação entre origens (:func:`compare_sources`) mede, portanto, uma mistura de
      qualidade de transcrição e sensibilidade ao corte.
    - **Não há distinção entre passado e presente.** O Comprehend Medical marca família e
      hipótese, mas não histórico: "I had an arm fracture when I was younger" entra como
      achado atual. Separar isso exigiria analisar o tempo verbal ou cruzar com as
      entidades de ``TIME_EXPRESSION``.
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


# ----- Sentimento (Amazon Comprehend, o serviço geral) -----
#
# A divisão da AWS difere da Azure: o Text Analytics reunia sentimento, frases-chave e
# entidades de saúde num serviço só; na AWS o **Comprehend Medical** faz as entidades
# clínicas e o **Comprehend** (geral) faz o sentimento. São dois clientes distintos.

SENTIMENT_LABELS_PT = {
    "POSITIVE": "positivo",
    "NEGATIVE": "negativo",
    "NEUTRAL": "neutro",
    "MIXED": "misto",
}

# Limite de bytes por chamada do DetectSentiment, com folga.
SENTIMENT_MAX_BYTES = 4_500


def sentiment_cache_path(case: str) -> Path:
    return CACHE_DIR / f"{case}__sentiment.json"


def detect_sentiment(text: str, region: str) -> dict:
    """
    Sentimento predominante do texto e a pontuação de cada classe.

    Textos longos são divididos e as pontuações, ponderadas pelo tamanho de cada bloco —
    a média simples daria a um trecho de 20 palavras o mesmo peso de um de 400.
    """
    import boto3

    # >>> CHAMADA AWS: cliente do Amazon Comprehend (serviço geral, não o Medical)
    client = boto3.client("comprehend", region_name=region)
    chunks = _split_text(text, max_bytes=SENTIMENT_MAX_BYTES)

    total = 0
    acc = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0, "Mixed": 0.0}
    for chunk in chunks:
        # >>> CHAMADA AWS: Amazon Comprehend / DetectSentiment — tom geral do relato
        r = client.detect_sentiment(Text=chunk, LanguageCode="en")
        peso = len(chunk)
        total += peso
        for k, v in r["SentimentScore"].items():
            acc[k] += v * peso

    scores = {k: round(v / total, 4) for k, v in acc.items()} if total else acc
    predominante = max(scores, key=scores.get).upper()
    return {"sentiment": predominante, "scores": scores, "chunks": len(chunks)}


def sentiment_by_turn(turns: list[dict], region: str, max_turns: int = 25) -> list[dict]:
    """
    Sentimento de cada fala do paciente, para localizar onde o relato é mais negativo.

    Usa ``BatchDetectSentiment`` (até 25 documentos por chamada), o que reduz o número de
    requisições. Turnos muito curtos ("yeah", "no") são descartados: não carregam
    sentimento avaliável e só empurrariam a média para NEUTRAL.
    """
    import boto3

    uteis = [t for t in turns if len(t["text"].split()) >= 8][:max_turns]
    if not uteis:
        return []

    client = boto3.client("comprehend", region_name=region)
    # >>> CHAMADA AWS: Amazon Comprehend / BatchDetectSentiment — sentimento por turno
    r = client.batch_detect_sentiment(
        TextList=[t["text"][:SENTIMENT_MAX_BYTES] for t in uteis], LanguageCode="en")

    out = []
    for item in r.get("ResultList", []):
        t = uteis[item["Index"]]
        out.append({
            "order": t.get("order"),
            "text": t["text"],
            "sentiment": item["Sentiment"],
            "negative": round(item["SentimentScore"]["Negative"], 4),
        })
    return out


def analyze_sentiment(case: str, root: str | Path, region: str,
                      force: bool = False) -> dict:
    """Sentimento do relato do paciente (geral e por turno), com cache."""
    def _compute() -> dict:
        from .consultations import load_transcript

        text = patient_text(root, case)
        if not text.strip():
            raise ValueError(f"texto vazio para {case}")

        df = load_transcript(root, case)
        turns = df[df["speaker"] == "patient"].to_dict("records")

        result = detect_sentiment(text, region)
        result["by_turn"] = sentiment_by_turn(turns, region)
        return result

    return cached_json(sentiment_cache_path(case), _compute, force=force, indent=1)


def extract(case: str, root: str | Path, region: str, source: str = "human",
            force: bool = False) -> list[dict]:
    """
    Extrai entidades de um caso, a partir da referência humana ou da transcrição da AWS.

    Nos dois casos usa-se **apenas a fala do paciente**: na origem humana pelos rótulos
    ``D:``/``P:`` do dataset, na origem AWS pela diarização do Transcribe. Comparar fala
    do paciente contra consulta inteira inflaria a origem AWS com as perguntas do médico,
    que não são achados do paciente.
    """
    # A leitura do texto fica DENTRO da função de cálculo: para ``source='aws'`` ela
    # depende da transcrição em cache, e um caso já extraído não deve falhar só porque
    # a transcrição foi apagada depois.
    def _compute() -> list[dict]:
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
        return simplify(detect_entities(text, region))

    return cached_json(cache_path(case, source), _compute, force=force, indent=1)


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
