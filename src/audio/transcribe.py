"""
Transcrição de consultas médicas com Amazon Transcribe (Entrega 2).

Fluxo, porque o Transcribe **não aceita o áudio na chamada**: o arquivo é enviado ao S3,
inicia-se um job assíncrono apontando para a URI, espera-se a conclusão e busca-se o JSON
do resultado.

    MP3 local ──► S3 ──► StartTranscriptionJob ──► (polling) ──► JSON ──► texto + WER

Como o dataset traz **transcrição humana revisada**, o resultado não é só exibido: é
medido. O WER (Word Error Rate) contra a referência humana dá um número defensável para o
relatório, em vez de uma captura de tela.

**Custo.** O Transcribe cobra por minuto de áudio processado e cada consulta tem 11-15 min.
Por isso o módulo *sempre* cacheia em ``reports/transcriptions/<case>.json`` e nunca
reprocessa um caso já transcrito, a menos que se peça ``--force``.

Módulo de biblioteca — o ponto de entrada é ``src.audio.cli``:

    python -m src.audio.cli --case RES0091          # pipeline completo
    python -m src.audio.cli --report                # WER do que está em cache
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..common.config import ROOT_DIR
from .consultations import audio_path, full_text, load_transcript

CACHE_DIR = ROOT_DIR / "reports" / "transcriptions"
S3_PREFIX = "consultations"

# Marcadores de hesitação. O anotador humano transcreveu "um", "uh", "ahh"; o Transcribe
# normalmente os omite. Contá-los como erro mede a convenção de anotação, não a qualidade
# do reconhecimento — por isso são removíveis dos dois lados (ver --keep-fillers).
FILLERS = {
    "um", "uh", "uhh", "umm", "ah", "ahh", "ahhh", "hmm", "mmm", "mhm",
    "er", "erm", "eh", "oh", "hm",
}


# ----- Normalização e WER -----

def normalize(text: str, drop_fillers: bool = True) -> list[str]:
    """
    Reduz o texto a uma lista de palavras comparável entre referência e hipótese.

    Minúsculas, sem pontuação e com espaços colapsados — sem isso o WER mediria
    diferenças de estilo de anotação (vírgulas, maiúsculas) em vez de reconhecimento.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)   # mantém apóstrofo: "i'm" != "im"
    words = text.split()
    if drop_fillers:
        words = [w for w in words if w not in FILLERS]
    return words


def wer(reference: list[str], hypothesis: list[str]) -> dict:
    """
    Word Error Rate entre duas listas de palavras, com o detalhe dos erros.

    WER = (substituições + inserções + deleções) / palavras da referência, obtido por
    programação dinâmica (distância de edição em nível de palavra). Devolve também cada
    tipo de erro, porque eles têm causas diferentes: deleção costuma ser fala sobreposta,
    inserção costuma ser ruído interpretado como palavra.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return {"wer": float("nan"), "sub": 0, "ins": m, "del": 0, "ref_words": 0}

    # d[i][j] = custo mínimo para alinhar reference[:i] com hypothesis[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,         # deleção
                          d[i][j - 1] + 1,         # inserção
                          d[i - 1][j - 1] + cost)  # substituição/acerto

    # Retrocede pelo caminho ótimo para separar os tipos de erro.
    i, j, sub, ins, dele = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dele += 1
            i -= 1

    return {"wer": (sub + ins + dele) / n, "sub": sub, "ins": ins, "del": dele,
            "ref_words": n}


# ----- AWS -----

def _clients(region: str):
    import boto3
    return boto3.client("s3", region_name=region), boto3.client("transcribe", region_name=region)


def upload_to_s3(case: str, root: str | Path, bucket: str, region: str) -> str:
    """Envia o MP3 do caso para o S3 (se ainda não estiver lá) e devolve a URI."""
    from botocore.exceptions import ClientError

    s3, _ = _clients(region)
    key = f"{S3_PREFIX}/{case}.mp3"
    try:
        s3.head_object(Bucket=bucket, Key=key)   # já enviado numa execução anterior
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        s3.upload_file(str(audio_path(root, case)), bucket, key)
    return f"s3://{bucket}/{key}"


def transcribe_case(
    case: str,
    root: str | Path,
    bucket: str,
    region: str,
    language: str = "en-US",
    speakers: int = 2,
    poll_interval: float = 15.0,
    timeout: float = 1800.0,
    progress=None,
) -> dict:
    """
    Roda o job do Transcribe para um caso e devolve o resultado bruto da AWS.

    ``speakers=2`` liga a diarização: a consulta é um diálogo médico-paciente, e separar
    os interlocutores permite comparar com os turnos ``D:``/``P:`` da referência humana.

    O nome do job carrega o instante de início porque a AWS **não permite reaproveitar o
    nome de um job existente**, e um job com falha deixaria o nome ocupado.
    """
    import urllib.request

    _, tr = _clients(region)
    uri = upload_to_s3(case, root, bucket, region)
    job_name = f"consultation-{case}-{int(time.time())}"

    tr.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": uri},
        MediaFormat="mp3",
        LanguageCode=language,
        Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": speakers},
    )

    t0 = time.perf_counter()
    while True:
        job = tr.get_transcription_job(TranscriptionJobName=job_name)["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            break
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"job {job_name} passou de {timeout:.0f}s ainda em {status}")
        if progress:
            progress(case, status, time.perf_counter() - t0)
        time.sleep(poll_interval)

    if status == "FAILED":
        raise RuntimeError(f"job {job_name} falhou: {job.get('FailureReason', '?')}")

    url = job["Transcript"]["TranscriptFileUri"]
    with urllib.request.urlopen(url) as r:  # noqa: S310 — URL vem da própria AWS
        data = json.load(r)

    data["_meta"] = {
        "case": case,
        "job": job_name,
        "seconds": round(time.perf_counter() - t0, 1),
        "uri": uri,
        "language": language,
    }
    return data


# ----- Cache e resultado -----

def cache_path(case: str) -> Path:
    return CACHE_DIR / f"{case}.json"


def transcript_text(data: dict) -> str:
    """Texto corrido devolvido pelo Transcribe."""
    return data["results"]["transcripts"][0]["transcript"]


def speaker_turns(data: dict) -> list[dict]:
    """
    Reconstrói os turnos a partir da diarização.

    O Transcribe devolve rótulos por *segmento* e as palavras separadamente; aqui os
    segmentos contíguos de um mesmo falante são unidos, para o formato ficar comparável
    ao da referência humana (um turno por fala).
    """
    segments = data["results"].get("speaker_labels", {}).get("segments", [])
    items = {i["start_time"]: i for i in data["results"]["items"] if "start_time" in i}

    turns: list[dict] = []
    for seg in segments:
        words = [items[it["start_time"]]["alternatives"][0]["content"]
                 for it in seg.get("items", []) if it["start_time"] in items]
        if not words:
            continue
        text = " ".join(words)
        if turns and turns[-1]["speaker"] == seg["speaker_label"]:
            turns[-1]["text"] += f" {text}"
        else:
            turns.append({"speaker": seg["speaker_label"], "text": text})
    return turns


def identify_patient_speaker(turns: list[dict]) -> str | None:
    """
    Descobre qual rótulo da diarização (``spk_0``/``spk_1``) é o paciente.

    O Transcribe rotula os falantes por ordem de aparição, sem saber quem é quem. Dois
    sinais independentes identificam o paciente no formato OSCE deste dataset:

    1. **quem fala mais** — o paciente descreve sintomas, o médico faz perguntas curtas;
    2. **quem NÃO abre a consulta** — quem inicia é sempre o médico ("what brings you
       in today?").

    Usa-se a contagem de palavras como critério e a ordem de fala como conferência: se os
    dois discordarem, devolve ``None`` em vez de arriscar um palpite — atribuir as falas
    ao papel errado inverteria todo o relatório clínico.
    """
    if not turns:
        return None

    words: dict[str, int] = {}
    for t in turns:
        words[t["speaker"]] = words.get(t["speaker"], 0) + len(t["text"].split())
    if len(words) < 2:
        return None

    by_words = max(words, key=words.get)
    first_speaker = turns[0]["speaker"]
    return by_words if by_words != first_speaker else None


def patient_text_from_aws(data: dict) -> str:
    """
    Texto só das falas do paciente, a partir da diarização do Transcribe.

    Sem isso, a extração de entidades rodaria sobre a consulta inteira e contaria as
    **perguntas do médico** ("any rashes? skin changes?") como achados do paciente.
    """
    turns = speaker_turns(data)
    patient = identify_patient_speaker(turns)
    if patient is None:
        return transcript_text(data)   # não deu para separar: devolve tudo, sem inventar
    return " ".join(t["text"] for t in turns if t["speaker"] == patient)


def evaluate(case: str, root: str | Path, data: dict, keep_fillers: bool = False) -> dict:
    """Compara a transcrição da AWS com a referência humana e devolve as métricas."""
    ref = normalize(full_text(root, case), drop_fillers=not keep_fillers)
    hyp = normalize(transcript_text(data), drop_fillers=not keep_fillers)
    # full_text() prefixa "doctor:"/"patient:" em cada linha — fora da comparação.
    ref = [w for w in ref if w not in ("doctor", "patient")]

    m = wer(ref, hyp)
    human = load_transcript(root, case)
    return {
        "case": case,
        "wer": round(m["wer"], 4),
        "substitutions": m["sub"],
        "insertions": m["ins"],
        "deletions": m["del"],
        "reference_words": m["ref_words"],
        "aws_words": len(hyp),
        "human_turns": len(human),
        "aws_turns": len(speaker_turns(data)),
        "job_seconds": data.get("_meta", {}).get("seconds"),
    }


def process(
    cases: list[str],
    root: str | Path,
    bucket: str,
    region: str,
    force: bool = False,
    keep_fillers: bool = False,
) -> list[dict]:
    """Transcreve (ou reaproveita do cache) e avalia cada caso."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for case in cases:
        cache = cache_path(case)
        if cache.exists() and not force:
            print(f"[{case}] reaproveitando transcrição em cache")
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"[{case}] enviando ao S3 e iniciando o job...")
            data = transcribe_case(
                case, root, bucket, region,
                progress=lambda c, s, t: print(f"[{c}] {s} — {t:.0f}s", end="\r"),
            )
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(f"[{case}] concluído em {data['_meta']['seconds']:.0f}s"
                  f" — salvo em {cache.relative_to(ROOT_DIR)}")

        results.append(evaluate(case, root, data, keep_fillers=keep_fillers))
    return results
