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
Por isso o módulo *sempre* cacheia em ``reports/transcricoes/<caso>.json`` e nunca
reprocessa um caso já transcrito, a menos que se peça ``--forcar``.

Uso:
    python -m src.audio.transcribe --casos RES0001              # um caso
    python -m src.audio.transcribe --casos RES0001 RES0010      # vários
    python -m src.audio.transcribe --relatorio                  # WER do que já foi transcrito
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from ..common.config import ROOT_DIR, get_aws_config
from .consultas import audio_path, full_text, load_transcript

CACHE_DIR = ROOT_DIR / "reports" / "transcricoes"
PREFIXO_S3 = "consultas"

# Marcadores de hesitação. O anotador humano transcreveu "um", "uh", "ahh"; o Transcribe
# normalmente os omite. Contá-los como erro mede a convenção de anotação, não a qualidade
# do reconhecimento — por isso são removíveis dos dois lados (ver --com-hesitacao).
HESITACOES = {
    "um", "uh", "uhh", "umm", "ah", "ahh", "ahhh", "hmm", "mmm", "mhm",
    "er", "erm", "eh", "oh", "hm",
}


# ----- Normalização e WER -----

def normalizar(texto: str, remover_hesitacao: bool = True) -> list[str]:
    """
    Reduz o texto a uma lista de palavras comparável entre referência e hipótese.

    Minúsculas, sem pontuação e com espaços colapsados — sem isso o WER mediria
    diferenças de estilo de anotação (vírgulas, maiúsculas) em vez de reconhecimento.
    """
    texto = texto.lower()
    texto = re.sub(r"[^\w\s']", " ", texto)   # mantém apóstrofo: "i'm" != "im"
    palavras = texto.split()
    if remover_hesitacao:
        palavras = [p for p in palavras if p not in HESITACOES]
    return palavras


def wer(referencia: list[str], hipotese: list[str]) -> dict:
    """
    Word Error Rate entre duas listas de palavras, com o detalhe dos erros.

    WER = (substituições + inserções + deleções) / palavras da referência, obtido por
    programação dinâmica (distância de edição em nível de palavra). Devolve também cada
    tipo de erro, porque eles têm causas diferentes: deleção costuma ser fala sobreposta,
    inserção costuma ser ruído interpretado como palavra.
    """
    n, m = len(referencia), len(hipotese)
    if n == 0:
        return {"wer": float("nan"), "sub": 0, "ins": m, "del": 0, "palavras_ref": 0}

    # d[i][j] = custo mínimo para alinhar referencia[:i] com hipotese[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            custo = 0 if referencia[i - 1] == hipotese[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,          # deleção
                          d[i][j - 1] + 1,          # inserção
                          d[i - 1][j - 1] + custo)  # substituição/acerto

    # Retrocede pelo caminho ótimo para separar os tipos de erro.
    i, j, sub, ins, dele = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and referencia[i - 1] == hipotese[j - 1] and d[i][j] == d[i - 1][j - 1]:
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
            "palavras_ref": n}


# ----- AWS -----

def _clientes(region: str):
    import boto3
    return boto3.client("s3", region_name=region), boto3.client("transcribe", region_name=region)


def enviar_para_s3(caso: str, root: str | Path, bucket: str, region: str) -> str:
    """Envia o MP3 do caso para o S3 (se ainda não estiver lá) e devolve a URI."""
    from botocore.exceptions import ClientError

    s3, _ = _clientes(region)
    chave = f"{PREFIXO_S3}/{caso}.mp3"
    try:
        s3.head_object(Bucket=bucket, Key=chave)   # já enviado numa execução anterior
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        s3.upload_file(str(audio_path(root, caso)), bucket, chave)
    return f"s3://{bucket}/{chave}"


def transcrever(
    caso: str,
    root: str | Path,
    bucket: str,
    region: str,
    idioma: str = "en-US",
    falantes: int = 2,
    intervalo: float = 15.0,
    timeout: float = 1800.0,
    progresso=None,
) -> dict:
    """
    Roda o job do Transcribe para um caso e devolve o resultado bruto da AWS.

    ``falantes=2`` liga a diarização: a consulta é um diálogo médico-paciente, e separar
    os interlocutores permite comparar com os turnos ``D:``/``P:`` da referência humana.

    O nome do job carrega o instante de início porque a AWS **não permite reaproveitar o
    nome de um job existente**, e um job com falha deixaria o nome ocupado.
    """
    import urllib.request

    _, tr = _clientes(region)
    uri = enviar_para_s3(caso, root, bucket, region)
    nome_job = f"consulta-{caso}-{int(time.time())}"

    tr.start_transcription_job(
        TranscriptionJobName=nome_job,
        Media={"MediaFileUri": uri},
        MediaFormat="mp3",
        LanguageCode=idioma,
        Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": falantes},
    )

    t0 = time.perf_counter()
    while True:
        job = tr.get_transcription_job(TranscriptionJobName=nome_job)["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            break
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"job {nome_job} passou de {timeout:.0f}s ainda em {status}")
        if progresso:
            progresso(caso, status, time.perf_counter() - t0)
        time.sleep(intervalo)

    if status == "FAILED":
        raise RuntimeError(f"job {nome_job} falhou: {job.get('FailureReason', '?')}")

    url = job["Transcript"]["TranscriptFileUri"]
    with urllib.request.urlopen(url) as r:  # noqa: S310 — URL vem da própria AWS
        dados = json.load(r)

    dados["_meta"] = {
        "caso": caso,
        "job": nome_job,
        "segundos": round(time.perf_counter() - t0, 1),
        "uri": uri,
        "idioma": idioma,
    }
    return dados


# ----- Cache e resultado -----

def caminho_cache(caso: str) -> Path:
    return CACHE_DIR / f"{caso}.json"


def texto_da_transcricao(dados: dict) -> str:
    """Texto corrido devolvido pelo Transcribe."""
    return dados["results"]["transcripts"][0]["transcript"]


def turnos_por_falante(dados: dict) -> list[dict]:
    """
    Reconstrói os turnos a partir da diarização.

    O Transcribe devolve rótulos por *segmento* e as palavras separadamente; aqui os
    segmentos contíguos de um mesmo falante são unidos, para o formato ficar comparável
    ao da referência humana (um turno por fala).
    """
    segmentos = dados["results"].get("speaker_labels", {}).get("segments", [])
    itens = {i["start_time"]: i for i in dados["results"]["items"] if "start_time" in i}

    turnos: list[dict] = []
    for seg in segmentos:
        palavras = [itens[it["start_time"]]["alternatives"][0]["content"]
                    for it in seg.get("items", []) if it["start_time"] in itens]
        if not palavras:
            continue
        texto = " ".join(palavras)
        if turnos and turnos[-1]["falante"] == seg["speaker_label"]:
            turnos[-1]["texto"] += f" {texto}"
        else:
            turnos.append({"falante": seg["speaker_label"], "texto": texto})
    return turnos


def avaliar(caso: str, root: str | Path, dados: dict, com_hesitacao: bool = False) -> dict:
    """Compara a transcrição da AWS com a referência humana e devolve as métricas."""
    ref = normalizar(full_text(root, caso), remover_hesitacao=not com_hesitacao)
    hip = normalizar(texto_da_transcricao(dados), remover_hesitacao=not com_hesitacao)
    # full_text() prefixa "medico:"/"paciente:" em cada linha — fora da comparação.
    ref = [p for p in ref if p not in ("medico", "paciente")]

    m = wer(ref, hip)
    humanos = load_transcript(root, caso)
    return {
        "caso": caso,
        "wer": round(m["wer"], 4),
        "substituicoes": m["sub"],
        "insercoes": m["ins"],
        "delecoes": m["del"],
        "palavras_referencia": m["palavras_ref"],
        "palavras_aws": len(hip),
        "turnos_humanos": len(humanos),
        "turnos_aws": len(turnos_por_falante(dados)),
        "segundos_job": dados.get("_meta", {}).get("segundos"),
    }


def processar(
    casos: list[str],
    root: str | Path,
    bucket: str,
    region: str,
    forcar: bool = False,
    com_hesitacao: bool = False,
) -> list[dict]:
    """Transcreve (ou reaproveita do cache) e avalia cada caso."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resultados = []

    for caso in casos:
        cache = caminho_cache(caso)
        if cache.exists() and not forcar:
            print(f"[{caso}] reaproveitando transcrição em cache")
            dados = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"[{caso}] enviando ao S3 e iniciando o job...")
            dados = transcrever(
                caso, root, bucket, region,
                progresso=lambda c, s, t: print(f"[{c}] {s} — {t:.0f}s", end="\r"),
            )
            cache.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
            print(f"[{caso}] concluído em {dados['_meta']['segundos']:.0f}s"
                  f" — salvo em {cache.relative_to(ROOT_DIR)}")

        resultados.append(avaliar(caso, root, dados, com_hesitacao=com_hesitacao))
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcrição de consultas com Amazon Transcribe.")
    ap.add_argument("--root", default="data/audio/consultas", help="raiz do dataset")
    ap.add_argument("--casos", nargs="+", help="casos a transcrever (ex.: RES0001)")
    ap.add_argument("--forcar", action="store_true",
                    help="reprocessa mesmo se já houver cache (custa dinheiro de novo)")
    ap.add_argument("--com-hesitacao", action="store_true",
                    help="conta 'um', 'uh' etc. no WER (padrão: remove dos dois lados)")
    ap.add_argument("--relatorio", action="store_true",
                    help="avalia tudo o que já está em cache, sem chamar a AWS")
    ap.add_argument("--out", help="salva as métricas em CSV")
    args = ap.parse_args()

    cfg = get_aws_config()

    if args.relatorio:
        casos = sorted(p.stem for p in CACHE_DIR.glob("*.json"))
        if not casos:
            print("nenhuma transcrição em cache — rode com --casos primeiro")
            return
    elif args.casos:
        casos = args.casos
    else:
        ap.print_help()
        return

    if not args.relatorio and (not cfg["region"] or not cfg["s3_bucket"]):
        print("AWS não configurada. Rode: python -m src.common.config")
        return

    resultados = processar(casos, args.root, cfg["s3_bucket"], cfg["region"],
                           forcar=args.forcar, com_hesitacao=args.com_hesitacao)

    import pandas as pd
    df = pd.DataFrame(resultados)
    print(f"\n{df.to_string(index=False)}")
    if len(df) > 1:
        print(f"\nWER médio: {df['wer'].mean():.3f}  (mediana {df['wer'].median():.3f})")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"salvo em {args.out}")


if __name__ == "__main__":
    main()
