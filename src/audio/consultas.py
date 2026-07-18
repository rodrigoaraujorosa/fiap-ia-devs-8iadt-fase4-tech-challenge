"""
Loader do dataset de consultas médicas simuladas (Entrega 2 — Análise de Áudio).

Dataset: *A dataset of simulated patient-physician medical interviews with a focus on
respiratory cases* (figshare, DOI 10.6084/m9.figshare.16550013.v1, licença CC0).

São 272 consultas em formato OSCE, cada uma com o áudio (MP3 16 kHz mono, 11-15 min) e a
**transcrição humana revisada**. O nome do arquivo carrega a especialidade:

    RES0001.mp3  ->  caso respiratório nº 1

Papel na entrega: é a fonte de **fala clínica espontânea**, que alimenta o Amazon
Transcribe e, na sequência, o Comprehend Medical. O Coswara (ver ``dataset.py``) cobre a
outra metade — fonação sustentada e respiração, que a consulta não tem e sem as quais não
se calcula jitter/shimmer de forma confiável.

A transcrição humana é **ground-truth**: permite medir o erro do Transcribe em vez de
apenas exibir o resultado.

Uso:
    python -m src.audio.consultas --root data/audio/consultas --resumo
    python -m src.audio.consultas --root data/audio/consultas --caso RES0001
    python -m src.audio.consultas --root data/audio/consultas --amostra 5 --out reports/consultas.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Prefixo do nome do arquivo -> especialidade.
CATEGORIAS = {
    "RES": "respiratório",
    "MSK": "musculoesquelético",
    "GAS": "gastrointestinal",
    "CAR": "cardíaco",
    "DER": "dermatológico",
    "GEN": "geral",
}

AUDIO_DIR = "Data/Audio Recordings"
TRANSCRIPT_DIR = "Data/Clean Transcripts"

# As falas vêm marcadas por "D:" (doctor) e "P:" (patient) no início da linha.
_TURNO = re.compile(r"^\s*([DP])\s*:\s*(.*)$")


def _categoria(caso: str) -> str:
    return CATEGORIAS.get(caso[:3], "desconhecido")


def _read_text(path: Path) -> str:
    """
    Lê a transcrição respeitando a codificação do arquivo.

    O dataset **não é homogêneo**: 2 dos 213 casos respiratórios (RES0002 e RES0054) estão
    em UTF-16, o resto em UTF-8. Ler tudo como UTF-8 não levanta erro — devolve texto
    corrompido, e o caso simplesmente aparece com zero turnos de fala. Por isso a
    codificação é detectada pelo BOM em vez de assumida.
    """
    dados = path.read_bytes()
    for bom, enc in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"), (b"\xef\xbb\xbf", "utf-8-sig")):
        if dados.startswith(bom):
            return dados.decode(enc)
    return dados.decode("utf-8", errors="replace")


def list_cases(root: str | Path) -> pd.DataFrame:
    """
    Lista os casos disponíveis, com especialidade e presença de áudio/transcrição.

    Não lê o conteúdo — serve para escolher o recorte antes de processar qualquer coisa.
    """
    root = Path(root)
    audios = {p.stem: p for p in (root / AUDIO_DIR).glob("*.mp3")}
    textos = {p.stem: p for p in (root / TRANSCRIPT_DIR).glob("*.txt")}

    linhas = []
    for caso in sorted(set(audios) | set(textos)):
        linhas.append({
            "caso": caso,
            "categoria": _categoria(caso),
            "tem_audio": caso in audios,
            "tem_transcricao": caso in textos,
        })
    return pd.DataFrame(linhas)


def audio_path(root: str | Path, caso: str) -> Path:
    """Caminho do MP3 de um caso."""
    return Path(root) / AUDIO_DIR / f"{caso}.mp3"


def transcript_path(root: str | Path, caso: str) -> Path:
    """Caminho da transcrição humana de um caso."""
    return Path(root) / TRANSCRIPT_DIR / f"{caso}.txt"


def load_transcript(root: str | Path, caso: str) -> pd.DataFrame:
    """
    Lê a transcrição e devolve os turnos de fala (ordem, falante, texto).

    ``falante`` é ``medico`` ou ``paciente``. Linhas de continuação (sem o prefixo
    ``D:``/``P:``) são coladas no turno anterior — o arquivo quebra falas longas em
    várias linhas, e tratá-las como turnos novos fragmentaria as frases justamente
    onde estão as descrições de sintoma.
    """
    texto = _read_text(transcript_path(root, caso))

    turnos: list[dict] = []
    for linha in texto.splitlines():
        m = _TURNO.match(linha)
        if m:
            turnos.append({
                "falante": "medico" if m.group(1) == "D" else "paciente",
                "texto": m.group(2).strip(),
            })
        elif turnos and linha.strip():
            turnos[-1]["texto"] = f"{turnos[-1]['texto']} {linha.strip()}".strip()

    df = pd.DataFrame(turnos)
    if not df.empty:
        df.insert(0, "ordem", range(1, len(df) + 1))
        df["caso"] = caso
    return df


def patient_text(root: str | Path, caso: str) -> str:
    """
    Concatena só as falas do paciente.

    É esse texto que vai para o Comprehend Medical: as perguntas do médico não descrevem
    sintomas do paciente e só adicionariam entidades que não são achados clínicos dele.
    """
    df = load_transcript(root, caso)
    if df.empty:
        return ""
    return " ".join(df.loc[df["falante"] == "paciente", "texto"])


def full_text(root: str | Path, caso: str) -> str:
    """Transcrição completa em texto corrido, com marcação de falante."""
    df = load_transcript(root, caso)
    if df.empty:
        return ""
    return "\n".join(f"{r.falante}: {r.texto}" for r in df.itertuples())


def build_sample(
    root: str | Path,
    categoria: str = "respiratório",
    n: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monta o recorte de trabalho, com estatísticas de cada caso.

    O padrão é a especialidade respiratória, que concentra 213 dos 272 casos e é a que
    dialoga com o sintoma-alvo da entrega ("dificuldades respiratórias e cansaço").

    ``n`` limita a amostra: transcrever as 272 consultas custaria ~54 h de áudio no
    Transcribe, o que não se justifica para uma demonstração.
    """
    casos = list_cases(root)
    casos = casos[casos["tem_audio"] & casos["tem_transcricao"]]
    if categoria:
        casos = casos[casos["categoria"] == categoria]
    if n:
        casos = casos.sample(min(n, len(casos)), random_state=seed).sort_values("caso")

    linhas = []
    for caso in casos["caso"]:
        turnos = load_transcript(root, caso)
        pac = turnos[turnos["falante"] == "paciente"] if not turnos.empty else turnos
        linhas.append({
            "caso": caso,
            "categoria": _categoria(caso),
            "turnos": len(turnos),
            "turnos_paciente": len(pac),
            "palavras_paciente": sum(len(t.split()) for t in pac["texto"]) if len(pac) else 0,
            "audio_mb": round(audio_path(root, caso).stat().st_size / 1024 / 1024, 1),
        })
    return pd.DataFrame(linhas).reset_index(drop=True)


def _resumo(root: str | Path) -> None:
    casos = list_cases(root)
    print(f"casos: {len(casos)}")
    print(f"  com áudio e transcrição: {int((casos.tem_audio & casos.tem_transcricao).sum())}")
    print("\npor especialidade:")
    for cat, n in casos["categoria"].value_counts().items():
        print(f"  {cat:22s} {n:4d} ({n / len(casos) * 100:.1f}%)")

    resp = build_sample(root, categoria="respiratório")
    if not resp.empty:
        print("\ncasos respiratórios — estatística dos turnos:")
        print(f"  turnos por consulta:     média {resp.turnos.mean():.0f}"
              f"  (min {resp.turnos.min()}, max {resp.turnos.max()})")
        print(f"  turnos do paciente:      média {resp.turnos_paciente.mean():.0f}")
        print(f"  palavras do paciente:    média {resp.palavras_paciente.mean():.0f}"
              f"  (min {resp.palavras_paciente.min()}, max {resp.palavras_paciente.max()})")
        print(f"  áudio total:             {resp.audio_mb.sum() / 1024:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Loader do dataset de consultas médicas simuladas.")
    ap.add_argument("--root", default="data/audio/consultas", help="raiz do dataset")
    ap.add_argument("--resumo", action="store_true", help="estatísticas do dataset")
    ap.add_argument("--caso", help="mostra a transcrição de um caso (ex.: RES0001)")
    ap.add_argument("--paciente", action="store_true",
                    help="com --caso, mostra só as falas do paciente")
    ap.add_argument("--amostra", type=int, metavar="N", help="monta um recorte de N casos")
    ap.add_argument("--categoria", default="respiratório", help="especialidade do recorte")
    ap.add_argument("--out", help="salva o recorte em CSV")
    args = ap.parse_args()

    if args.caso:
        if args.paciente:
            print(patient_text(args.root, args.caso))
        else:
            print(full_text(args.root, args.caso))
        return

    if args.amostra:
        s = build_sample(args.root, categoria=args.categoria, n=args.amostra)
        print(f"recorte: {len(s)} casos ({args.categoria})")
        print(s.to_string(index=False))
        print(f"\náudio somado: {s.audio_mb.sum():.0f} MB")
        if args.out:
            s.to_csv(args.out, index=False)
            print(f"salvo em {args.out}")
        return

    if args.resumo:
        _resumo(args.root)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
