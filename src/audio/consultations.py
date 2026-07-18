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
    python -m src.audio.consultations --root data/audio/consultas --summary
    python -m src.audio.consultations --root data/audio/consultas --case RES0001
    python -m src.audio.consultations --root data/audio/consultas --sample 5 --out amostra.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Prefixo do nome do arquivo -> especialidade. Os rótulos ficam em pt-BR por serem
# conteúdo exibido ao usuário, não identificadores de código.
CATEGORIES = {
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
_TURN = re.compile(r"^\s*([DP])\s*:\s*(.*)$")


def _category(case: str) -> str:
    return CATEGORIES.get(case[:3], "desconhecido")


def _read_text(path: Path) -> str:
    """
    Lê a transcrição respeitando a codificação do arquivo.

    O dataset **não é homogêneo**: 2 dos 213 casos respiratórios (RES0002 e RES0054) estão
    em UTF-16, o resto em UTF-8. Ler tudo como UTF-8 não levanta erro — devolve texto
    corrompido, e o caso simplesmente aparece com zero turnos de fala. Por isso a
    codificação é detectada pelo BOM em vez de assumida.
    """
    raw = path.read_bytes()
    for bom, enc in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return raw.decode(enc)
    return raw.decode("utf-8", errors="replace")


def list_cases(root: str | Path) -> pd.DataFrame:
    """
    Lista os casos disponíveis, com especialidade e presença de áudio/transcrição.

    Não lê o conteúdo — serve para escolher o recorte antes de processar qualquer coisa.
    """
    root = Path(root)
    audios = {p.stem for p in (root / AUDIO_DIR).glob("*.mp3")}
    texts = {p.stem for p in (root / TRANSCRIPT_DIR).glob("*.txt")}

    rows = [
        {
            "case": case,
            "category": _category(case),
            "has_audio": case in audios,
            "has_transcript": case in texts,
        }
        for case in sorted(audios | texts)
    ]
    return pd.DataFrame(rows)


def audio_path(root: str | Path, case: str) -> Path:
    """Caminho do MP3 de um caso."""
    return Path(root) / AUDIO_DIR / f"{case}.mp3"


def transcript_path(root: str | Path, case: str) -> Path:
    """Caminho da transcrição humana de um caso."""
    return Path(root) / TRANSCRIPT_DIR / f"{case}.txt"


def load_transcript(root: str | Path, case: str) -> pd.DataFrame:
    """
    Lê a transcrição e devolve os turnos de fala (order, speaker, text).

    ``speaker`` é ``doctor`` ou ``patient``. Linhas de continuação (sem o prefixo
    ``D:``/``P:``) são coladas no turno anterior — o arquivo quebra falas longas em
    várias linhas, e tratá-las como turnos novos fragmentaria as frases justamente
    onde estão as descrições de sintoma.
    """
    text = _read_text(transcript_path(root, case))

    turns: list[dict] = []
    for line in text.splitlines():
        m = _TURN.match(line)
        if m:
            turns.append({
                "speaker": "doctor" if m.group(1) == "D" else "patient",
                "text": m.group(2).strip(),
            })
        elif turns and line.strip():
            turns[-1]["text"] = f"{turns[-1]['text']} {line.strip()}".strip()

    df = pd.DataFrame(turns)
    if not df.empty:
        df.insert(0, "order", range(1, len(df) + 1))
        df["case"] = case
    return df


def patient_text(root: str | Path, case: str) -> str:
    """
    Concatena só as falas do paciente.

    É esse texto que vai para o Comprehend Medical: as perguntas do médico não descrevem
    sintomas do paciente e só adicionariam entidades que não são achados clínicos dele.
    """
    df = load_transcript(root, case)
    if df.empty:
        return ""
    return " ".join(df.loc[df["speaker"] == "patient", "text"])


def full_text(root: str | Path, case: str) -> str:
    """Transcrição completa em texto corrido, com marcação de falante."""
    df = load_transcript(root, case)
    if df.empty:
        return ""
    return "\n".join(f"{r.speaker}: {r.text}" for r in df.itertuples())


def build_sample(
    root: str | Path,
    category: str = "respiratório",
    n: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monta o recorte de trabalho, com estatísticas de cada caso.

    O padrão é a especialidade respiratória, que concentra 213 dos 272 casos e é a que
    dialoga com o sintoma-alvo da entrega ("dificuldades respiratórias e cansaço").

    ``n`` limita a amostra: transcrever as 272 consultas custaria ~39 h de áudio no
    Transcribe, o que não se justifica para uma demonstração.
    """
    cases = list_cases(root)
    cases = cases[cases["has_audio"] & cases["has_transcript"]]
    if category:
        cases = cases[cases["category"] == category]
    if n:
        cases = cases.sample(min(n, len(cases)), random_state=seed).sort_values("case")

    rows = []
    for case in cases["case"]:
        turns = load_transcript(root, case)
        patient = turns[turns["speaker"] == "patient"] if not turns.empty else turns
        rows.append({
            "case": case,
            "category": _category(case),
            "turns": len(turns),
            "patient_turns": len(patient),
            "patient_words": sum(len(t.split()) for t in patient["text"]) if len(patient) else 0,
            "audio_mb": round(audio_path(root, case).stat().st_size / 1024 / 1024, 1),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def _summary(root: str | Path) -> None:
    cases = list_cases(root)
    print(f"casos: {len(cases)}")
    print(f"  com áudio e transcrição: {int((cases.has_audio & cases.has_transcript).sum())}")
    print("\npor especialidade:")
    for cat, n in cases["category"].value_counts().items():
        print(f"  {cat:22s} {n:4d} ({n / len(cases) * 100:.1f}%)")

    resp = build_sample(root, category="respiratório")
    if not resp.empty:
        print("\ncasos respiratórios — estatística dos turnos:")
        print(f"  turnos por consulta:     média {resp.turns.mean():.0f}"
              f"  (min {resp.turns.min()}, max {resp.turns.max()})")
        print(f"  turnos do paciente:      média {resp.patient_turns.mean():.0f}")
        print(f"  palavras do paciente:    média {resp.patient_words.mean():.0f}"
              f"  (min {resp.patient_words.min()}, max {resp.patient_words.max()})")
        print(f"  áudio total:             {resp.audio_mb.sum() / 1024:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Loader do dataset de consultas médicas simuladas.")
    ap.add_argument("--root", default="data/audio/consultas", help="raiz do dataset")
    ap.add_argument("--summary", action="store_true", help="estatísticas do dataset")
    ap.add_argument("--case", help="mostra a transcrição de um caso (ex.: RES0001)")
    ap.add_argument("--patient-only", action="store_true",
                    help="com --case, mostra só as falas do paciente")
    ap.add_argument("--sample", type=int, metavar="N", help="monta um recorte de N casos")
    ap.add_argument("--category", default="respiratório", help="especialidade do recorte")
    ap.add_argument("--out", help="salva o recorte em CSV")
    args = ap.parse_args()

    if args.case:
        print(patient_text(args.root, args.case) if args.patient_only
              else full_text(args.root, args.case))
        return

    if args.sample:
        s = build_sample(args.root, category=args.category, n=args.sample)
        print(f"recorte: {len(s)} casos ({args.category})")
        print(s.to_string(index=False))
        print(f"\náudio somado: {s.audio_mb.sum():.0f} MB")
        if args.out:
            s.to_csv(args.out, index=False)
            print(f"salvo em {args.out}")
        return

    if args.summary:
        _summary(args.root)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
