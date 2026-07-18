"""
Loader do dataset Coswara (Entrega 2 — Análise de Áudio).

O Coswara publica, por lote de coleta, um `.tar.gz` fatiado em partes de 100 MB
(`20220224.tar.gz.aa`, `.ab`, ...) e, na raiz do repositório, três metadados leves:

- ``combined_data.csv``  — uma linha por participante (sintomas, comorbidades, idade...)
- ``csv_labels_legend.json`` — legenda das colunas abreviadas (``bd``, ``ftg``, ...)
- ``annotations/<sound>_labels.csv`` — qualidade de cada gravação, avaliada por escuta
  manual: 0 (ruim), 1 (boa), 2 (excelente)

Cada participante contribui com **nove** gravações. Este módulo lê os metadados,
cruza com a qualidade e monta a coorte de trabalho — sem tocar no áudio, o que
permite decidir o recorte antes de gastar chamadas de nuvem.

Uso:
    python -m src.audio.dataset --root data/audio/coswara --summary
    python -m src.audio.dataset --root data/audio/coswara --cohort --per-group 30
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

import pandas as pd

# As nove gravações de cada participante, na nomenclatura do dataset.
SOUNDS = (
    "breathing-deep", "breathing-shallow",
    "cough-heavy", "cough-shallow",
    "counting-fast", "counting-normal",
    "vowel-a", "vowel-e", "vowel-o",
)

# Fala contínua — é a única modalidade que faz sentido mandar para transcrição.
SPEECH_SOUNDS = ("counting-normal", "counting-fast")

# Sintomas do enunciado do desafio ("dificuldades respiratórias e cansaço"),
# nas abreviações do Coswara.
SYMPTOM_COLS = {
    "bd": "breathing_difficulty",
    "ftg": "fatigue",
    "cough": "cough",
    "fever": "fever",
    "st": "sore_throat",
    "cld": "chronic_lung_disease",
    "asthma": "asthma",
    "pneumonia": "pneumonia",
}

# Rótulo exibido para cada sintoma, usado nos relatórios em pt-BR.
SYMPTOM_LABELS_PT = {
    "breathing_difficulty": "dificuldade respiratória",
    "fatigue": "fadiga",
    "cough": "tosse",
    "fever": "febre",
    "sore_throat": "dor de garganta",
    "chronic_lung_disease": "doença pulmonar crônica",
    "asthma": "asma",
    "pneumonia": "pneumonia",
}

QUALITY_OK = 1  # >= 1 significa "boa" ou "excelente"


def _as_bool(s: pd.Series) -> pd.Series:
    """Coswara grava os sintomas como 'True'/NaN; NaN significa ausência."""
    return s.astype(str).str.strip().str.lower().eq("true")


def load_metadata(root: str | Path) -> pd.DataFrame:
    """
    Lê ``combined_data.csv`` e devolve um DataFrame com os sintomas normalizados.

    Acrescenta as colunas booleanas de :data:`SYMPTOM_COLS`, além de ``symptomatic``
    (dificuldade respiratória **ou** fadiga), ``any_symptom`` e ``healthy``.
    """
    root = Path(root)
    df = pd.read_csv(root / "combined_data.csv")

    for abbrev, col in SYMPTOM_COLS.items():
        df[col] = _as_bool(df[abbrev]) if abbrev in df.columns else False

    df["symptomatic"] = df["breathing_difficulty"] | df["fatigue"]
    df["any_symptom"] = df[list(SYMPTOM_COLS.values())].any(axis=1)

    # `covid_status == healthy` NÃO basta para o grupo de controle: 121 dos 1.433
    # participantes assim declarados relatam algum sintoma (12 deles justamente
    # dificuldade respiratória ou fadiga). O controle exige ausência de sintomas.
    df["healthy"] = df["covid_status"].eq("healthy") & ~df["any_symptom"]
    return df


def load_legend(root: str | Path) -> dict[str, str]:
    """Legenda das colunas abreviadas do Coswara."""
    with open(Path(root) / "csv_labels_legend.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_quality(root: str | Path) -> pd.DataFrame:
    """
    Lê ``annotations/*_labels.csv`` e devolve (id, sound, quality) em formato longo.

    A qualidade vem de escuta manual: 0 ruim, 1 boa, 2 excelente. Filtrar por ela
    evita mandar gravação inaudível para a nuvem — o que custa dinheiro e polui o
    resultado.
    """
    root = Path(root)
    rows = []
    for sound in SOUNDS:
        f = root / "annotations" / f"{sound}_labels.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d["id"] = d["FILENAME"].str.replace(f"_{sound}", "", regex=False)
        d["sound"] = sound
        rows.append(d[["id", "sound", "QUALITY"]].rename(columns={"QUALITY": "quality"}))
    if not rows:
        return pd.DataFrame(columns=["id", "sound", "quality"])
    return pd.concat(rows, ignore_index=True)


def load_folder_ids(root: str | Path) -> pd.DataFrame:
    """
    Mapeia participante -> lote de coleta, a partir dos CSVs por pasta.

    O ``record_date`` do participante **não** identifica a pasta (são 398 datas de
    gravação para 45 lotes), então esse mapa é a única forma de saber em qual
    arquivo o áudio de um participante está.
    """
    root = Path(root)
    rows = []
    for f in sorted((root / "folder_csv").glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001 — CSV corrompido no dataset: ignora o lote
            continue
        if "id" not in d.columns:
            continue
        rows.append(pd.DataFrame({"id": d["id"], "batch": f.stem}))
    if not rows:
        return pd.DataFrame(columns=["id", "batch"])
    return pd.concat(rows, ignore_index=True).drop_duplicates("id")


def build_cohort(
    root: str | Path,
    batches: list[str] | None = None,
    sound: str = "counting-normal",
    per_group: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monta a coorte de trabalho: participantes com áudio disponível e de boa qualidade.

    Devolve um DataFrame com ``group`` ('symptomatic' ou 'healthy'), pronto para o
    pipeline. ``per_group`` equilibra as classes por amostragem — sem isso a coorte
    fica enviesada, já que os lotes têm muito mais sintomáticos que saudáveis.

    ``sound`` é a gravação usada como critério de qualidade (padrão: a fala que vai
    para o Transcribe).
    """
    meta = load_metadata(root)
    batch_map = load_folder_ids(root)
    quality = load_quality(root)

    df = meta.merge(batch_map, on="id", how="inner")
    if batches:
        df = df[df["batch"].isin(batches)]

    q = quality[quality["sound"] == sound][["id", "quality"]]
    df = df.merge(q, on="id", how="left")
    df = df[df["quality"] >= QUALITY_OK]

    # Só interessam os dois extremos, e eles precisam ser mutuamente exclusivos:
    # quem tem dificuldade respiratória ou fadiga, e quem não relata sintoma nenhum.
    df = df[df["symptomatic"] | df["healthy"]].copy()
    df["group"] = df["symptomatic"].map({True: "symptomatic", False: "healthy"})

    if per_group:
        df = (df.groupby("group", group_keys=False)[df.columns]
                .apply(lambda g: g.sample(min(len(g), per_group), random_state=seed)))

    cols = ["id", "batch", "group", "a", "g", "covid_status", "quality",
            *SYMPTOM_COLS.values()]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def extract_batch(root: str | Path, batch: str, dest: str | Path | None = None) -> Path:
    """
    Junta as partes ``<batch>.tar.gz.a*`` e extrai o lote.

    O Coswara fatia cada lote em partes de 100 MB porque o GitHub limita o tamanho
    de arquivo; elas precisam ser concatenadas antes de descompactar. Se o destino
    já existir, não refaz o trabalho.
    """
    root = Path(root)
    source = root / "raw" / batch
    dest = Path(dest) if dest else root / "extracted"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / batch).exists():
        return dest / batch

    parts = sorted(source.glob(f"{batch}.tar.gz.*"))
    if not parts:
        raise FileNotFoundError(f"nenhuma parte encontrada em {source}")

    merged = source / f"{batch}.tar.gz"
    with open(merged, "wb") as out:
        for p in parts:
            with open(p, "rb") as part:
                while chunk := part.read(1024 * 1024):
                    out.write(chunk)
    with tarfile.open(merged, "r:gz") as tar:
        # filter="data": recusa caminhos absolutos e ".." no tar — o padrão do Python 3.14
        # e a única postura sensata ao extrair arquivo baixado da internet.
        tar.extractall(dest, filter="data")
    os.remove(merged)  # o .tar.gz reconstruído é redundante depois de extraído
    return dest / batch


def audio_path(root: str | Path, batch: str, user_id: str, sound: str) -> Path:
    """Caminho da gravação de um participante, dentro do lote já extraído."""
    return Path(root) / "extracted" / batch / user_id / f"{sound}.wav"


def _summary(root: str | Path) -> None:
    meta = load_metadata(root)
    quality = load_quality(root)
    batch_map = load_folder_ids(root)
    print(f"participantes no combined_data: {len(meta)}")
    print(f"lotes mapeados: {batch_map['batch'].nunique()}"
          f" | participantes mapeados: {len(batch_map)}")
    print("\nstatus de saúde:")
    print(meta["covid_status"].value_counts().to_string())
    print("\nsintomas (True):")
    for col in SYMPTOM_COLS.values():
        n = int(meta[col].sum())
        rotulo = SYMPTOM_LABELS_PT.get(col, col)
        print(f"  {rotulo:26s} {n:5d} ({n / len(meta) * 100:.1f}%)")
    print(f"\nsintomáticos (resp. ou fadiga): {int(meta['symptomatic'].sum())}")
    print(f"saudáveis (sem sintoma algum): {int(meta['healthy'].sum())}")
    if not quality.empty:
        print("\nqualidade das gravações (0 ruim, 1 boa, 2 excelente):")
        print(quality.pivot_table(index="sound", columns="quality", aggfunc="size",
                                  fill_value=0).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Loader e recorte do dataset Coswara.")
    ap.add_argument("--root", default="data/audio/coswara", help="raiz dos dados do Coswara")
    ap.add_argument("--summary", action="store_true", help="estatísticas dos metadados")
    ap.add_argument("--cohort", action="store_true", help="monta a coorte de trabalho")
    ap.add_argument("--batches", nargs="*", help="lotes a considerar (ex.: 20220224 20210406)")
    ap.add_argument("--sound", default="counting-normal", help="gravação usada como critério")
    ap.add_argument("--per-group", type=int, help="equilibra as classes com N por grupo")
    ap.add_argument("--extract", metavar="BATCH", help="junta as partes e extrai um lote")
    ap.add_argument("--out", help="salva a coorte em CSV")
    args = ap.parse_args()

    if args.extract:
        dest = extract_batch(args.root, args.extract)
        print(f"lote extraído em {dest}")
        return

    if args.summary:
        _summary(args.root)
        return

    if args.cohort:
        c = build_cohort(args.root, batches=args.batches, sound=args.sound,
                         per_group=args.per_group)
        print(f"coorte: {len(c)} participantes")
        print(c["group"].value_counts().to_string())
        print(f"\nidade média: {c['a'].mean():.1f} | gênero: {c['g'].value_counts().to_dict()}")
        print(f"\n{c.head(10).to_string(index=False)}")
        if args.out:
            c.to_csv(args.out, index=False)
            print(f"\nsalvo em {args.out}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
