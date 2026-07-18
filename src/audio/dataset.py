"""
Loader do dataset Coswara (Entrega 2 — Análise de Áudio).

O Coswara publica, por lote de coleta, um `.tar.gz` fatiado em partes de 100 MB
(`20220224.tar.gz.aa`, `.ab`, ...) e, na raiz do repositório, três metadados leves:

- ``combined_data.csv``  — uma linha por participante (sintomas, comorbidades, idade...)
- ``csv_labels_legend.json`` — legenda das colunas abreviadas (``bd``, ``ftg``, ...)
- ``annotations/<som>_labels.csv`` — qualidade de cada gravação, avaliada por escuta
  manual: 0 (ruim), 1 (boa), 2 (excelente)

Cada participante contribui com **nove** gravações. Este módulo lê os metadados,
cruza com a qualidade e monta a coorte de trabalho — sem tocar no áudio, o que
permite decidir o recorte antes de gastar chamadas de nuvem.

Uso:
    python -m src.audio.dataset --root data/audio/coswara --resumo
    python -m src.audio.dataset --root data/audio/coswara --coorte --por-grupo 30
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
    "bd": "dificuldade_respiratoria",
    "ftg": "fadiga",
    "cough": "tosse",
    "fever": "febre",
    "st": "dor_de_garganta",
    "cld": "doenca_pulmonar_cronica",
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

    Acrescenta as colunas booleanas de :data:`SYMPTOM_COLS` (em português), além de
    ``sintomatico`` (dificuldade respiratória **ou** fadiga) e ``saudavel``.
    """
    root = Path(root)
    df = pd.read_csv(root / "combined_data.csv")

    for col, nome in SYMPTOM_COLS.items():
        df[nome] = _as_bool(df[col]) if col in df.columns else False

    df["sintomatico"] = df["dificuldade_respiratoria"] | df["fadiga"]
    df["algum_sintoma"] = df[list(SYMPTOM_COLS.values())].any(axis=1)

    # `covid_status == healthy` NÃO basta para o grupo de controle: 121 dos 1.433
    # participantes assim declarados relatam algum sintoma (12 deles justamente
    # dificuldade respiratória ou fadiga). O controle exige ausência de sintomas.
    df["saudavel"] = df["covid_status"].eq("healthy") & ~df["algum_sintoma"]
    return df


def load_legend(root: str | Path) -> dict[str, str]:
    """Legenda das colunas abreviadas do Coswara."""
    with open(Path(root) / "csv_labels_legend.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_quality(root: str | Path) -> pd.DataFrame:
    """
    Lê ``annotations/*_labels.csv`` e devolve (id, som, qualidade) em formato longo.

    A qualidade vem de escuta manual: 0 ruim, 1 boa, 2 excelente. Filtrar por ela
    evita mandar gravação inaudível para a nuvem — o que custa dinheiro e polui o
    resultado.
    """
    root = Path(root)
    linhas = []
    for som in SOUNDS:
        f = root / "annotations" / f"{som}_labels.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d["id"] = d["FILENAME"].str.replace(f"_{som}", "", regex=False)
        d["som"] = som
        linhas.append(d[["id", "som", "QUALITY"]].rename(columns={"QUALITY": "qualidade"}))
    if not linhas:
        return pd.DataFrame(columns=["id", "som", "qualidade"])
    return pd.concat(linhas, ignore_index=True)


def load_folder_ids(root: str | Path) -> pd.DataFrame:
    """
    Mapeia participante -> lote de coleta, a partir dos CSVs por pasta.

    O ``record_date`` do participante **não** identifica a pasta (são 398 datas de
    gravação para 45 lotes), então esse mapa é a única forma de saber em qual
    arquivo o áudio de um participante está.
    """
    root = Path(root)
    linhas = []
    for f in sorted((root / "folder_csv").glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001 — CSV corrompido no dataset: ignora o lote
            continue
        if "id" not in d.columns:
            continue
        linhas.append(pd.DataFrame({"id": d["id"], "lote": f.stem}))
    if not linhas:
        return pd.DataFrame(columns=["id", "lote"])
    return pd.concat(linhas, ignore_index=True).drop_duplicates("id")


def build_cohort(
    root: str | Path,
    lotes: list[str] | None = None,
    som: str = "counting-normal",
    por_grupo: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monta a coorte de trabalho: participantes com áudio disponível e de boa qualidade.

    Devolve um DataFrame com ``grupo`` ('sintomatico' ou 'saudavel'), pronto para o
    pipeline. ``por_grupo`` equilibra as classes por amostragem — sem isso a coorte
    fica enviesada, já que os lotes têm muito mais sintomáticos que saudáveis.

    ``som`` é a gravação usada como critério de qualidade (padrão: a fala que vai
    para o Transcribe).
    """
    meta = load_metadata(root)
    mapa = load_folder_ids(root)
    qual = load_quality(root)

    df = meta.merge(mapa, on="id", how="inner")
    if lotes:
        df = df[df["lote"].isin(lotes)]

    q = qual[qual["som"] == som][["id", "qualidade"]]
    df = df.merge(q, on="id", how="left")
    df = df[df["qualidade"] >= QUALITY_OK]

    # Só interessam os dois extremos, e eles precisam ser mutuamente exclusivos:
    # quem tem dificuldade respiratória ou fadiga, e quem não relata sintoma nenhum.
    df = df[df["sintomatico"] | df["saudavel"]].copy()
    df["grupo"] = df["sintomatico"].map({True: "sintomatico", False: "saudavel"})

    if por_grupo:
        df = (df.groupby("grupo", group_keys=False)[df.columns]
                .apply(lambda g: g.sample(min(len(g), por_grupo), random_state=seed)))

    cols = ["id", "lote", "grupo", "a", "g", "covid_status", "qualidade",
            *SYMPTOM_COLS.values()]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def extract_lote(root: str | Path, lote: str, dest: str | Path | None = None) -> Path:
    """
    Junta as partes ``<lote>.tar.gz.a*`` e extrai o lote.

    O Coswara fatia cada lote em partes de 100 MB porque o GitHub limita o tamanho
    de arquivo; elas precisam ser concatenadas antes de descompactar. Se o destino
    já existir, não refaz o trabalho.
    """
    root = Path(root)
    origem = root / "raw" / lote
    dest = Path(dest) if dest else root / "extracted"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / lote).exists():
        return dest / lote

    partes = sorted(origem.glob(f"{lote}.tar.gz.*"))
    if not partes:
        raise FileNotFoundError(f"nenhuma parte encontrada em {origem}")

    combinado = origem / f"{lote}.tar.gz"
    with open(combinado, "wb") as saida:
        for p in partes:
            with open(p, "rb") as parte:
                while chunk := parte.read(1024 * 1024):
                    saida.write(chunk)
    with tarfile.open(combinado, "r:gz") as tar:
        # filter="data": recusa caminhos absolutos e ".." no tar — o padrão do Python 3.14
        # e a única postura sensata ao extrair arquivo baixado da internet.
        tar.extractall(dest, filter="data")
    os.remove(combinado)  # o .tar.gz reconstruído é redundante depois de extraído
    return dest / lote


def audio_path(root: str | Path, lote: str, user_id: str, som: str) -> Path:
    """Caminho da gravação de um participante, dentro do lote já extraído."""
    return Path(root) / "extracted" / lote / user_id / f"{som}.wav"


def _resumo(root: str | Path) -> None:
    meta = load_metadata(root)
    qual = load_quality(root)
    mapa = load_folder_ids(root)
    print(f"participantes no combined_data: {len(meta)}")
    print(f"lotes mapeados: {mapa['lote'].nunique()} | participantes mapeados: {len(mapa)}")
    print("\nstatus de saúde:")
    print(meta["covid_status"].value_counts().to_string())
    print("\nsintomas (True):")
    for nome in SYMPTOM_COLS.values():
        n = int(meta[nome].sum())
        print(f"  {nome:26s} {n:5d} ({n / len(meta) * 100:.1f}%)")
    print(f"\nsintomáticos (resp. ou fadiga): {int(meta['sintomatico'].sum())}")
    print(f"saudáveis: {int(meta['saudavel'].sum())}")
    if not qual.empty:
        print("\nqualidade das gravações (0 ruim, 1 boa, 2 excelente):")
        print(qual.pivot_table(index="som", columns="qualidade", aggfunc="size",
                               fill_value=0).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Loader e recorte do dataset Coswara.")
    ap.add_argument("--root", default="data/audio/coswara", help="raiz dos dados do Coswara")
    ap.add_argument("--resumo", action="store_true", help="estatísticas dos metadados")
    ap.add_argument("--coorte", action="store_true", help="monta a coorte de trabalho")
    ap.add_argument("--lotes", nargs="*", help="lotes a considerar (ex.: 20220224 20210406)")
    ap.add_argument("--som", default="counting-normal", help="gravação usada como critério")
    ap.add_argument("--por-grupo", type=int, help="equilibra as classes com N por grupo")
    ap.add_argument("--extrair", metavar="LOTE", help="junta as partes e extrai um lote")
    ap.add_argument("--out", help="salva a coorte em CSV")
    args = ap.parse_args()

    if args.extrair:
        destino = extract_lote(args.root, args.extrair)
        print(f"lote extraído em {destino}")
        return

    if args.resumo:
        _resumo(args.root)
        return

    if args.coorte:
        c = build_cohort(args.root, lotes=args.lotes, som=args.som, por_grupo=args.por_grupo)
        print(f"coorte: {len(c)} participantes")
        print(c["grupo"].value_counts().to_string())
        print(f"\nidade média: {c['a'].mean():.1f} | gênero: {c['g'].value_counts().to_dict()}")
        print(f"\n{c.head(10).to_string(index=False)}")
        if args.out:
            c.to_csv(args.out, index=False)
            print(f"\nsalvo em {args.out}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
