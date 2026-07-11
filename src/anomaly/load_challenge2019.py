"""
Loader + baseline de deteccao de anomalia — PhysioNet/CinC Challenge 2019.
Sinais vitais horarios de UTI (1 arquivo .psv por paciente).

Uso:
    python load_challenge2019.py --data ./challenge2019
    python load_challenge2019.py --data ./challenge2019 --patient p000001

Requer: pandas numpy scikit-learn matplotlib
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]


def load_patient(path: str) -> pd.DataFrame:
    """Le um arquivo .psv (pipe-delimited) de um paciente."""
    df = pd.read_csv(path, sep="|")
    df["patient"] = os.path.splitext(os.path.basename(path))[0]
    df["hour"] = np.arange(len(df))
    return df


def load_dataset(data_dir: str, limit: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"Nenhum .psv encontrado em {data_dir}")
    if limit:
        files = files[:limit]
    print(f"Carregando {len(files)} pacientes...")
    return pd.concat((load_patient(f) for f in files), ignore_index=True)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline de anomalia sobre os sinais vitais usando IsolationForest.
    Retorna df com colunas 'anomaly_score' e 'is_anomaly'.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer

    X = df[VITALS].copy()
    X = SimpleImputer(strategy="median").fit_transform(X)  # vitais tem muitos NaN
    model = IsolationForest(contamination=0.02, random_state=42)
    df = df.copy()
    df["anomaly_score"] = model.fit_predict(X)          # -1 anomalia, 1 normal
    df["is_anomaly"] = (df["anomaly_score"] == -1).astype(int)
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Estrutura ===")
    print(f"Linhas (horas-paciente): {len(df):,}")
    print(f"Pacientes: {df['patient'].nunique():,}")
    print(f"Colunas: {df.shape[1]}")
    if "SepsisLabel" in df.columns:
        pos = df.groupby("patient")["SepsisLabel"].max().mean()
        print(f"Fracao de pacientes que desenvolvem sepse: {pos:.1%}")
    print("\n=== Cobertura dos sinais vitais (nao-nulos) ===")
    print((df[VITALS].notna().mean() * 100).round(1).astype(str) + " %")
    if "is_anomaly" in df.columns:
        print(f"\nHoras sinalizadas como anomalia: {df['is_anomaly'].mean():.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta com os .psv")
    ap.add_argument("--limit", type=int, default=500, help="max pacientes a carregar")
    ap.add_argument("--patient", help="inspeciona um paciente especifico e plota")
    args = ap.parse_args()

    df = load_dataset(args.data, limit=args.limit)
    df = detect_anomalies(df)
    summarize(df)

    if args.patient:
        sub = df[df["patient"] == args.patient]
        if sub.empty:
            print(f"Paciente {args.patient} nao encontrado.")
            return
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        for v in ["HR", "SBP", "Resp", "O2Sat"]:
            ax.plot(sub["hour"], sub[v], label=v, alpha=0.8)
        anom = sub[sub["is_anomaly"] == 1]
        ax.scatter(anom["hour"], anom["HR"], color="red", zorder=5, label="anomalia")
        ax.set_xlabel("hora de internacao"); ax.set_ylabel("valor")
        ax.set_title(f"Sinais vitais — {args.patient}"); ax.legend()
        out = f"{args.patient}_vitais.png"
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"Grafico salvo em {out}")


if __name__ == "__main__":
    main()
