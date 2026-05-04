import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

COLORS = {
    "Base": "#7f7f7f",
    "AVX256": "#1f77b4",
    "AVX512": "#ff7f0e",
    "AVX512_v2": "#2ca02c",
    "OpenMP_v1": "#d62728",
    "OpenMP_v2": "#9467bd",
    "OpenMP_v3": "#8c564b",
}

def plot_speedup(df, metric, title, filename):
    _, ax = plt.subplots(figsize=(10, 6))

    for variante in df["variante"].unique():
        subset = df[df["variante"] == variante].sort_values("N")
        ax.plot(
            subset["N"],
            subset[metric],
            label=variante,
            color=COLORS.get(variante, None),
            linewidth=2,
            markersize=6,
        )

    ax.set_xscale("log")

    ax.set_xlabel("N (dimensão da matriz)", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_speedup_general(df, metrics, labels_metrics, title, filename):
    variantes = df["variante"].values
    n_metrics = len(metrics)

    _, ax = plt.subplots(figsize=(max(10, len(variantes) * 2), 6))

    bar_width = 0.8 / n_metrics
    x = np.arange(len(variantes))

    bar_colors = ["#4285F4", "#EA4335"]

    for i, (metric, label) in enumerate(zip(metrics, labels_metrics)):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            df[metric].values,
            bar_width,
            label=label,
            color=bar_colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(variantes, rotation=25, ha="right", fontsize=11)
    ax.set_xlabel("Variante", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


os.chdir("../experiments/spmv")

for dirname in os.listdir():
    if len(os.listdir(dirname)) == 0:
        continue

    df = pd.read_csv(f"{dirname}/spmv_runs.csv")

    plot_speedup(
        df,
        metric="speedup_mean",
        title="SpMV — Speedup Médio por Variante",
        filename=f"{dirname}/spmv_speedup_mean.png",
    )

    plot_speedup(
        df,
        metric="speedup_median",
        title="SpMV — Mediana do Speedup por Variante",
        filename=f"{dirname}/spmv_speedup_median.png",
    )

    df = pd.read_csv(f"{dirname}/spmv_general.csv")

    plot_speedup_general(
        df,
        metrics=["speedup_geral_mean", "speedup_geral_median"],
        labels_metrics=["Speedup Médio", "Mediana do Speedup"],
        title="SpMV — Speedup por Variante (Geral)",
        filename=f"{dirname}/spmv_speedup_general.png",
    )
