import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

COLORS = {
    "Escalar":    "#6b7280",
    "AVX256":     "#f59e0b",
    "AVX512":     "#3b82f6",
    "AVX512_v2":  "#8b5cf6",
    "OpenMP_v1":  "#ef4444",
    "OpenMP_v2":  "#84cc16",
    "OpenMP_v3":  "#f97316",
}

MARKERS = {
    "Escalar":    "o",
    "AVX256":     "s",
    "AVX512":     "^",
    "AVX512_v2":  "D",
    "OpenMP_v1":  "X",
    "OpenMP_v2":  "*",
    "OpenMP_v3":  "h",
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
            marker=MARKERS.get(variante, "o"),
            linewidth=2,
            markersize=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("N (dimensão da matriz)", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_speedup_bars(df, metric, title, filename):
    variantes = df["variante"].unique()
    n_values = sorted(df["N"].unique())
    n_variantes = len(variantes)

    _, ax = plt.subplots(figsize=(max(10, len(n_values) * 2.5), 6))

    bar_width = 0.8 / n_variantes
    x = np.arange(len(n_values))

    for i, variante in enumerate(variantes):
        valores = []
        for n in n_values:
            row = df[(df["variante"] == variante) & (df["N"] == n)]
            valores.append(row[metric].values[0] if len(row) > 0 else 0)

        offset = (i - n_variantes / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            valores,
            bar_width,
            label=variante,
            color=COLORS.get(variante, None),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("N (dimensão da matriz)", fontsize=12)
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

    plot_speedup_bars(
        df,
        metric="speedup_mean",
        title="SpMV — Speedup Médio por Variante (Barras)",
        filename=f"{dirname}/spmv_speedup_mean_bars.png",
    )

    plot_speedup_bars(
        df,
        metric="speedup_median",
        title="SpMV — Mediana do Speedup por Variante (Barras)",
        filename=f"{dirname}/spmv_speedup_median_bars.png",
    )