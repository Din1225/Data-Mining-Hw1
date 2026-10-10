"""
程式用途：由 results/ 下的 CV 結果畫出報告用的圖。

圖：
1. tuning_curves：Ridge、Lasso、ElasticNet 在 Set C 上的 5-fold CV RMSLE 對正則化強度 α 的曲線
   （陰影為 fold 間 ±1 std；水平線為 OLS）。Ridge 與 Lasso／ElasticNet 的 α 尺度不同
   （Ridge 乘在未除以 n 的平方誤差上），因此分成三個 panel、各自的 x 軸。
2. ablation：(a) Set A／B／B'／C 的 CV RMSLE；(b)(c) Set C 移除或加入一組特徵後，
   相對 Set C 的 ΔRMSLE 與 ΔMAE（同一 fold 相減後的 mean ± std）。

只讀取既有結果檔，不訓練模型，也不讀取 held-out test。
PNG 輸出至 png/（與 part 1 的圖放在一起），PDF 向量版輸出至 results/report/ 供 LaTeX 使用。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402


# 類別色依固定順序使用（前三色彼此在色盲模擬下仍可區分）；文字一律用墨色，不用資料色。
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)
ABLATION_SETS = {
    "set_a": "A: raw",
    "set_b_top31": "B: top-31 |r|",
    "set_b_top31_log_ranked": "B′: top-31 |r|, log",
    "set_c": "C: ours",
}
ABLATION_CHANGES = {
    "set_c_minus_domain_rules": "− domain rules",
    "set_c_minus_overall": "− overall (H1)",
    "set_c_minus_nonlinear": "− nonlinear",
    "set_c_minus_reputation": "− reputation (H2)",
    "set_c_minus_interaction": "− interaction",
    "set_c_plus_aggregates": "+ aggregates",
}
BAR_HEIGHT = 0.55


# 帶正負號的數值；四捨五入後為 0 時不加正負號，避免出現「+-0」。
def signed(value: float, digits: int, prefix: str = "", suffix: str = "") -> str:
    rounded = round(value, digits)
    sign = "+" if rounded > 0 else "−" if rounded < 0 else ""
    return f"{sign}{prefix}{abs(rounded):,.{digits}f}{suffix}"


def save(figure: plt.Figure, name: str, png_dir: Path, pdf_dir: Path) -> None:
    figure.savefig(png_dir / f"{name}.png")
    figure.savefig(pdf_dir / f"{name}.pdf")
    plt.close(figure)


def plot_curve(axis: plt.Axes, alpha: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, label: str | None) -> None:
    order = np.argsort(alpha)
    alpha, mean, std = alpha[order], mean[order], std[order]
    axis.fill_between(alpha, mean - std, mean + std, color=color, alpha=0.1, linewidth=0)
    axis.plot(alpha, mean, color=color, linewidth=1.5, marker="o", markersize=3.5,
              markeredgecolor="white", markeredgewidth=0.8, solid_capstyle="round", label=label)


def tuning_figure(main_dir: Path) -> plt.Figure:
    curves = pd.read_csv(main_dir / "tuning_curves.csv")
    curves["params"] = curves["params"].map(json.loads)
    ols = curves.loc[curves["variant"] == "ols"].iloc[0]
    figure, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)
    for axis, variant, title in zip(axes, ["ridge", "lasso", "elastic_net"], ["Ridge", "Lasso", "Elastic Net"]):
        rows = curves[curves["variant"] == variant]
        alpha = rows["params"].map(lambda p: p["alpha"]).to_numpy()
        if variant == "elastic_net":
            ratios = rows["params"].map(lambda p: p["l1_ratio"]).to_numpy()
            for color, ratio in zip(SERIES, sorted(set(ratios))):
                mask = ratios == ratio
                plot_curve(axis, alpha[mask], rows["rmsle_mean"].to_numpy()[mask],
                           rows["rmsle_std"].to_numpy()[mask], color, f"L1 ratio {ratio:g}")
            axis.legend(loc="upper left", handlelength=1.5)
        else:
            plot_curve(axis, alpha, rows["rmsle_mean"].to_numpy(), rows["rmsle_std"].to_numpy(), SERIES[0], None)
        axis.axhline(ols["rmsle_mean"], color=INK_MUTED, linewidth=0.8, zorder=1)
        axis.annotate(f"OLS {ols['rmsle_mean']:.3f}", xy=(1, ols["rmsle_mean"]), xycoords=("axes fraction", "data"),
                      xytext=(0, -3), textcoords="offset points", ha="right", va="top", color=INK_SECONDARY, fontsize=7)
        axis.set_xscale("log")
        axis.set_title(title, loc="left", color=INK)
        axis.set_xlabel("α (log scale)")
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("CV RMSLE")
    axes[0].set_ylim(0, None)
    figure.tight_layout(w_pad=1.0)
    return figure


def horizontal_bars(axis: plt.Axes, labels: list[str], mean: np.ndarray, std: np.ndarray, text: list[str]) -> None:
    positions = np.arange(len(labels))[::-1]
    axis.barh(positions, mean, height=BAR_HEIGHT, color=SERIES[0], zorder=2)
    axis.errorbar(mean, positions, xerr=std, fmt="none", ecolor=INK_SECONDARY, elinewidth=0.8, capsize=2, zorder=3)
    span = (mean + std).max()
    for position, value, error, label in zip(positions, mean, std, text):
        axis.annotate(label, xy=(max(value + error, 0), position), xytext=(4, 0), textcoords="offset points",
                      va="center", ha="left", color=INK, fontsize=7)
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, span * 1.45)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="y", visible=False)


def ablation_figure(ablation_dir: Path) -> plt.Figure:
    summary = pd.read_csv(ablation_dir / "cv_summary.csv").set_index("model")
    folds = pd.read_csv(ablation_dir / "cv_fold_scores.csv")
    reference_mae = folds[folds["model"] == "set_c"].set_index("fold")["mae_eur"]
    mae_delta = (folds.assign(delta=folds["mae_eur"] - folds["fold"].map(reference_mae))
                 .groupby("model")["delta"].agg(["mean", "std"]))

    figure, axes = plt.subplots(1, 3, figsize=(6.5, 2.0))
    sets = summary.loc[list(ABLATION_SETS)]
    horizontal_bars(axes[0], list(ABLATION_SETS.values()), sets["rmsle_mean"].to_numpy(), sets["rmsle_std"].to_numpy(),
                    [f"{v:.3f}" for v in sets["rmsle_mean"]])
    axes[0].set_title("(a) Feature sets", loc="left")
    axes[0].set_xlabel("CV RMSLE")

    changes = summary.loc[list(ABLATION_CHANGES)]
    delta, delta_std = changes["delta_rmsle_vs_set_c_mean"].to_numpy(), changes["delta_rmsle_vs_set_c_std"].to_numpy()
    horizontal_bars(axes[1], list(ABLATION_CHANGES.values()), delta, delta_std, [signed(v, 3) for v in delta])
    axes[1].set_title("(b) Change from Set C: RMSLE", loc="left")
    axes[1].set_xlabel("ΔRMSLE vs Set C")

    mae = mae_delta.loc[list(ABLATION_CHANGES)] / 1e3
    horizontal_bars(axes[2], list(ABLATION_CHANGES.values()), mae["mean"].to_numpy(), mae["std"].to_numpy(),
                    [signed(v, 0, "€", "K") for v in mae["mean"]])
    axes[2].set_yticklabels([])
    axes[2].set_title("(c) Change from Set C: MAE", loc="left")
    axes[2].set_xlabel("ΔMAE vs Set C (€K)")
    figure.tight_layout(w_pad=0.8)
    return figure


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    png_dir, pdf_dir = repo_root / "png", repo_root / "results" / "report"
    png_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save(tuning_figure(repo_root / "results" / "linear" / "main"), "tuning_curves", png_dir, pdf_dir)
    save(ablation_figure(repo_root / "results" / "linear" / "ablation"), "ablation", png_dir, pdf_dir)
    print(f"Figures: {png_dir.relative_to(repo_root)}/(tuning_curves, ablation).png, "
          f"{pdf_dir.relative_to(repo_root)}/(tuning_curves, ablation).pdf")


if __name__ == "__main__":
    main()
