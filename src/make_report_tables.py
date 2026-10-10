"""
程式用途：把 results/ 下的 CV、消融、decision threshold 與 test 結果整理成報告用的表格（Markdown 與 LaTeX booktabs）。

表格：
1. main_cv：主實驗表（三層 baseline 與最終模型，5-fold CV mean ± std，每欄最佳值粗體）。
2. linear_variants：四種線性變體的搜尋範圍、選到的超參數與 CV RMSLE。
3. ablation：§7.2 特徵集比較，含相對 Set C 的配對差異與 paired t-test。
4. slices：最終模型與 strong baseline 在各分群的 CV 表現（較佳者粗體）。
5. threshold_sensitivity：門檻改為 ±10%／±30%／±50% 時各模型在門檻內的比例。
6. test：held-out test 唯一一次評估的結果（只讀取既有檔案）。

每個結果都與 decision threshold（±20% 相對誤差，見 doc/Decision_Threshold.md）比較：
- 「Within ±20% (%)」欄：身價 > 0 的球員中，預測在門檻內的比例。
- RMSLE 後的 †：RMSLE ≤ ln(1.2) ≈ 0.182，也就是典型相對誤差 exp(RMSLE) − 1 在門檻內。
門檻指標來自 results/threshold/（由 threshold_metrics.py 產生，需先執行）。

只讀取既有結果檔，不訓練模型，也不重新評估 test。LaTeX 版本需要 booktabs 與 eurosym 套件。
輸出至 results/report/。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import DECISION_THRESHOLD
from model_registry import LINEAR_VARIANTS


MAIN_MODELS = {
    "trivial_mean": "Trivial (training mean)",
    "simple_ols_overall": "Simple (OLS on overall)",
    "strong_random_forest": "Strong (Random Forest)",
    "ours_ols": "Ours (OLS, Set C)",
}
# (欄位, 表頭, 越低越好, 換算倍率, 小數位數)
SCORE_COLUMNS = [
    ("rmsle", "RMSLE", True, 1, 3),
    ("rmsle_positive", "RMSLE (value > 0)", True, 1, 3),
    ("mae_eur", "MAE (€K)", True, 1e-3, 0),
    ("medae_eur", "MedAE (€K)", True, 1e-3, 0),
    ("r2_eur", "R²", False, 1, 3),
]
RMSLE_COLUMNS = {"rmsle", "rmsle_positive"}
# 整體比較用的 log 門檻：RMSLE ≤ ln(1 + 20%) 等價於典型相對誤差 exp(RMSLE) − 1 ≤ 20%。
LOG_THRESHOLD = float(np.log1p(DECISION_THRESHOLD))
WITHIN = f"Within ±{DECISION_THRESHOLD:.0%} (%)"
THRESHOLD_NOTE = (
    f"Decision threshold: ±{DECISION_THRESHOLD:.0%} relative error. Within: share of players with value > 0 whose "
    f"prediction is within the threshold. †: RMSLE ≤ ln(1.2) = {LOG_THRESHOLD:.3f}, i.e. the typical error is within "
    "the threshold."
)
ABLATION_SETS = {
    "set_a": "A: processed raw columns",
    "set_b_top31": "B: top-31 by |Pearson r| with value",
    "set_b_top31_log_ranked": "B': top-31 by |Pearson r| with log1p(value)",
    "set_c": "C: ours (final)",
    "set_c_minus_nonlinear": "C − nonlinear (overall², age²)",
    "set_c_minus_interaction": "C − interaction (overall × reputation)",
    "set_c_minus_domain_rules": "C − domain rules (free agent, age ≥ 40, top-5)",
    "set_c_minus_overall": "C − overall and derived (H1)",
    "set_c_minus_reputation": "C − reputation and derived (H2)",
    "set_c_plus_aggregates": "C + club/league mean overall",
}
SLICE_MODELS = {"strong_random_forest": "RF", "ours_ols": "Ours"}
SLICE_ORDER = {
    "value_band": ["€0", "<€1M", "€1M–10M", "≥€10M"],
    "league_level": ["level 1", "level 2", "level 3–4", "no club"],
    "position": ["GK", "DEF", "MID", "FWD"],
    "age_band": ["≤21", "22–29", "≥30"],
}
# 逐字元替換，避免先替換出的反斜線或大括號被再次跳脫。
LATEX_REPLACEMENTS = {
    "\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "&": "\\&", "%": "\\%", "_": "\\_", "#": "\\#",
    "€": "\\euro{}", "±": "$\\pm$", "≥": "$\\geq$", "≤": "$\\leq$", "<": "$<$", ">": "$>$",
    "–": "--", "—": "---", "−": "$-$", "×": "$\\times$", "²": "$^2$", "|": "$|$", "Δ": "$\\Delta$",
    "†": "$^\\dagger$",
}


# 四捨五入後加 0.0，避免顯示 -0.000。
def number(value: float, scale: float, digits: int) -> str:
    return f"{round(value * scale, digits) + 0.0:,.{digits}f}"


def mean_std(mean: float, std: float, scale: float, digits: int) -> str:
    return f"{number(mean, scale, digits)} ± {number(std, scale, digits)}"


# RMSLE 低於 log 門檻時加註 †。
def mark_threshold(text: str, rmsle: float) -> str:
    return f"{text} †" if rmsle <= LOG_THRESHOLD else text


# 每欄最佳值的位置（依 mean 比較）。
def best_mask(values: pd.DataFrame, lower_is_better: dict[str, bool]) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=values.index, columns=values.columns)
    for column, lower in lower_is_better.items():
        target = values[column].min() if lower else values[column].max()
        mask[column] = np.isclose(values[column], target)
    return mask


def to_markdown(table: pd.DataFrame, bold: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "|".join(["---"] * len(table.columns)) + "|"]
    for index, row in table.iterrows():
        cells = [str(value).replace("|", "\\|") for value in row]
        cells = [f"**{cell}**" if bold.at[index, column] else cell for cell, column in zip(cells, table.columns)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return "".join(LATEX_REPLACEMENTS.get(character, character) for character in text)


def to_latex(table: pd.DataFrame, bold: pd.DataFrame, caption: str, label: str) -> str:
    align = "l" + "r" * (len(table.columns) - 1)
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
        " & ".join(latex_escape(column) for column in table.columns) + " \\\\", "\\midrule",
    ]
    for index, row in table.iterrows():
        cells = [latex_escape(str(value)) for value in row]
        cells = [f"\\textbf{{{cell}}}" if bold.at[index, column] else cell for cell, column in zip(cells, table.columns)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{latex_escape(caption)}}}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(lines)


# 主實驗表或 test 表：summary、within 為每個模型一列；std_suffix 為 None 時沒有 std（test 只有一次評估）。
def score_table(summary: pd.DataFrame, within: pd.DataFrame, std_suffix: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.set_index("model").loc[list(MAIN_MODELS)]
    within = within.set_index("model").loc[list(MAIN_MODELS)]
    table = pd.DataFrame({"Model": list(MAIN_MODELS.values())})
    means = pd.DataFrame(index=table.index)
    for column, header, _, scale, digits in SCORE_COLUMNS:
        mean = summary[f"{column}_mean" if std_suffix else column].to_numpy()
        means[header] = mean
        if std_suffix:
            cells = [mean_std(m, s, scale, digits) for m, s in zip(mean, summary[f"{column}{std_suffix}"])]
        else:
            cells = [number(m, scale, digits) for m in mean]
        if column in RMSLE_COLUMNS:
            cells = [mark_threshold(cell, m) for cell, m in zip(cells, mean)]
        table[header] = cells
    # 在門檻內的比例放在兩個 RMSLE 欄之後。
    within_mean = within["within_threshold_mean" if std_suffix else "within_threshold"].to_numpy()
    if std_suffix:
        cells = [mean_std(m, s, 100, 1) for m, s in zip(within_mean, within["within_threshold_std"])]
    else:
        cells = [number(m, 100, 1) for m in within_mean]
    table.insert(3, WITHIN, cells)
    means[WITHIN] = within_mean
    lower = {header: lower for _, header, lower, _, _ in SCORE_COLUMNS} | {WITHIN: False}
    bold = best_mask(means, lower)
    return table, bold.reindex(columns=table.columns, fill_value=False)


def linear_variant_table(main_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = pd.read_csv(main_dir / "linear_variants.csv")
    selected = json.loads((main_dir / "selected_model.json").read_text(encoding="utf-8"))["variant"]
    rows = []
    for row in variants.itertuples():
        space = LINEAR_VARIANTS[row.variant].search_space
        search = "; ".join(f"{name} ∈ [{min(values):g}, {max(values):g}] ({len(values)} values)"
                           for name, values in space.items()) or "—"
        rows.append({
            "Variant": row.variant + (" (selected)" if row.variant == selected else ""),
            "Search space": search,
            "Best setting": ", ".join(f"{k} = {v:g}" for k, v in ast.literal_eval(row.params).items()) or "—",
            "CV RMSLE": f"{row.cv_rmsle_mean:.4f} ± {row.cv_rmsle_std:.4f}",
        })
    table = pd.DataFrame(rows)
    return table, pd.DataFrame(False, index=table.index, columns=table.columns)


def ablation_table(ablation_dir: Path, threshold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ablation_dir / "cv_summary.csv").set_index("model").loc[list(ABLATION_SETS)]
    within = pd.read_csv(threshold_dir / "cv_ablation_summary.csv").set_index("model").loc[list(ABLATION_SETS)]
    counts = pd.read_csv(ablation_dir / "cv_fold_scores.csv").groupby("model")["feature_count"].agg(["min", "max"])
    table = pd.DataFrame({
        "Feature set": list(ABLATION_SETS.values()),
        "#Features": [f"{counts.at[m, 'min']:,}" if counts.at[m, "min"] == counts.at[m, "max"]
                      else f"{counts.at[m, 'min']:,}–{counts.at[m, 'max']:,}" for m in summary.index],
        "RMSLE": [mark_threshold(mean_std(m, s, 1, 3), m) for m, s in zip(summary["rmsle_mean"], summary["rmsle_std"])],
        WITHIN: [mean_std(m, s, 100, 1) for m, s in zip(within["within_threshold_mean"], within["within_threshold_std"])],
        "ΔRMSLE vs C": ["—" if m == "set_c" else mean_std(d, s, 1, 3) for m, d, s in
                        zip(summary.index, summary["delta_rmsle_vs_set_c_mean"], summary["delta_rmsle_vs_set_c_std"])],
        "Folds worse than C": ["—" if m == "set_c" else f"{n}/5" for m, n in zip(summary.index, summary["folds_set_c_better"])],
        "Paired t p": ["—" if m == "set_c" else (f"{p:.3f}" if p >= 0.001 else "< 0.001")
                       for m, p in zip(summary.index, summary["paired_t_p_vs_set_c"])],
        "MAE (€K)": [mean_std(m, s, 1e-3, 0) for m, s in zip(summary["mae_eur_mean"], summary["mae_eur_std"])],
        "R²": [mean_std(m, s, 1, 3) for m, s in zip(summary["r2_eur_mean"], summary["r2_eur_std"])],
    })
    bold = pd.DataFrame(False, index=table.index, columns=table.columns)
    bold.loc[list(summary.index).index("set_c"), "Feature set"] = True
    return table, bold


def slice_table(ablation_dir: Path, threshold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = [(dimension, value) for dimension, values in SLICE_ORDER.items() for value in values]

    def wide(path: Path, metric: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame = frame[frame["model"].isin(SLICE_MODELS)]
        return frame.pivot_table(index=["dimension", "slice"], columns="model", values=metric, dropna=False).loc[order]

    rmsle = wide(ablation_dir / "slice_metrics_main_models.csv", "rmsle")
    within = wide(threshold_dir / "cv_slices.csv", "within_threshold")
    counts = wide(ablation_dir / "slice_metrics_main_models.csv", "n")
    table = pd.DataFrame({
        "Dimension": rmsle.index.get_level_values(0).str.replace("_", " "),
        "Slice": rmsle.index.get_level_values(1),
        "n": counts["ours_ols"].astype(int).map("{:,}".format).to_numpy(),
    })
    bold = pd.DataFrame(False, index=table.index, columns=table.columns)
    for values, header, scale, digits, lower in [(rmsle, "RMSLE", 1, 3, True), (within, WITHIN, 100, 1, False)]:
        values = values[list(SLICE_MODELS)]
        better = values.fillna(np.inf).idxmin(axis=1) if lower else values.fillna(-np.inf).idxmax(axis=1)
        for model, short in SLICE_MODELS.items():
            column = f"{header} {short}"
            table[column] = [("—" if np.isnan(v) else f"{v * scale:.{digits}f}") for v in values[model]]
            bold[column] = (better.to_numpy() == model) & values.notna().all(axis=1).to_numpy()
    return table, bold


def sensitivity_table(threshold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = pd.read_csv(threshold_dir / "cv_sensitivity.csv")
    wide = sensitivity.pivot(index="model", columns="threshold", values="within_threshold_mean").loc[list(MAIN_MODELS)]
    headers = {t: f"±{t:.0%}" + (" (decided)" if np.isclose(t, DECISION_THRESHOLD) else "") for t in wide.columns}
    table = pd.DataFrame({"Model": list(MAIN_MODELS.values())})
    means = pd.DataFrame(index=table.index)
    for threshold, header in headers.items():
        table[header] = [number(v, 100, 1) for v in wide[threshold]]
        means[header] = wide[threshold].to_numpy()
    bold = best_mask(means, {header: False for header in headers.values()})
    return table, bold.reindex(columns=table.columns, fill_value=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    main_dir = repo_root / "results" / "linear" / "main"
    ablation_dir = repo_root / "results" / "linear" / "ablation"
    threshold_dir = repo_root / "results" / "threshold"
    output_dir = repo_root / "results" / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "main_cv": (score_table(pd.read_csv(main_dir / "cv_summary.csv"),
                                pd.read_csv(threshold_dir / "cv_main_summary.csv"), "_std"),
                    "5-fold CV on the training portion (mean ± std over folds). Bold: best mean per column. "
                    + THRESHOLD_NOTE),
        "linear_variants": (linear_variant_table(main_dir),
                            "Linear variants on Set C, tuned by 5-fold CV. Every penalized variant selects the "
                            "smallest penalty in its grid, i.e. it reduces to OLS."),
        "ablation": (ablation_table(ablation_dir, threshold_dir),
                     "Feature-set ablation with the final OLS (same folds and tuning). Δ and paired t-test are "
                     "per-fold differences from Set C. " + THRESHOLD_NOTE),
        "slices": (slice_table(ablation_dir, threshold_dir),
                   f"CV out-of-fold error by slice. Within ±{DECISION_THRESHOLD:.0%}: share of players with value > 0 "
                   "whose prediction is within the decision threshold (undefined for €0 players). Bold: the better model."),
        "threshold_sensitivity": (sensitivity_table(threshold_dir),
                                  "Share of players with value > 0 whose CV prediction is within each candidate "
                                  "relative-error threshold (%, mean over 5 folds). The model ranking is the same at "
                                  "every threshold. Bold: best per column."),
        "test": (score_table(pd.read_csv(repo_root / "results" / "final_test" / "test_scores.csv"),
                             pd.read_csv(threshold_dir / "test_summary.csv"), None),
                 "Held-out test set (evaluated once, after all model and feature decisions were fixed). "
                 "Bold: best per column. "
                 + THRESHOLD_NOTE),
    }
    markdown = []
    for name, ((table, bold), caption) in tables.items():
        (output_dir / f"{name}.tex").write_text(to_latex(table, bold, caption, f"tab:{name}") + "\n", encoding="utf-8")
        markdown.append(f"## {name}\n\n{caption}\n\n{to_markdown(table, bold)}\n")
    (output_dir / "tables.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    print(f"Results: {output_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
