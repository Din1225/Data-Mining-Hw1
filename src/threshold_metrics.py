"""
程式用途：以組內決定的 decision threshold 評估每個結果，讓報告中的每個結果都能與門檻比較。

Decision threshold（2026-10-09 決定，理由與使用方式見 doc/Decision_Threshold.md）：
- 身價 > 0 的球員：相對誤差 |預測 / 實際 − 1| ≤ 20%（experiment_utils.DECISION_THRESHOLD）視為
  「在門檻內」，也就是這個預測不會改變球探對報價的判斷。主要指標是在門檻內的球員比例。
- 身價為 0 的球員：相對誤差沒有定義，不納入比例；另外回報人數與預測值的最大值。
- 整體比較：RMSLE ≤ ln(1 + 20%) ≈ 0.182，等價於典型相對誤差 exp(RMSLE) − 1 ≤ 20%
  （由 make_report_tables.py 在表格中標示）。

輸出至 results/threshold/：
- cv_main_fold_scores.csv、cv_main_summary.csv：主實驗各模型（5-fold CV 的 OOF 預測）。
- cv_ablation_fold_scores.csv、cv_ablation_summary.csv：消融實驗各特徵集，含相對 Set C 的配對差異。
- cv_slices.csv：主實驗各模型在各分群（身價、聯賽等級、位置、年齡）的結果。
- cv_sensitivity.csv：門檻改為 ±10%／±30%／±50% 時主實驗各模型的結果，檢查結論是否取決於門檻的選擇。
- test_summary.csv：held-out test 唯一一次評估的既有預測（results/final_test/test_predictions.csv）。
- test_player_errors.csv：test 每位球員在各模型下的相對誤差與是否在門檻內，供顯著性檢定與錯誤案例分析使用。

只讀取既有的預測檔，不訓練任何模型，也不重新評估 test。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from experiment_utils import DECISION_THRESHOLD, ID_COLUMNS, resolve_split_dirs
from run_linear_ablation import slice_labels


ID_AND_LABEL_COLUMNS = {*ID_COLUMNS, "fold", "y_true"}
SENSITIVITY_THRESHOLDS = [0.1, 0.2, 0.3, 0.5]


def model_columns(predictions: pd.DataFrame) -> list[str]:
    return [column for column in predictions.columns if column not in ID_AND_LABEL_COLUMNS]


# 一組預測（一個模型在一個 fold、一個分群或整個 test）的門檻指標；相對誤差只對身價 > 0 的球員有定義。
def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = DECISION_THRESHOLD) -> dict[str, float]:
    positive = y_true > 0
    relative_error = np.abs(y_pred[positive] / y_true[positive] - 1)
    zero_predictions = y_pred[~positive]
    return {
        "n_positive": int(positive.sum()),
        "within_threshold": float(np.mean(relative_error <= threshold)) if relative_error.size else np.nan,
        "median_abs_pct_error": float(np.median(relative_error)) if relative_error.size else np.nan,
        "n_zero": int((~positive).sum()),
        "zero_pred_max_eur": float(zero_predictions.max()) if zero_predictions.size else np.nan,
    }


def fold_scores(oof: pd.DataFrame, models: list[str], threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    rows = []
    for fold, group in oof.groupby("fold"):
        y_true = group["y_true"].to_numpy(dtype=float)
        for model in models:
            metrics = threshold_metrics(y_true, group[model].to_numpy(dtype=float), threshold)
            rows.append({"model": model, "fold": fold, **metrics})
    return pd.DataFrame(rows)


# 逐 fold 結果整理成 mean ± std；€0 球員的人數加總，預測最大值取所有 fold 中的最大值。
def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("model", sort=False)
        .agg(
            within_threshold_mean=("within_threshold", "mean"),
            within_threshold_std=("within_threshold", "std"),
            median_abs_pct_error_mean=("median_abs_pct_error", "mean"),
            median_abs_pct_error_std=("median_abs_pct_error", "std"),
            n_positive=("n_positive", "sum"),
            n_zero=("n_zero", "sum"),
            zero_pred_max_eur=("zero_pred_max_eur", "max"),
        )
        .reset_index()
    )


# 消融實驗：同一個 fold 上與 Set C 相減；Set C 較好表示其他特徵集在門檻內的比例較低。
def paired_vs_set_c(scores: pd.DataFrame) -> pd.DataFrame:
    reference = scores[scores["model"] == "set_c"].set_index("fold")["within_threshold"]
    deltas = scores.assign(delta=scores["within_threshold"] - scores["fold"].map(reference))
    return deltas.groupby("model", sort=False)["delta"].agg(
        delta_within_vs_set_c_mean="mean",
        delta_within_vs_set_c_std="std",
        folds_set_c_better=lambda s: int((s < 0).sum()),
        paired_t_p_vs_set_c=lambda s: stats.ttest_1samp(s, 0).pvalue if s.any() else np.nan,
    )


# 分群結果：合併 5 個 fold 的 OOF 預測後再計算（與 slice_metrics 的做法相同）。
def slice_scores(oof: pd.DataFrame, labels: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    frame = oof.merge(labels, on="original_index", validate="one_to_one")
    rows = []
    for dimension in ["value_band", "league_level", "position", "age_band"]:
        for value, group in frame.groupby(dimension):
            y_true = group["y_true"].to_numpy(dtype=float)
            for model in models:
                metrics = threshold_metrics(y_true, group[model].to_numpy(dtype=float))
                rows.append({"dimension": dimension, "slice": value, "model": model, "n": len(group), **metrics})
    return pd.DataFrame(rows)


# 門檻改成其他候選值時，各模型在門檻內的比例與排名（1 = 比例最高）。
def sensitivity(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    frames = []
    for threshold in SENSITIVITY_THRESHOLDS:
        scores = fold_scores(oof, models, threshold)
        summary = scores.groupby("model", sort=False)["within_threshold"].agg(
            within_threshold_mean="mean", within_threshold_std="std"
        )
        frames.append(summary.reset_index().assign(threshold=threshold, decided=np.isclose(threshold, DECISION_THRESHOLD)))
    table = pd.concat(frames, ignore_index=True)
    table["rank"] = table.groupby("threshold")["within_threshold_mean"].rank(ascending=False, method="min").astype(int)
    return table[["threshold", "decided", "model", "within_threshold_mean", "within_threshold_std", "rank"]]


# test 每位球員的相對誤差與是否在門檻內；身價為 0 的球員兩欄皆為空值。
def player_errors(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    y_true = predictions["y_true"].to_numpy(dtype=float)
    positive = y_true > 0
    table = predictions[[*ID_COLUMNS, "y_true"]].copy()
    for model in models:
        relative_error = np.full(len(y_true), np.nan)
        relative_error[positive] = np.abs(predictions[model].to_numpy(dtype=float)[positive] / y_true[positive] - 1)
        within = pd.array(np.where(positive, relative_error <= DECISION_THRESHOLD, False), dtype="boolean")
        within[~positive] = pd.NA
        table[f"{model}_rel_error"] = relative_error
        table[f"{model}_within_threshold"] = within
    return table


def main() -> None:
    repo_root, split_dir, _ = resolve_split_dirs("src/threshold_metrics.py")
    main_dir = repo_root / "results" / "linear" / "main"
    ablation_dir = repo_root / "results" / "linear" / "ablation"
    output_dir = repo_root / "results" / "threshold"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads((main_dir / "selected_model.json").read_text(encoding="utf-8"))["variant"]
    reported = ["trivial_mean", "simple_ols_overall", "strong_random_forest", f"ours_{selected}"]

    main_oof = pd.read_csv(main_dir / "oof_predictions.csv")
    main_scores = fold_scores(main_oof, model_columns(main_oof))
    main_summary = summarize(main_scores)

    ablation_oof = pd.read_csv(ablation_dir / "oof_predictions.csv")
    ablation_scores = fold_scores(ablation_oof, model_columns(ablation_oof))
    ablation_summary = summarize(ablation_scores).merge(paired_vs_set_c(ablation_scores), left_on="model", right_index=True)

    slices = slice_scores(main_oof, slice_labels(split_dir), reported)
    sensitivity_table = sensitivity(main_oof, reported)

    test = pd.read_csv(repo_root / "results" / "final_test" / "test_predictions.csv")
    test_models = model_columns(test)
    y_test = test["y_true"].to_numpy(dtype=float)
    test_summary = pd.DataFrame(
        [{"model": model, **threshold_metrics(y_test, test[model].to_numpy(dtype=float))} for model in test_models]
    )

    outputs = {
        "cv_main_fold_scores": main_scores,
        "cv_main_summary": main_summary,
        "cv_ablation_fold_scores": ablation_scores,
        "cv_ablation_summary": ablation_summary,
        "cv_slices": slices,
        "cv_sensitivity": sensitivity_table,
        "test_summary": test_summary,
        "test_player_errors": player_errors(test, test_models),
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.6f")

    shown = ["model", "within_threshold_mean", "within_threshold_std", "median_abs_pct_error_mean", "n_zero", "zero_pred_max_eur"]
    with pd.option_context("display.width", 250):
        print(f"Decision threshold: relative error <= {DECISION_THRESHOLD:.0%} (value > 0); "
              f"RMSLE <= ln(1 + {DECISION_THRESHOLD:g}) = {np.log1p(DECISION_THRESHOLD):.4f}")
        print("\n[CV main]\n" + main_summary.loc[main_summary["model"].isin(reported), shown].round(4).to_string(index=False))
        print("\n[CV ablation]\n" + ablation_summary[[*shown, "delta_within_vs_set_c_mean", "folds_set_c_better",
                                                       "paired_t_p_vs_set_c"]].round(4).to_string(index=False))
        print("\n[CV sensitivity]\n" + sensitivity_table.pivot(index="model", columns="threshold",
                                                               values="within_threshold_mean").loc[reported].round(4).to_string())
        print("\n[Test]\n" + test_summary.round(4).to_string(index=False))
    print(f"\nResults: {output_dir.relative_to(repo_root)}")
    print("Held-out test predictions read from results/final_test/ (existing; no model refit)")


if __name__ == "__main__":
    main()
