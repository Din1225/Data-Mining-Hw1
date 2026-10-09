"""
程式用途：在固定 5-fold CV 上為 strong baseline（RandomForest，log1p target）搜尋超參數，並提供共用的搜尋函式。

主要執行流程：
1. 讀取 processed 5-fold 資料（Set A）。
2. 搜尋時 training 與 validation 都只保留 value_eur > 0 的球員：身價為 0 的 93 位球員在 log 尺度上
   誤差極大，會主導 RMSLE，使超參數的選擇只反映少數球員。最終 baseline 以選出的超參數訓練全部球員。
3. 以 validation RMSLE 選出最佳設定，輸出所有設定的逐 fold 分數、摘要與 best_params.json
   至 results/tuning/random_forest_log1p/。

search_configs() 也供 run_linear_models.py 與 run_linear_ablation.py 搜尋線性模型的正則化強度。

執行方式：
FC26_SPLIT_DIR=data/splits FC26_PROCESSED_DIR=data/processed_splits python src/run_tuning.py

搜尋與選模只使用 training portion 的 CV；held-out test 不參與。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_utils import (
    METRIC_COLUMNS,
    FoldData,
    ModelSpec,
    load_cv_folds,
    positive_only,
    resolve_split_dirs,
    run_cv,
)
from model_registry import STRONG_BASELINE, STRONG_BASELINE_NAME, ModelFamily


# 寫出一項搜尋的逐 fold 分數、摘要與最佳設定。
def write_search_outputs(
    output_dir: Path, fold_scores: pd.DataFrame, summary: pd.DataFrame, best: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(output_dir / "search_fold_scores.csv", index=False, float_format="%.6f")
    summary.to_csv(output_dir / "search_summary.csv", index=False, float_format="%.6f")
    (output_dir / "best_params.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# 在指定 folds 上評估一類模型 × target 的所有設定，回傳逐 fold 分數與依 RMSLE 排序的摘要。
def search_configs(
    model_name: str, family: ModelFamily, target: str, folds: list[FoldData]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = family.configs()
    score_frames = []
    for config_id, params in enumerate(configs):
        spec = ModelSpec(
            model_name,
            partial(family.build, params),
            log_target=target == "log1p",
            dense=family.dense,
        )
        fold_scores, _ = run_cv(spec, folds)
        fold_scores.insert(0, "config_id", config_id)
        fold_scores.insert(1, "params", json.dumps(params))
        score_frames.append(fold_scores)
        print(
            f"[{model_name}] config {config_id + 1}/{len(configs)} "
            f"RMSLE {fold_scores['rmsle'].mean():.4f} {params}",
            flush=True,
        )

    fold_scores = pd.concat(score_frames, ignore_index=True)
    summary = (
        fold_scores.groupby(["config_id", "params"])[[*METRIC_COLUMNS, "fit_seconds"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("rmsle_mean").reset_index(drop=True)
    return fold_scores, summary


# 主流程：在正身價球員上搜尋 strong baseline 的超參數。
def main() -> None:
    repo_root, split_dir, processed_dir = resolve_split_dirs("src/run_tuning.py")
    positive_folds = positive_only(load_cv_folds(split_dir, processed_dir))
    model_name = f"{STRONG_BASELINE_NAME}_log1p"

    fold_scores, summary = search_configs(model_name, STRONG_BASELINE, "log1p", positive_folds)
    best_row = summary.iloc[0]
    best = {
        "model": STRONG_BASELINE_NAME,
        "target": "log1p",
        "params": json.loads(best_row["params"]),
        "cv_rmsle_positive_mean": float(best_row["rmsle_mean"]),
        "cv_rmsle_positive_std": float(best_row["rmsle_std"]),
        "n_configs": len(summary),
    }
    write_search_outputs(repo_root / "results" / "tuning" / model_name, fold_scores, summary, best)
    print(f"[{model_name}] best {best}")
    print("Held-out test accessed: False")


if __name__ == "__main__":
    main()
