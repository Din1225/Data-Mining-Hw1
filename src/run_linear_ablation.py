"""
程式用途：依作業 §7.2，在相同線性模型、切分與調參流程下比較特徵集，並報告分群指標。

主要執行流程：
1. 讀取 results/linear/main/selected_model.json 的線性變體，使用其完整搜尋空間。
2. 特徵集：
   - Set A：前處理後的全部原始欄位（各 fold 1,163–1,181 欄）。
   - Set B：在每個 fold 的 training subset 上，依與 value_eur 的 |Pearson r| 選出前 k 個 Set A 欄位；
     k = Set C 的特徵數，讓兩者的特徵預算相同。
   - Set C：最終模型使用的精簡 + 工程特徵。
   - Set C 逐一移除一組工程特徵（nonlinear、interaction、domain_rules）。
   - Set C + aggregates：加回試過但未採用的球會／聯賽平均實力，記錄它沒有幫助。
3. 每個特徵集都用同一個搜尋空間、同一組 5-fold 重新選正則化強度；只有特徵集不同。
4. 輸出各特徵集的逐 fold 分數、相對 Set C 的配對差異、OOF 預測，以及依身價、聯賽等級、
   位置與年齡分群的指標至 results/linear/ablation/。

held-out test 不參與。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error

from experiment_utils import (
    FoldData,
    ModelSpec,
    drop_features,
    format_summary,
    load_cv_folds,
    resolve_split_dirs,
    run_experiments,
    summarize_scores,
    write_experiment_outputs,
)
from linear_features import ENGINEERED_GROUPS, build_set_b_folds, build_set_c_folds
from model_registry import LINEAR_VARIANTS
from run_tuning import search_configs


# 依 selector 保留部分欄位，產生新的 folds（移除一組工程特徵用）。
def restrict_folds(folds: list[FoldData], selector: Callable[[pd.DataFrame], np.ndarray]) -> list[FoldData]:
    restricted = []
    for data in folds:
        columns = selector(data.feature_mapping)
        mapping = data.feature_mapping.set_index("output_index").loc[columns].reset_index()
        restricted.append(
            replace(
                data,
                X_train=data.X_train[:, columns],
                X_valid=data.X_valid[:, columns],
                feature_mapping=mapping.assign(output_index=range(len(columns))),
            )
        )
    return restricted


# 分群標籤：只使用 training portion 的原始欄位，依 original_index 對到 OOF 預測。
def slice_labels(split_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(
        split_dir / "train.csv",
        usecols=["original_index", "value_eur", "league_level", "player_positions", "age"],
    )
    value, level, age = train["value_eur"], train["league_level"], train["age"]
    primary = train["player_positions"].str.split(",").str[0].str.strip()
    return pd.DataFrame(
        {
            "original_index": train["original_index"],
            "value_band": np.select(
                [value == 0, value < 1e6, value < 1e7], ["€0", "<€1M", "€1M–10M"], "≥€10M"
            ),
            "league_level": np.select(
                [level.isna(), level == 1, level == 2], ["no club", "level 1", "level 2"], "level 3–4"
            ),
            "position": np.select(
                [primary == "GK", primary.isin(["CB", "LB", "RB", "LWB", "RWB"]),
                 primary.isin(["ST", "CF", "LW", "RW", "LF", "RF"])],
                ["GK", "DEF", "FWD"],
                "MID",
            ),
            "age_band": np.select([age <= 21, age <= 29], ["≤21", "22–29"], "≥30"),
        }
    )


# 每個分群、每個特徵集的 RMSLE、MAE 與中位數相對誤差（只算正身價球員）。
def slice_metrics(oof: pd.DataFrame, labels: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    frame = oof.merge(labels, on="original_index", validate="one_to_one")
    rows = []
    for dimension in ["value_band", "league_level", "position", "age_band"]:
        for value, group in frame.groupby(dimension):
            positive = group["y_true"] > 0
            for model in models:
                relative = (group.loc[positive, model] / group.loc[positive, "y_true"] - 1).abs()
                rows.append(
                    {
                        "dimension": dimension,
                        "slice": value,
                        "model": model,
                        "n": len(group),
                        "rmsle": root_mean_squared_log_error(group["y_true"], group[model]),
                        "mae_eur": mean_absolute_error(group["y_true"], group[model]),
                        "median_abs_pct_error": relative.median() if positive.any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# 主流程：每個特徵集都用同一搜尋空間選出最佳設定，再以最佳設定產生 OOF 預測與比較表。
def main() -> None:
    repo_root, split_dir, processed_dir = resolve_split_dirs("src/run_linear_ablation.py")
    selected = json.loads((repo_root / "results" / "linear" / "main" / "selected_model.json").read_text(encoding="utf-8"))
    family = LINEAR_VARIANTS[selected["variant"]]
    output_dir = repo_root / "results" / "linear" / "ablation"

    set_a = load_cv_folds(split_dir, processed_dir)
    set_c = build_set_c_folds(split_dir, set_a)
    k = set_c[0].X_train.shape[1]
    feature_sets = {
        "set_a": set_a,
        f"set_b_top{k}": build_set_b_folds(set_a, k),
        "set_c": set_c,
        **{
            f"set_c_minus_{group}": restrict_folds(set_c, drop_features(transformers=[group]))
            for group in ENGINEERED_GROUPS
        },
        "set_c_plus_aggregates": build_set_c_folds(split_dir, set_a, exclude_groups=()),
    }

    score_frames, oof, searches, specs = [], None, [], []
    for name, folds in feature_sets.items():
        _, search = search_configs(name, family, "log1p", folds)
        searches.append(search.assign(feature_set=name))
        params = json.loads(search.loc[0, "params"])
        spec = ModelSpec(name, partial(family.build, params), log_target=True)
        specs.append(spec)
        scores, _, set_oof = run_experiments([spec], folds)
        score_frames.append(scores.assign(params=json.dumps(params)))
        oof = set_oof if oof is None else oof.merge(set_oof[["original_index", name]], on="original_index", validate="one_to_one")

    fold_scores = pd.concat(score_frames, ignore_index=True)
    summary = summarize_scores(fold_scores.drop(columns="params"))
    reference = fold_scores[fold_scores["model"] == "set_c"].set_index("fold")["rmsle"]
    paired = (
        fold_scores.assign(delta=lambda d: d["rmsle"] - d["fold"].map(reference))
        .groupby("model")["delta"]
        .agg(delta_rmsle_vs_set_c_mean="mean", delta_rmsle_vs_set_c_std="std",
             folds_set_c_better=lambda s: int((s > 0).sum()))
    )
    summary = summary.merge(paired, left_on="model", right_index=True)
    summary.insert(2, "params", summary["model"].map(fold_scores.groupby("model")["params"].first()))
    summary.insert(3, "feature_count", summary["model"].map(fold_scores.groupby("model")["feature_count"].mean()))

    write_experiment_outputs(output_dir, specs, fold_scores, summary, oof)
    pd.concat(searches, ignore_index=True).to_csv(output_dir / "search_summary.csv", index=False, float_format="%.6f")
    slices = slice_metrics(oof, slice_labels(split_dir), list(feature_sets))
    slices.to_csv(output_dir / "slice_metrics.csv", index=False, float_format="%.6f")

    print(f"Variant: {selected['variant']}; Set B k = {k}")
    print(format_summary(summary))
    print(summary[["model", "feature_count", "params", "delta_rmsle_vs_set_c_mean",
                   "delta_rmsle_vs_set_c_std", "folds_set_c_better"]].round(4).to_string(index=False))
    print(f"Results: {output_dir.relative_to(repo_root)}")
    print("Held-out test accessed: False")


if __name__ == "__main__":
    main()
