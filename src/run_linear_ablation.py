"""
程式用途：依作業 §7.2，在相同線性模型、切分與調參流程下比較特徵集，並報告分群指標。

主要執行流程：
1. 讀取 results/linear/main/selected_model.json 的線性變體，使用其完整搜尋空間。
2. 特徵集：
   - Set A：前處理後的原始欄位。part 1 前處理另外加入的兩個工程欄位（overall²、
     overall × international_reputation，transformer = engineered）不是原始欄位，因此排除。
   - Set B：在每個 fold 的 training subset 上，依與 value_eur 的 |Pearson r| 選出前 k 個 Set A 欄位；
     k = Set C 的特徵數，讓兩者的特徵預算相同。另以 log1p(value_eur) 排序的版本做 robustness 檢查。
   - Set C：最終模型使用的精簡 + 工程特徵。
   - Set C 逐一移除一組工程特徵（nonlinear、interaction、domain_rules）。
   - Set C 移除 part 1 假設的特徵（H1：overall 及其衍生特徵；H2：international_reputation 及其衍生特徵）。
   - Set C + aggregates：加回試過但未採用的球會／聯賽平均實力，記錄它沒有幫助。
3. 每個特徵集都用同一個搜尋空間、同一組 5-fold 重新選正則化強度；只有特徵集不同。
4. 輸出至 results/linear/ablation/：
   - 各特徵集的逐 fold 分數、相對 Set C 的配對差異（含 paired t-test）與 OOF 預測。
   - Set B 每個 fold 選到的欄位，並標出與排名更前面的已選欄位完全共線者。
   - 依身價、聯賽等級、位置與年齡分群的指標：特徵集之間（slice_metrics.csv），以及主實驗的
     baseline 與最終模型之間（slice_metrics_main_models.csv，使用 results/linear/main 的 OOF 預測）。

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
from scipy import stats
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


# part 1 的假設（doc/Exploratory_Feature_Analysis_and_Feature_Engineering.md）：
# 移除該欄位，以及 Set C 中所有由它衍生的特徵。
HYPOTHESIS_FEATURES = {
    "overall": ["overall", "overall_sq_centered", "overall_x_reputation"],  # H1
    "reputation": ["international_reputation", "overall_x_reputation"],  # H2
}
# 判定兩欄完全共線（相同、互補或只差線性轉換）的 |r| 門檻。
COLLINEAR_TOLERANCE = 1e-9


# 依 selector 保留部分欄位，產生新的 folds（Set A 排除工程欄位、Set C 移除特徵群組用）。
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


# Set B 每個 fold 選到的欄位，依 |r| 排名；collinear_with 記錄與它完全共線、排名更前面的已選欄位
# （例如 cm／lcm／rcm 三欄的 base 分數完全相同），這些欄位在線性模型中不提供新資訊。
def set_b_selections(name: str, folds: list[FoldData]) -> pd.DataFrame:
    frames = []
    for data in folds:
        mapping = data.feature_mapping.assign(abs_r=lambda m: m["pearson_r"].abs())
        order = mapping.sort_values("abs_r", ascending=False, kind="stable")["output_index"].to_numpy()
        corr = np.nan_to_num(np.corrcoef(data.X_train[:, order].toarray(), rowvar=False))
        names = mapping.set_index("output_index").loc[order, "output_feature"].tolist()
        collinear_with = [
            next((names[j] for j in range(i) if abs(corr[i, j]) > 1 - COLLINEAR_TOLERANCE), "")
            for i in range(len(order))
        ]
        frames.append(
            mapping.set_index("output_index")
            .loc[order, ["output_feature", "transformer", "pearson_r"]]
            .reset_index(drop=True)
            .assign(feature_set=name, fold=data.fold, rank=range(1, len(order) + 1), collinear_with=collinear_with)
        )
    columns = ["feature_set", "fold", "rank", "output_feature", "transformer", "pearson_r", "collinear_with"]
    return pd.concat(frames, ignore_index=True)[columns]


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


# 每個分群、每個模型的 RMSLE（全部與只算正身價）、MAE 與中位數相對誤差（只算正身價球員）。
def slice_metrics(oof: pd.DataFrame, labels: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    frame = oof.merge(labels, on="original_index", validate="one_to_one")
    rows = []
    for dimension in ["value_band", "league_level", "position", "age_band"]:
        for value, group in frame.groupby(dimension):
            positive = group["y_true"] > 0
            for model in models:
                y_pos, pred_pos = group.loc[positive, "y_true"], group.loc[positive, model]
                rows.append(
                    {
                        "dimension": dimension,
                        "slice": value,
                        "model": model,
                        "n": len(group),
                        "n_positive": int(positive.sum()),
                        "rmsle": root_mean_squared_log_error(group["y_true"], group[model]),
                        "rmsle_positive": root_mean_squared_log_error(y_pos, pred_pos) if positive.any() else np.nan,
                        "mae_eur": mean_absolute_error(group["y_true"], group[model]),
                        "median_abs_pct_error": (pred_pos / y_pos - 1).abs().median() if positive.any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# 主流程：每個特徵集都用同一搜尋空間選出最佳設定，再以最佳設定產生 OOF 預測與比較表。
def main() -> None:
    repo_root, split_dir, processed_dir = resolve_split_dirs("src/run_linear_ablation.py")
    main_dir = repo_root / "results" / "linear" / "main"
    selected = json.loads((main_dir / "selected_model.json").read_text(encoding="utf-8"))
    family = LINEAR_VARIANTS[selected["variant"]]
    output_dir = repo_root / "results" / "linear" / "ablation"

    processed = load_cv_folds(split_dir, processed_dir)
    set_a = restrict_folds(processed, drop_features(transformers=["engineered"]))
    set_c = build_set_c_folds(split_dir, processed)
    k = set_c[0].X_train.shape[1]
    set_b = {
        f"set_b_top{k}": build_set_b_folds(set_a, k),
        f"set_b_top{k}_log_ranked": build_set_b_folds(set_a, k, log_target=True),
    }
    feature_sets = {
        "set_a": set_a,
        **set_b,
        "set_c": set_c,
        **{
            f"set_c_minus_{group}": restrict_folds(set_c, drop_features(transformers=[group]))
            for group in ENGINEERED_GROUPS
        },
        **{
            f"set_c_minus_{name}": restrict_folds(set_c, drop_features(sources=features))
            for name, features in HYPOTHESIS_FEATURES.items()
        },
        "set_c_plus_aggregates": build_set_c_folds(split_dir, processed, exclude_groups=()),
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
    # 配對差異：同一個 fold 上與 Set C 相減；p 值為 5 個 fold 的 paired t-test（Set C 自己為 NaN）。
    paired = (
        fold_scores.assign(delta=lambda d: d["rmsle"] - d["fold"].map(reference))
        .groupby("model")["delta"]
        .agg(delta_rmsle_vs_set_c_mean="mean", delta_rmsle_vs_set_c_std="std",
             folds_set_c_better=lambda s: int((s > 0).sum()),
             paired_t_p_vs_set_c=lambda s: stats.ttest_1samp(s, 0).pvalue if s.any() else np.nan)
    )
    summary = summary.merge(paired, left_on="model", right_index=True)
    summary.insert(2, "params", summary["model"].map(fold_scores.groupby("model")["params"].first()))
    summary.insert(3, "feature_count", summary["model"].map(fold_scores.groupby("model")["feature_count"].mean()))

    write_experiment_outputs(output_dir, specs, fold_scores, summary, oof)
    pd.concat(searches, ignore_index=True).to_csv(output_dir / "search_summary.csv", index=False, float_format="%.6f")
    pd.concat([set_b_selections(name, folds) for name, folds in set_b.items()], ignore_index=True).to_csv(
        output_dir / "set_b_selected_features.csv", index=False, float_format="%.6f"
    )
    labels = slice_labels(split_dir)
    slice_metrics(oof, labels, list(feature_sets)).to_csv(output_dir / "slice_metrics.csv", index=False, float_format="%.6f")
    main_oof = pd.read_csv(main_dir / "oof_predictions.csv")
    main_models = ["trivial_mean", "simple_ols_overall", "strong_random_forest", f"ours_{selected['variant']}"]
    slice_metrics(main_oof, labels, main_models).to_csv(
        output_dir / "slice_metrics_main_models.csv", index=False, float_format="%.6f"
    )

    print(f"Variant: {selected['variant']}; Set B k = {k}")
    print(format_summary(summary))
    print(summary[["model", "feature_count", "params", "delta_rmsle_vs_set_c_mean", "delta_rmsle_vs_set_c_std",
                   "folds_set_c_better", "paired_t_p_vs_set_c"]].round(4).to_string(index=False))
    print(f"Results: {output_dir.relative_to(repo_root)}")
    print("Held-out test accessed: False")


if __name__ == "__main__":
    main()
