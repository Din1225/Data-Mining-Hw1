"""
程式用途：在 Set C 上比較 OLS、Ridge、Lasso、ElasticNet 四種線性迴歸，並與三層 baseline 一起產生主實驗表。

主要執行流程：
1. 建立 Set A（前處理後全部欄位）與 Set C（精簡 + 工程特徵，見 linear_features.py）的 5-fold 資料。
2. 在 Set C 上以 log1p(value_eur) 為 target，用固定 5-fold CV 搜尋各線性模型的正則化強度
   （搜尋範圍定義於 model_registry.LINEAR_VARIANTS），輸出完整調參曲線。
3. 以完整 training portion 計算 Set C 各特徵的 VIF，作為選擇線性變體的資料依據。
4. 主實驗表：
   - trivial：training-set mean（raw target）。
   - simple：只用 overall 的 OLS（log1p target）。
   - strong：RandomForest（Set A，使用 run_tuning.py 搜尋到的 log1p 超參數）。
   - ours：Set C 上四種線性模型各自的最佳設定。
5. 選模規則：n（約 1.2 萬）遠大於 p，除非有正則化的變體比 OLS 好超過一個標準誤（fold std / √5），
   否則採用最簡單、可直接做 t 檢定的 OLS。選定設定寫入 selected_model.json，供消融與最終 test 使用。
6. 輸出至 results/linear/main/。

所有模型共用同一組 folds 與指標；超參數在同一組 CV 上選擇，分數略為樂觀；held-out test 不參與。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from experiment_utils import (
    N_SPLITS,
    ModelSpec,
    columns_named,
    format_summary,
    load_cv_folds,
    resolve_split_dirs,
    run_experiments,
    summarize_scores,
    write_experiment_outputs,
)
from linear_features import FEATURE_GROUPS, RAW_COLUMNS, CompactFeatureBuilder, SetCTransformer, build_set_c_folds
from model_registry import LINEAR_VARIANTS, STRONG_BASELINE, STRONG_BASELINE_NAME, load_best_params
from run_tuning import search_configs


# VIF_j = 1 / (1 − R²_j)，等於相關係數矩陣反矩陣的對角線；用完整 training portion 的最終 Set C 特徵計算。
def compute_vif(split_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(split_dir / "train.csv", usecols=RAW_COLUMNS, low_memory=False)
    features = CompactFeatureBuilder().fit_transform(train)[SetCTransformer().columns]
    vif = np.diag(np.linalg.inv(np.corrcoef(features.to_numpy().T)))
    return pd.DataFrame(
        {
            "feature": features.columns,
            "group": [FEATURE_GROUPS[column] for column in features.columns],
            "vif": vif,
        }
    ).sort_values("vif", ascending=False)


# strong baseline：在 Set A 上用 run_tuning.py 搜尋到的 log1p 超參數，單一模型直接預測全部球員。
def strong_baseline_spec(tuning_dir: Path) -> ModelSpec:
    params = load_best_params(tuning_dir, f"{STRONG_BASELINE_NAME}_log1p")
    return ModelSpec(
        f"strong_{STRONG_BASELINE_NAME}",
        partial(STRONG_BASELINE.build, params),
        log_target=True,
        dense=STRONG_BASELINE.dense,
    )


# 主流程：調參 → 主實驗表 → VIF 與選定設定。
def main() -> None:
    repo_root, split_dir, processed_dir = resolve_split_dirs("src/run_linear_models.py")
    output_dir = repo_root / "results" / "linear" / "main"
    set_a = load_cv_folds(split_dir, processed_dir)
    set_c = build_set_c_folds(split_dir, set_a)

    curves, variant_rows, linear_specs = [], [], []
    for name, family in LINEAR_VARIANTS.items():
        _, summary = search_configs(f"{name}_set_c", family, "log1p", set_c)
        curves.append(summary.assign(variant=name))
        params = json.loads(summary.loc[0, "params"])
        linear_specs.append(ModelSpec(f"ours_{name}", partial(family.build, params), log_target=True))
        variant_rows.append(
            {
                "variant": name,
                "params": params,
                "cv_rmsle_mean": float(summary.loc[0, "rmsle_mean"]),
                "cv_rmsle_std": float(summary.loc[0, "rmsle_std"]),
                "n_configs": len(summary),
            }
        )

    tuning_dir = repo_root / "results" / "tuning"
    baseline_specs = [
        ModelSpec("trivial_mean", partial(DummyRegressor, strategy="mean")),
        ModelSpec(
            "simple_ols_overall",
            LinearRegression,
            log_target=True,
            select_features=columns_named("numeric__overall"),
        ),
        strong_baseline_spec(tuning_dir),
    ]
    a_scores, _, a_oof = run_experiments(baseline_specs, set_a)
    c_scores, _, c_oof = run_experiments(linear_specs, set_c)

    fold_scores = pd.concat(
        [a_scores.assign(feature_set="A"), c_scores.assign(feature_set="C")], ignore_index=True
    )
    summary = summarize_scores(fold_scores.drop(columns="feature_set"))
    summary.insert(2, "feature_set", summary["model"].map(fold_scores.groupby("model")["feature_set"].first()))
    oof = a_oof.merge(
        c_oof[["original_index", *[spec.name for spec in linear_specs]]],
        on="original_index",
        validate="one_to_one",
    )

    write_experiment_outputs(output_dir, [*baseline_specs, *linear_specs], fold_scores, summary, oof)
    pd.concat(curves, ignore_index=True).to_csv(output_dir / "tuning_curves.csv", index=False, float_format="%.6f")
    vif = compute_vif(split_dir)
    vif.to_csv(output_dir / "vif.csv", index=False, float_format="%.3f")
    pd.DataFrame(variant_rows).to_csv(output_dir / "linear_variants.csv", index=False, float_format="%.6f")
    best = min(variant_rows, key=lambda row: row["cv_rmsle_mean"])
    ols = next(row for row in variant_rows if row["variant"] == "ols")
    one_standard_error = best["cv_rmsle_std"] / np.sqrt(N_SPLITS)
    chosen = ols if ols["cv_rmsle_mean"] <= best["cv_rmsle_mean"] + one_standard_error else best
    selected = {
        **chosen,
        "selection_rule": "OLS unless a penalized variant beats it by more than one standard error of CV RMSLE",
        "best_variant_by_cv": best["variant"],
        "one_standard_error": float(one_standard_error),
    }
    (output_dir / "selected_model.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    print(format_summary(summary))
    print(pd.DataFrame(variant_rows).to_string(index=False))
    print(f"VIF > 10: {vif.loc[vif['vif'] > 10, ['feature', 'vif']].round(1).to_dict('records')}")
    print(f"Selected: {selected}")
    print(f"Results: {output_dir.relative_to(repo_root)}")
    print("Held-out test accessed: False")


if __name__ == "__main__":
    main()
