"""
程式用途：所有設計決定固定後，以完整 training portion 重新訓練最終模型與所有 baseline，在 held-out test 上評估唯一一次。

主要執行流程：
1. 讀取 results/linear/main/selected_model.json（最終線性變體與超參數），以及 run_tuning.py 搜尋到的
   strong baseline 超參數。
2. 以完整 training portion fit：
   - trivial：training-set mean；simple：只用 overall 的 OLS；strong：RandomForest（Set A）。
   - 最終模型：Set C 上的選定線性模型。
   Set A 使用 data/processed_splits/ 根目錄矩陣（前處理只在 training fit）；Set C 的建構與標準化也只在 training fit。
3. 在 held-out test 上計算與 CV 相同的指標，輸出逐筆預測供統計檢定與錯誤分析使用。
4. 以完整 training portion 估計最終模型係數（§8.1）：標準化係數 β、classical OLS 標準誤、t 檢定 p 值、
   95% CI、VIF，以及每增加 1 SD 對身價的近似百分比影響 exp(β) − 1（target 為 log1p 身價）。
5. 為確保 test 只使用一次：results/final_test/test_scores.csv 已存在時拒絕執行。

輸出至 results/final_test/。
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from experiment_utils import (
    ID_COLUMNS,
    N_JOBS,
    TARGET,
    ModelSpec,
    build_estimator,
    compute_metrics,
    resolve_split_dirs,
)
from linear_features import FEATURE_GROUPS, RAW_COLUMNS, SetCTransformer
from model_registry import LINEAR_VARIANTS, STRONG_BASELINE, STRONG_BASELINE_NAME, load_best_params
from run_linear_models import compute_vif


# 讀取根目錄的 Set A 矩陣與對齊的 raw 資料；確認列順序與 indices 一致。
def load_full_split(split_dir: Path, processed_dir: Path, part: str) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    matrix = sparse.load_npz(processed_dir / f"{part}.npz").tocsr()
    indices = pd.read_csv(processed_dir / f"{part}_indices.csv")["original_index"]
    raw = pd.read_csv(split_dir / f"{part}.csv", usecols=RAW_COLUMNS, low_memory=False)
    if not raw["original_index"].equals(indices) or matrix.shape[0] != len(raw):
        raise RuntimeError(f"{part}：processed matrix 與 raw 資料的列無法對齊")
    return matrix, raw


# 最終模型係數的 classical OLS 推論（標準化特徵、log1p target）。
# pct_change_per_unit 換回原始單位：overall 每 +1 分、或二元特徵 0 → 1 時身價的近似百分比變化。
def ols_inference(
    X: np.ndarray, y: np.ndarray, names: list[str], feature_sd: np.ndarray, vif: pd.DataFrame
) -> pd.DataFrame:
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    dof = len(y) - design.shape[1]
    sigma2 = residuals @ residuals / dof
    standard_error = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
    t_value = beta / standard_error
    critical = stats.t.ppf(0.975, dof)
    table = pd.DataFrame(
        {
            "feature": ["(intercept)", *names],
            "group": ["intercept", *[FEATURE_GROUPS[name] for name in names]],
            "beta_per_sd": beta,
            "std_error": standard_error,
            "t_value": t_value,
            "p_value": 2 * stats.t.sf(np.abs(t_value), dof),
            "ci95_low": beta - critical * standard_error,
            "ci95_high": beta + critical * standard_error,
            "pct_change_per_sd": (np.exp(beta) - 1) * 100,
            "feature_sd": np.r_[np.nan, feature_sd],
            "pct_change_per_unit": (np.exp(beta / np.r_[np.nan, feature_sd]) - 1) * 100,
        }
    )
    table.loc[0, "pct_change_per_sd"] = np.nan
    return table.merge(vif[["feature", "vif"]], on="feature", how="left")


# 主流程：檢查 test 尚未使用 → 訓練 → 唯一一次 test 評估 → 係數表。
def main() -> None:
    repo_root, split_dir, processed_dir = resolve_split_dirs("src/run_final_test.py")
    output_dir = repo_root / "results" / "final_test"
    if (output_dir / "test_scores.csv").exists():
        raise RuntimeError(
            f"held-out test 已評估過（{output_dir / 'test_scores.csv'}）；"
            "作業規定 test 只能使用一次，若確定要重跑請手動刪除該檔案"
        )

    selected = json.loads((repo_root / "results" / "linear" / "main" / "selected_model.json").read_text(encoding="utf-8"))
    tuning_dir = repo_root / "results" / "tuning"
    X_train_a, train = load_full_split(split_dir, processed_dir, "train")
    X_test_a, test = load_full_split(split_dir, processed_dir, "test")
    mapping_a = pd.read_csv(processed_dir / "feature_mapping.csv")
    y_train, y_test = train[TARGET].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)

    set_c = SetCTransformer()
    X_train_c, X_test_c = set_c.fit_transform(train), set_c.transform(test)
    overall_column = mapping_a.loc[mapping_a["output_feature"] == "numeric__overall", "output_index"].to_numpy()

    strong_params = load_best_params(tuning_dir, f"{STRONG_BASELINE_NAME}_log1p")
    final_family = LINEAR_VARIANTS[selected["variant"]]
    models = [
        ("trivial_mean", "A", ModelSpec("trivial_mean", partial(DummyRegressor, strategy="mean")), X_train_a, X_test_a),
        ("simple_ols_overall", "A", ModelSpec("simple_ols_overall", LinearRegression, log_target=True),
         X_train_a[:, overall_column], X_test_a[:, overall_column]),
        (f"strong_{STRONG_BASELINE_NAME}", "A",
         ModelSpec(f"strong_{STRONG_BASELINE_NAME}", partial(STRONG_BASELINE.build, strong_params),
                   log_target=True, dense=STRONG_BASELINE.dense),
         X_train_a, X_test_a),
        (f"ours_{selected['variant']}", "C",
         ModelSpec(f"ours_{selected['variant']}", partial(final_family.build, selected["params"]), log_target=True),
         X_train_c, X_test_c),
    ]

    scores, predictions = [], test[ID_COLUMNS].copy()
    predictions["y_true"] = y_test
    with threadpool_limits(limits=N_JOBS):
        for name, feature_set, spec, X_fit, X_eval in models:
            if spec.dense:
                X_fit, X_eval = X_fit.toarray(), X_eval.toarray()
            estimator = build_estimator(spec)
            start = time.perf_counter()
            estimator.fit(X_fit, y_train)
            fit_seconds = time.perf_counter() - start
            y_pred = np.clip(estimator.predict(X_eval), 0, y_train.max())
            predictions[name] = y_pred
            scores.append({"model": name, "feature_set": feature_set, **compute_metrics(y_test, y_pred), "fit_seconds": fit_seconds})
            print(f"[test] {name}: RMSLE {scores[-1]['rmsle']:.4f}", flush=True)

        vif = compute_vif(split_dir)
        coefficients = ols_inference(X_train_c, np.log1p(y_train), set_c.columns, set_c.scaler_.scale_, vif)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"model": name, "feature_set": fs, "estimator": repr(spec.build()), "log_target": spec.log_target}
         for name, fs, spec, _, _ in models]
    ).to_csv(output_dir / "model_specs.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, float_format="%.2f")
    coefficients.to_csv(output_dir / "coefficients.csv", index=False, float_format="%.6g")
    (output_dir / "final_model.json").write_text(
        json.dumps(
            {**selected, "features": set_c.columns, "train_rows": len(train), "test_rows": len(test)},
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    test_scores = pd.DataFrame(scores)
    test_scores.to_csv(output_dir / "test_scores.csv", index=False, float_format="%.6f")

    with pd.option_context("display.width", 200):
        print(test_scores.round(4).to_string(index=False))
        print(coefficients.round(4).to_string(index=False))
    print(f"Results: {output_dir.relative_to(repo_root)}")
    print("Held-out test accessed: True (single final evaluation)")


if __name__ == "__main__":
    main()
