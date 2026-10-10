"""
程式用途：提供主實驗與消融實驗共用的資料讀取、特徵選取、評估指標與 5-fold CV 流程。

主要執行流程：
1. 從環境變數取得 raw split 與 processed split 目錄，並限制於 repository 內。
2. 逐 fold 讀取 processed sparse matrix，並從 raw split CSV 取得以 original_index 對齊的 value_eur。
3. 依 feature_mapping.csv 的欄名選取特徵；各 fold 欄數不同，不能共用 column index。
4. 每個 fold 只用該 fold 的 training subset fit 新模型，輸出逐 fold 分數與 out-of-fold 預測。

本模組只使用 training portion 的 5-fold CV，不讀取 held-out test set。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from threadpoolctl import threadpool_limits


TARGET = "value_eur"
N_SPLITS = 5
RANDOM_SEED = 42
# 共用伺服器上限制 CPU 執行緒數，避免 BLAS／OpenMP 佔滿所有核心；可用 FC26_N_JOBS 調整。
N_JOBS = int(os.environ.get("FC26_N_JOBS", "16"))
ID_COLUMNS = ["original_index", "player_id", "short_name"]
METRIC_COLUMNS = ["rmsle", "rmsle_positive", "rmse_eur", "mae_eur", "medae_eur", "r2_eur"]
EUR_AMOUNT_METRICS = {"rmse_eur", "mae_eur", "medae_eur"}
# Decision threshold（2026-10-09 組內決定，理由見 doc/Decision_Threshold.md）：身價 > 0 的球員，
# 相對誤差 |預測 / 實際 − 1| ≤ 20% 視為不會改變使用者（球探判斷報價）的決策。
DECISION_THRESHOLD = 0.20


# 單一 fold 的 training／validation 資料；X 與 y 已依 original_index 對齊。
@dataclass(frozen=True)
class FoldData:
    fold: int
    X_train: sparse.csr_matrix
    y_train: np.ndarray
    X_valid: sparse.csr_matrix
    y_valid: np.ndarray
    valid_ids: pd.DataFrame
    feature_mapping: pd.DataFrame


# 一個實驗設定；build 每次呼叫都要回傳新的 unfitted estimator，避免 fold 之間共用狀態。
# dense=True 用於不擅長 sparse 輸入的模型（例如 RandomForest）。
@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[], BaseEstimator]
    log_target: bool = False
    select_features: Callable[[pd.DataFrame], np.ndarray] | None = None
    dense: bool = False


# 從環境變數取得 split 目錄，並限制必須位於 repository 內，避免意外外部 I/O。
def resolve_split_dirs(script_path: str) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    split_value = os.environ.get("FC26_SPLIT_DIR")
    processed_value = os.environ.get("FC26_PROCESSED_DIR")
    if not split_value or not processed_value:
        raise RuntimeError(
            "請設定 FC26_SPLIT_DIR 與 FC26_PROCESSED_DIR，例如：\n"
            "FC26_SPLIT_DIR=data/splits FC26_PROCESSED_DIR=data/processed_splits "
            f"python {script_path}"
        )

    def resolve_dir(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError("資料目錄必須位於 repository 內") from exc
        if not path.is_dir():
            raise FileNotFoundError(f"找不到資料目錄：{path}")
        return path

    return repo_root, resolve_dir(split_value), resolve_dir(processed_value)


# 逐 fold 讀取 processed matrix 與 raw target，並確認列順序、欄數與 fold 劃分皆正確。
def load_cv_folds(split_dir: Path, processed_dir: Path) -> list[FoldData]:
    folds = []
    for fold in range(1, N_SPLITS + 1):
        raw_fold_dir = split_dir / "train_5-fold_cv" / f"fold_{fold}"
        processed_fold_dir = processed_dir / "train_5-fold_cv" / f"fold_{fold}"
        matrices, frames = {}, {}
        for part in ("train", "validation"):
            matrix = sparse.load_npz(processed_fold_dir / f"{part}.npz").tocsr()
            indices = pd.read_csv(processed_fold_dir / f"{part}_indices.csv")["original_index"]
            frame = pd.read_csv(
                raw_fold_dir / f"{part}.csv", usecols=[*ID_COLUMNS, TARGET], low_memory=False
            )
            if not frame["original_index"].equals(indices) or matrix.shape[0] != len(frame):
                raise RuntimeError(f"Fold {fold} {part}：processed matrix 與 raw target 的列無法對齊")
            matrices[part], frames[part] = matrix, frame

        mapping = pd.read_csv(processed_fold_dir / "feature_mapping.csv")
        if not (matrices["train"].shape[1] == matrices["validation"].shape[1] == len(mapping)):
            raise RuntimeError(f"Fold {fold}：feature 數與 feature_mapping 不一致")
        if set(frames["train"]["original_index"]) & set(frames["validation"]["original_index"]):
            raise RuntimeError(f"Fold {fold}：training 與 validation 發生重疊")

        folds.append(
            FoldData(
                fold=fold,
                X_train=matrices["train"],
                y_train=frames["train"][TARGET].to_numpy(dtype=float),
                X_valid=matrices["validation"],
                y_valid=frames["validation"][TARGET].to_numpy(dtype=float),
                valid_ids=frames["validation"][ID_COLUMNS].reset_index(drop=True),
                feature_mapping=mapping,
            )
        )

    expected = pd.read_csv(split_dir / "train_indices.csv")["original_index"]
    validation_indices = pd.concat([data.valid_ids["original_index"] for data in folds])
    if validation_indices.duplicated().any() or set(validation_indices) != set(expected):
        raise RuntimeError("5 個 validation fold 沒有剛好劃分 training portion")
    return folds


# 只保留 value_eur > 0 的列（training 與 validation 皆然）；strong baseline 的超參數搜尋使用。
def positive_only(folds: Sequence[FoldData]) -> list[FoldData]:
    result = []
    for data in folds:
        train_mask, valid_mask = data.y_train > 0, data.y_valid > 0
        result.append(
            FoldData(
                fold=data.fold,
                X_train=data.X_train[train_mask],
                y_train=data.y_train[train_mask],
                X_valid=data.X_valid[valid_mask],
                y_valid=data.y_valid[valid_mask],
                valid_ids=data.valid_ids[valid_mask].reset_index(drop=True),
                feature_mapping=data.feature_mapping,
            )
        )
    return result


# 依 feature_mapping 的輸出欄名選取特徵；各 fold 欄數不同，因此不可直接寫死 column index。
def columns_named(*names: str) -> Callable[[pd.DataFrame], np.ndarray]:
    def select(mapping: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(names) - set(mapping["output_feature"]))
        if missing:
            raise KeyError(f"feature_mapping 中找不到欄位：{missing}")
        return mapping.loc[mapping["output_feature"].isin(names), "output_index"].to_numpy()

    return select


# 消融實驗用：依來源欄位或 transformer 移除特徵群組，回傳保留的欄位。
# 交互作用特徵（例如 overall_x_reputation）只要任一來源被移除就一併移除。
def drop_features(
    sources: Sequence[str] = (), transformers: Sequence[str] = ()
) -> Callable[[pd.DataFrame], np.ndarray]:
    source_set, transformer_set = set(sources), set(transformers)

    def select(mapping: pd.DataFrame) -> np.ndarray:
        parts = mapping["source_feature"].str.split(r"\s*\+\s*").apply(set)
        unknown = (source_set - set().union(*parts)) | (transformer_set - set(mapping["transformer"]))
        if unknown:
            raise KeyError(f"feature_mapping 中找不到：{sorted(unknown)}")
        drop = parts.apply(lambda names: bool(names & source_set)) | mapping["transformer"].isin(
            transformer_set
        )
        return mapping.loc[~drop, "output_index"].to_numpy()

    return select


# 依 spec 建立新的 estimator；log1p target 以 TransformedTargetRegressor 在 fold 內轉換與還原。
def build_estimator(spec: ModelSpec) -> BaseEstimator:
    estimator = spec.build()
    if spec.log_target:
        return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    return estimator


# 同時計算 log 尺度（RMSLE）與歐元尺度指標，供選模與報告對照。
# 身價為 0 的球員在 log1p 尺度上誤差極大，另外計算只含正身價球員的 RMSLE 以便分開觀察。
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    positive = y_true > 0
    return {
        "rmsle": float(root_mean_squared_log_error(y_true, y_pred)),
        "rmsle_positive": float(root_mean_squared_log_error(y_true[positive], y_pred[positive])),
        "rmse_eur": float(root_mean_squared_error(y_true, y_pred)),
        "mae_eur": float(mean_absolute_error(y_true, y_pred)),
        "medae_eur": float(median_absolute_error(y_true, y_pred)),
        "r2_eur": float(r2_score(y_true, y_pred)),
    }


# 每個 fold 都 fit 新的 estimator，只看該 fold 的 training subset。
# 預測截在 [0, 該 fold training 最高身價]：身價不可能為負，且 log target 的線性模型會指數外推
# （例如曾把 Lamine Yamal 預測成 €765M）；上限只使用 training 資訊。
def run_cv(spec: ModelSpec, folds: Sequence[FoldData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 所有實驗都經過這裡，因此在此限制 BLAS／OpenMP 執行緒；共用伺服器上不限制時，
    # 小型矩陣運算會因執行緒爭用慢上數百倍。
    with threadpool_limits(limits=N_JOBS):
        return _run_cv(spec, folds)


def _run_cv(spec: ModelSpec, folds: Sequence[FoldData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    for data in folds:
        X_train, X_valid = data.X_train, data.X_valid
        if spec.select_features is not None:
            columns = spec.select_features(data.feature_mapping)
            X_train, X_valid = X_train[:, columns], X_valid[:, columns]
        if spec.dense:
            X_train, X_valid = X_train.toarray(), X_valid.toarray()

        estimator = build_estimator(spec)
        start = time.perf_counter()
        estimator.fit(X_train, data.y_train)
        fit_seconds = time.perf_counter() - start
        y_pred = np.clip(estimator.predict(X_valid), 0, data.y_train.max())

        rows.append(
            {
                "model": spec.name,
                "target": "log1p" if spec.log_target else "raw",
                "fold": data.fold,
                "train_rows": X_train.shape[0],
                "validation_rows": X_valid.shape[0],
                "feature_count": X_train.shape[1],
                **compute_metrics(data.y_valid, y_pred),
                "fit_seconds": fit_seconds,
            }
        )
        predictions.append(
            pd.DataFrame(
                {"original_index": data.valid_ids["original_index"].to_numpy(), spec.name: y_pred}
            )
        )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


# 將逐 fold 分數整理成 mean 與 std（ddof=1），作為主實驗表格的來源。
def summarize_scores(fold_scores: pd.DataFrame) -> pd.DataFrame:
    columns = [*METRIC_COLUMNS, "fit_seconds"]
    summary = fold_scores.groupby(["model", "target"], sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


# 依序執行多個實驗設定；OOF 表中每位 training 球員恰好一列，預測來自沒看過該球員的 fold 模型。
def run_experiments(
    specs: Sequence[ModelSpec], folds: Sequence[FoldData]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("實驗名稱不可重複")

    oof = pd.concat(
        [data.valid_ids.assign(fold=data.fold, y_true=data.y_valid) for data in folds],
        ignore_index=True,
    )
    score_frames = []
    for spec in specs:
        fold_scores, predictions = run_cv(spec, folds)
        score_frames.append(fold_scores)
        oof = oof.merge(predictions, on="original_index", how="left", validate="one_to_one")
        print(
            f"[done] {spec.name}: RMSLE {fold_scores['rmsle'].mean():.3f}, "
            f"fit {fold_scores['fit_seconds'].sum():.0f}s",
            flush=True,
        )
    if oof[names].isna().any().any():
        raise RuntimeError("OOF 預測有缺漏")

    fold_scores = pd.concat(score_frames, ignore_index=True)
    oof = oof.sort_values("original_index").reset_index(drop=True)
    return fold_scores, summarize_scores(fold_scores), oof


# 輸出模型設定、逐 fold 分數、摘要與 OOF 預測；OOF 以 original_index 排序，供統計檢定與錯誤分析對回原始資料。
def write_experiment_outputs(
    output_dir: Path,
    specs: Sequence[ModelSpec],
    fold_scores: pd.DataFrame,
    summary: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": [spec.name for spec in specs],
            "target": ["log1p" if spec.log_target else "raw" for spec in specs],
            "estimator": [repr(spec.build()) for spec in specs],
            "dense_input": [spec.dense for spec in specs],
        }
    ).to_csv(output_dir / "model_specs.csv", index=False)
    fold_scores.to_csv(output_dir / "cv_fold_scores.csv", index=False, float_format="%.6f")
    summary.to_csv(output_dir / "cv_summary.csv", index=False, float_format="%.6f")
    oof.to_csv(output_dir / "oof_predictions.csv", index=False, float_format="%.2f")


# 終端顯示用：比例型指標保留三位小數，歐元金額以千歐元顯示。
def format_summary(summary: pd.DataFrame) -> str:
    table = summary[["model", "target"]].copy()
    for metric in METRIC_COLUMNS:
        means, stds = summary[f"{metric}_mean"], summary[f"{metric}_std"]
        if metric in EUR_AMOUNT_METRICS:
            table[metric] = [f"{m / 1e3:,.0f}k ± {s / 1e3:,.0f}k" for m, s in zip(means, stds)]
        else:
            table[metric] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(means, stds)]
    return table.to_string(index=False)
