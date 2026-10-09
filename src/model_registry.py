"""
程式用途：集中定義實驗用的模型、固定參數與超參數搜尋空間。

- STRONG_BASELINE：作業 §6 要求的 strong baseline（RandomForest），超參數由 run_tuning.py 搜尋。
- LINEAR_VARIANTS：最終模型的四種線性變體（作業規定最終預測器必須是線性迴歸），由 run_linear_models.py 比較。

run_tuning.py、run_linear_models.py 與 run_final_test.py 都由此重建模型，確保設定一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import ParameterGrid, ParameterSampler

from experiment_utils import N_JOBS, RANDOM_SEED


# 一類模型：fixed_params 不參與搜尋；n_iter 為 None 時跑完整 grid，否則以固定 seed 抽樣不重複的設定。
@dataclass(frozen=True)
class ModelFamily:
    estimator: type[BaseEstimator]
    fixed_params: dict[str, Any]
    search_space: dict[str, list[Any]]
    n_iter: int | None = None
    dense: bool = False
    targets: tuple[str, ...] = ("log1p",)

    def build(self, params: dict[str, Any]) -> BaseEstimator:
        return self.estimator(**self.fixed_params, **params)

    def configs(self) -> list[dict[str, Any]]:
        if self.n_iter is None:
            return list(ParameterGrid(self.search_space))
        return list(ParameterSampler(self.search_space, n_iter=self.n_iter, random_state=RANDOM_SEED))


STRONG_BASELINE_NAME = "random_forest"
STRONG_BASELINE = ModelFamily(
    RandomForestRegressor,
    fixed_params={"n_estimators": 300, "n_jobs": N_JOBS, "random_state": RANDOM_SEED},
    search_space={"max_features": [0.1, 0.2, 0.33, 0.5, 1.0], "min_samples_leaf": [1, 2, 4]},
    dense=True,
)


def _log_grid(low: int, high: int, count: int) -> list[float]:
    return [float(f"{value:.3g}") for value in np.logspace(low, high, count)]


# Ridge 的 alpha 乘在未除以 n 的平方誤差上；Lasso／ElasticNet 的 alpha 乘在除以 2n 的平方誤差上，
# 因此兩者的搜尋範圍不同。
LINEAR_VARIANTS = {
    "ols": ModelFamily(LinearRegression, fixed_params={}, search_space={}),
    "ridge": ModelFamily(Ridge, fixed_params={}, search_space={"alpha": _log_grid(-3, 4, 15)}),
    "lasso": ModelFamily(
        Lasso, fixed_params={"max_iter": 20000}, search_space={"alpha": _log_grid(-6, -1, 11)}
    ),
    "elastic_net": ModelFamily(
        ElasticNet,
        fixed_params={"max_iter": 20000},
        search_space={"alpha": _log_grid(-6, -1, 11), "l1_ratio": [0.2, 0.5, 0.8]},
    ),
}


# 讀取一項搜尋的最佳設定；缺檔時提示先執行 run_tuning.py。
def load_best_params(tuning_dir: Path, name: str) -> dict[str, Any]:
    path = tuning_dir / name / "best_params.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}；請先執行 src/run_tuning.py")
    return json.loads(path.read_text(encoding="utf-8"))["params"]
