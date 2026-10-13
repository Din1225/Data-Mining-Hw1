"""第三部分正式分析的共用檢查、來源紀錄與誤差圖。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels

BLUE = "#295F8A"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})


def configure_chinese_font() -> str:
    """選擇可用中文字型，避免繁體中文圖表變成缺字方框。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ['PingFang TC','Noto Sans CJK TC','Noto Sans TC','Microsoft JhengHei','Heiti TC','Arial Unicode MS']:
        if name in available:
            rcParams['font.family'] = [name,'DejaVu Sans']
            rcParams['axes.unicode_minus'] = False
            return name
    raise RuntimeError('找不到中文字型。請安裝思源黑體、Noto Sans CJK TC 或其他支援繁體中文的字型，再重新產生圖表。')


configure_chinese_font()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)



def output_directory(path: str | Path) -> Path:
    path = Path(path)
    require(not path.exists() or not any(path.iterdir()),
            f"輸出資料夾必須是新資料夾或空資料夾；既有檔案保留： {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path



def provenance(output: Path, inputs: list[str | Path], settings: dict) -> None:
    files = []
    repository_root = Path(__file__).resolve().parents[1]
    for item in inputs:
        if item is not None:
            path = Path(item).resolve()
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            recorded_path = str(path.relative_to(repository_root)) if path.is_relative_to(repository_root) else str(path)
            files.append({"path": recorded_path, "sha256": digest.hexdigest()})
    payload = {"path_base": "repository_root", "inputs": files, "settings": settings, "versions": {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__}}
    (output / "provenance.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")



def design_diagnostics(matrix: np.ndarray) -> tuple:
    means, scales = matrix.mean(axis=0), matrix.std(axis=0)
    constants = scales <= 1e-12
    safe_scale = np.where(constants, 1, scales)
    z = (matrix - means) / safe_scale
    design = np.column_stack([np.ones(len(z)), z])
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = singular_values[0] * max(design.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    full_rank = rank == design.shape[1]
    condition = singular_values[0] / singular_values[-1] if full_rank else np.inf
    vif = np.full(matrix.shape[1], np.inf)
    if full_rank:
        correlation = z.T @ z / len(z)
        vif = np.maximum(1, np.diag(np.linalg.inv(correlation)))
    return means, scales, constants, rank, condition, vif



def residual_chart(selected: pd.DataFrame, output: Path) -> None:
    """繪製正文使用的 actual − predicted 誤差圖；負值為高估。"""
    y = selected.y_true / 1e6
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(y, (selected.y_true - selected.y_pred) / 1e6, color=BLUE, s=16, alpha=0.55)
    ax.axhline(0, linestyle="--", color="#555555")
    ax.set(xlabel="實際身價（百萬 EUR）", ylabel="actual − predicted（百萬 EUR）",
           title="預測誤差：負值為高估")
    fig.tight_layout(); fig.savefig(output / "residual_vs_actual.png"); plt.close(fig)
