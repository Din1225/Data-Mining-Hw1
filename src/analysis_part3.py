"""第三部分：係數推論、對齊預測比較與錯誤分析。

程式只讀取明確提供的資料；示範僅使用虛構資料。
使用 ``python src/analysis_part3.py --help`` 查看三種輸入流程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse, stats
import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
import statsmodels
import statsmodels.api as sm
from part3_chinese import configure_chinese_font


PREDICTION_COLUMNS = [
    "original_index", "player_id", "model", "variant", "split", "fold", "seed",
    "y_true", "y_pred",
]
ID_COLUMNS = ["original_index", "player_id"]
BLUE, GOLD = "#295F8A", "#B97819"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})
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


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ID_COLUMNS:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="raise")
            require(np.isfinite(values).all() and (values >= 0).all()
                    and (values == np.floor(values)).all(), f"整數識別碼無效： {column}")
            frame[column] = values.astype("int64")
    return frame


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


def bootstrap_mean_ci(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple:
    require(repeats >= 200, "重抽樣至少200次；報告建議2,000次以上。")
    estimates = np.empty(repeats)
    for start in range(0, repeats, 64):
        stop = min(start + 64, repeats)
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def metric_row(frame: pd.DataFrame) -> dict:
    error = frame.y_pred.to_numpy() - frame.y_true.to_numpy()
    denominator = np.square(frame.y_true - frame.y_true.mean()).sum()
    return {"n": len(frame), "mae_eur": np.abs(error).mean(),
            "rmse_eur": np.sqrt(np.square(error).mean()),
            "r2": 1 - np.square(error).sum() / denominator if denominator > 0 else np.nan,
            "mean_residual_eur": error.mean()}


def validate_predictions(path: Path, split: str, indices: Path | None) -> pd.DataFrame:
    data = normalize_ids(read_csv(path))
    require(set(PREDICTION_COLUMNS) <= set(data),
            f"缺少預測欄位： {sorted(set(PREDICTION_COLUMNS) - set(data))}")
    require(data.split.isin(["test", "oof"]).all(), "split必須為test或oof。")
    for column in ["model", "variant", "seed"]:
        require(data[column].str.len().gt(0).all(), f"{column}不可為空。")
        require(not data[column].str.contains("|", regex=False).any(),
                f"模型識別鍵分隔符號|為保留字元： {column}")
    for column in ["y_true", "y_pred"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        require(np.isfinite(data[column]).all(), f"{column}含非有限數值。")
    require(data.y_true.ge(0).all(), "FC26的y_true必須是非負歐元身價。")
    data = data.loc[data.split.eq(split)].copy()
    require(len(data) > 0, f"找不到指定切分的預測：{split}.")
    if split == "test":
        require(data.fold.eq("").all(), "測試列的fold欄必須留空。")
    else:
        folds = pd.to_numeric(data.fold, errors="raise")
        require(folds.isin([1, 2, 3, 4, 5]).all(), "折外預測的折次必須是1至5的整數。")
        data["fold"] = folds.astype(int)
    data["model_key"] = data.model + "|" + data.variant + "|" + data.seed
    require(not data.duplicated(["model_key", "original_index"]).any(),
            "模型／特徵版本／隨機種子／樣本重複；不同種子不可當成新增球員樣本。")
    identities = data.groupby("original_index").player_id.nunique()
    require(identities.eq(1).all(), "original_index對應到不一致的球員識別碼。")
    require(data.groupby("player_id").original_index.nunique().eq(1).all(),
            "同一player_id對應多個原始索引。")
    expected = None
    if indices is not None:
        table = normalize_ids(read_csv(indices))
        require("original_index" in table and not table.original_index.duplicated().any(),
                "索引必須包含不重複的original_index。")
        expected = set(table.original_index)
        if "player_id" in table:
            identity = data[ID_COLUMNS].drop_duplicates().merge(
                table[ID_COLUMNS], on="original_index", suffixes=("_pred", "_expected"))
            require(identity.player_id_pred.eq(identity.player_id_expected).all(),
                    "球員識別碼與固定切分索引不一致。")
    require(indices is not None, "請以--indices提供完整的固定訓練或測試切分索引。")
    for key, part in data.groupby("model_key"):
        require(set(part.original_index) == expected, f"切分樣本不完整或成員不正確： {key}")
        if split == "oof":
            require(set(part.fold) == {1, 2, 3, 4, 5}, f"折外預測必須包含全部五折： {key}")
    # 檢查所有模型的實際值與折外折次是否對齊。
    reference = data[data.model_key.eq(data.model_key.iloc[0])].set_index(ID_COLUMNS).sort_index()
    for key, part in data.groupby("model_key"):
        part = part.set_index(ID_COLUMNS).sort_index()
        require(part.index.equals(reference.index), f"球員識別碼不一致： {key}")
        require(np.allclose(part.y_true, reference.y_true, rtol=0, atol=1e-6),
                f"不同模型的實際值不一致： {key}")
        require(part.fold.equals(reference.fold), f"折外折次指派不一致： {key}")
    return data


def compare_predictions(data: pd.DataFrame, plan: pd.DataFrame, split: str,
                        repeats: int, seed: int) -> pd.DataFrame:
    require({"model_a", "model_b"} <= set(plan), "比較CSV必須包含model_a與model_b。")
    require(len(plan) > 0 and not plan.duplicated(["model_a", "model_b"]).any(),
            "比較清單不可為空或重複。")
    unordered = [tuple(sorted(pair)) for pair in plan[["model_a", "model_b"]].itertuples(index=False, name=None)]
    require(len(set(unordered)) == len(unordered), "比較清單包含重複的反向比較。")
    rows, rng = [], np.random.default_rng(seed)
    for a, b in plan[["model_a", "model_b"]].itertuples(index=False, name=None):
        require(a != b and {a, b} <= set(data.model_key), f"未知或相同的模型識別鍵： {a}, {b}")
        left = data[data.model_key.eq(a)].set_index(ID_COLUMNS).sort_index()
        right = data[data.model_key.eq(b)].set_index(ID_COLUMNS).sort_index()
        ea = np.abs(left.y_pred - left.y_true).to_numpy()
        eb = np.abs(right.y_pred - right.y_true).to_numpy()
        d = ea - eb
        row = {"model_a": a, "model_b": b, "n": len(d), "mae_a_eur": ea.mean(),
               "mae_b_eur": eb.mean(), "mae_reduction_b_eur": d.mean(),
               "improvement_b_pct": 100 * d.mean() / ea.mean() if ea.mean() else np.nan,
               "p_paired_t": np.nan, "ci_low_t_eur": np.nan, "ci_high_t_eur": np.nan,
               "ci_low_bootstrap_eur": np.nan, "ci_high_bootstrap_eur": np.nan,
               "inference_status": "exploratory_oof_no_independence_assumption"}
        if split == "test":
            row["inference_status"] = "fixed_models_iid_player_approximation"
            if len(d) > 1 and np.std(d, ddof=1) > 0:
                result = stats.ttest_rel(ea, eb, alternative="two-sided")
                ci = result.confidence_interval()
                row.update(p_paired_t=result.pvalue, ci_low_t_eur=ci.low, ci_high_t_eur=ci.high)
                lo, hi = bootstrap_mean_ci(d, repeats, rng)
                row.update(ci_low_bootstrap_eur=lo, ci_high_bootstrap_eur=hi)
            else:
                row["inference_status"] = "degenerate_difference_no_valid_t_inference"
        rows.append(row)
    result = pd.DataFrame(rows)
    result["p_bonferroni"] = (result.p_paired_t * len(plan)).clip(upper=1)
    result["significant_after_correction"] = result.p_bonferroni.lt(0.05)
    return result


def attach_metadata(data: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None:
        return data
    context = normalize_ids(read_csv(path))
    require(set(ID_COLUMNS) <= set(context), "背景資料必須包含original_index與player_id。")
    require(not context.original_index.duplicated().any() and not context.player_id.duplicated().any(),
            "背景資料每位球員只能有一列。")
    keep = ID_COLUMNS + [c for c in context if c not in data.columns]
    merged = data.merge(context[keep], on=ID_COLUMNS, how="left", validate="many_to_one", indicator=True)
    require(merged._merge.eq("both").all(), "背景資料缺少預測樣本，或識別碼不一致。")
    return merged.drop(columns="_merge")


def prediction_charts(selected: pd.DataFrame, output: Path, demo: bool) -> None:
    label = "合成資料示範｜" if demo else ""
    y, pred = selected.y_true / 1e6, selected.y_pred / 1e6
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter(y, pred, s=16, alpha=0.55, color=BLUE, label=f"樣本數＝ {len(selected):,}")
    low, high = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([low, high], [low, high], "--", color="#555555", label="完全準確的預測")
    ax.set(xlabel="實際身價（百萬 EUR）", ylabel="預測身價（百萬 EUR）",
           title=f"{label}實際與預測身價 ({selected.split.iloc[0]})")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "actual_vs_predicted.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(y, (selected.y_true - selected.y_pred) / 1e6, color=BLUE, s=16, alpha=0.55)
    ax.axhline(0, linestyle="--", color="#555555")
    ax.set(xlabel="實際身價（百萬 EUR）", ylabel="actual − predicted（百萬 EUR）",
           title=f"{label}預測誤差：負值為高估")
    fig.tight_layout(); fig.savefig(output / "residual_vs_actual.png"); plt.close(fig)


def run_predictions(args: argparse.Namespace) -> Path:
    data = validate_predictions(Path(args.predictions), args.split, Path(args.indices))
    plan = read_csv(args.comparisons)
    comparisons = compare_predictions(data, plan, args.split, args.bootstrap, args.seed)
    data = attach_metadata(data, Path(args.metadata) if args.metadata else None)
    require(args.final_model in set(data.model_key), "--final-model必須對應已提供的模型識別鍵。")
    output = output_directory(args.output)
    comparisons.to_csv(output / "model_comparisons.csv", index=False)
    metrics = [{"model_key": key, "split": args.split, **metric_row(group)}
               for key, group in data.groupby("model_key")]
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    if args.split == "oof":
        folds = [{"model_key": key, "fold": fold, **metric_row(group)}
                 for (key, fold), group in data.groupby(["model_key", "fold"])]
        pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    selected = data[data.model_key.eq(args.final_model)].copy()
    selected["residual_eur"] = selected.y_pred - selected.y_true
    selected["absolute_error_eur"] = np.abs(selected.residual_eur)
    selected["error_actual_minus_predicted_eur"] = -selected.residual_eur
    selected["negative_prediction"] = selected.y_pred.lt(0)
    selected = selected.sort_values(["absolute_error_eur", "original_index"], ascending=[False, True])
    selected.to_csv(output / "all_residuals.csv", index=False)
    selected.head(args.top).to_csv(output / "largest_error_cases.csv", index=False)
    slices = selected.copy()
    slices["value_band_eur"] = pd.cut(slices.y_true, [-np.inf, 1e6, 1e7, 5e7, np.inf],
                                      labels=["<=1M", "1M–10M", "10M–50M", ">50M"])
    if "age" in slices:
        slices["age_band_years"] = pd.cut(pd.to_numeric(slices.age, errors="raise"),
                                          [-np.inf, 21, 29, np.inf], labels=["<=21", "22–29", ">=30"])
    slice_rows = []
    for column in ["value_band_eur", "age_band_years", "international_reputation", "player_positions", "league_id"]:
        if column in slices:
            for value, group in slices.groupby(column, observed=True):
                slice_rows.append({"slice_feature": column, "slice_value": str(value), **metric_row(group)})
    pd.DataFrame(slice_rows).to_csv(output / "slice_metrics.csv", index=False)
    prediction_charts(selected, output, args.demo)
    cases = ["# 錯誤案例工作表", "",
             "**合成資料示範：以下是虛構資料，不是作業實驗結果。**" if args.demo else "正式分析候選；原因仍需核對 training 資料。",
             "", "誤差定義：predicted − actual；正值為高估。另存 actual − predicted，與規格範例一致。",
             ""]
    for rank, (_, row) in enumerate(selected.head(3).iterrows(), start=1):
        cases += [f"## 案例 {rank}：player_id={row.player_id}",
                  f"- Actual：€{row.y_true:,.2f}；Predicted：€{row.y_pred:,.2f}。",
                  f"- Residual：€{row.residual_eur:,.2f}；Absolute error：€{row.absolute_error_eur:,.2f}。",
                  "- 輸入背景：[填入年齡、overall、聲望、位置、聯賽等已核對資料]。",
                  "- Training 證據：[相似球員數、該群體誤差或其他模型比較]。",
                  "- 可能原因與替代解釋：[待分析，勿自動歸因為 noise]。", ""]
    cases += ["## 共同模式與改善", "[先用 slice_metrics.csv 或殘差圖查證，再寫第一個想做的改善；不回頭用 test 調參。]", ""]
    (output / "case_workbook.md").write_text("\n".join(cases), encoding="utf-8")
    notes = {"demo": args.demo, "split": args.split, "final_model": args.final_model,
             "formal_p_values": args.split == "test", "unit": "EUR",
             "positive_delta": "MAE_A－MAE_B大於0，表示模型B的平均絕對誤差較低。",
             "ci": "95%信賴區間按各項比較分別計算，未進行多重比較校正。",
             "assumptions": "測試推論將球員視為獨立、模型視為固定；不涵蓋球會／聯賽相依性與選模不確定性。",
             "oof": "折外預測僅提供描述性結果；重疊訓練集不適合直接視為獨立樣本推論。",
             "negative_predictions": "保留並標記負預測，未自動裁切。",
             "case_selection": "依絕對誤差排序；相同誤差按原始索引排序。"}
    provenance(output, [args.predictions, args.comparisons, args.indices, args.metadata], notes)
    (output / "README.md").write_text(
        "# " + ("合成資料示範｜" if args.demo else "") + "預測分析\n\n"
        + f"Split: `{args.split}`. Model: `{args.final_model}`. Units: EUR.\n\n"
        + "假設請見`provenance.json`；平均絕對誤差減少為正值，表示模型B較佳。"
        + "Bonferroni校正使用所有預先列出的比較；95%信賴區間未校正。"
        + "折外預測不提供正式p值或區間。錯誤原因需要另行查證。\n",
        encoding="utf-8")
    return output


def read_design(matrix_path: Path, mapping_path: Path, max_cells: int) -> tuple:
    mapping = read_csv(mapping_path)
    require({"output_index", "output_feature"} <= set(mapping), "特徵對照必須包含output_index與output_feature。")
    mapping["output_index"] = pd.to_numeric(mapping.output_index, errors="raise")
    require(mapping.output_index.tolist() == list(range(len(mapping))), "特徵對照索引必須符合矩陣欄位順序。")
    require(mapping.output_feature.str.len().gt(0).all() and not mapping.output_feature.duplicated().any(),
            "特徵名稱不可為空或重複。")
    if matrix_path.suffix == ".npy":
        matrix = np.load(matrix_path, allow_pickle=False)
    else:
        matrix = sparse.load_npz(matrix_path)
    require(len(matrix.shape) == 2 and matrix.shape[1] == len(mapping), "矩陣欄數與特徵對照不一致。")
    require(matrix.shape[0] >= 2 and matrix.shape[1] >= 1, "設計矩陣必須包含樣本與至少一項特徵。")
    require(matrix.shape[0] * matrix.shape[1] <= max_cells,
            "密集係數分析超過--max-cells限制；請使用已核對的最終特徵集。")
    matrix = matrix.toarray() if sparse.issparse(matrix) else matrix
    matrix = np.asarray(matrix, dtype=float)
    require(np.isfinite(matrix).all(), "設計矩陣含非有限數值。")
    return matrix, mapping


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


def coefficient_chart(table: pd.DataFrame, output: Path, demo: bool) -> None:
    usable = table.dropna(subset=["beta_per_sd_eur", "ci_low_per_sd_eur", "ci_high_per_sd_eur"])
    chosen = usable.iloc[np.argsort(np.abs(usable.beta_per_sd_eur.to_numpy()))[-min(20, len(usable)):]]
    if len(chosen) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.34 * len(chosen) + 1.7)))
    y = np.arange(len(chosen))
    # 分別繪製區間線段，支援非對稱百分位數區間。
    ax.hlines(y, chosen.ci_low_per_sd_eur / 1e6, chosen.ci_high_per_sd_eur / 1e6, color=BLUE, linewidth=2)
    ax.scatter(chosen.beta_per_sd_eur / 1e6, y, color=GOLD, zorder=3)
    ax.axvline(0, color="#555555", linestyle="--")
    ax.set_yticks(y, chosen.feature)
    title = "合成資料示範｜" if demo else ""
    ax.set(title=title + "係數與95% CI", xlabel="每 1 training SD 的係數（百萬 EUR）")
    fig.tight_layout(); fig.savefig(output / "coefficient_forest.png"); plt.close(fig)


def run_coefficients(args: argparse.Namespace) -> Path:
    spec = json.loads(Path(args.spec).read_text())
    require(isinstance(spec, dict), "模型設定必須是JSON物件。")
    require(spec.get("kind") in ["ols", "ridge"], "支援OLS或Ridge；Lasso與ElasticNet需另行核對推論方法。")
    allowed = {"kind", "fit_intercept", "target_transform", "target_unit"}
    if spec["kind"] == "ridge":
        allowed.add("alpha")
    require(set(spec) <= allowed, f"不支援的擬合設定： {sorted(set(spec) - allowed)}")
    require(spec.get("target_transform") == "identity" and spec.get("target_unit") == "EUR",
            "此入口僅支援未轉換的歐元目標；對數目標係數不能直接解讀成歐元效果。")
    require(spec.get("fit_intercept") is True, "此流程要求由模型擬合截距，X不可另放截距欄。")
    matrix, mapping = read_design(Path(args.matrix), Path(args.mapping), args.max_cells)
    labels = normalize_ids(read_csv(args.labels))
    matrix_indices = normalize_ids(read_csv(args.matrix_indices))
    require(set(ID_COLUMNS + ["value_eur"]) <= set(labels), "訓練答案必須包含識別碼與value_eur。")
    require("original_index" in matrix_indices and not matrix_indices.original_index.duplicated().any(),
            "矩陣列索引不可重複。")
    require(len(labels) == len(matrix) == len(matrix_indices), "訓練列數不一致。")
    require(labels.original_index.tolist() == matrix_indices.original_index.tolist(),
            "訓練答案列必須完全符合matrix_indices順序；程式不會自動重新對齊。")
    if "player_id" in matrix_indices:
        require(labels.player_id.tolist() == matrix_indices.player_id.tolist(),
                "訓練球員識別碼與matrix_indices不一致。")
    require(not labels.original_index.duplicated().any() and not labels.player_id.duplicated().any(),
            "訓練答案識別碼不可重複。")
    y = pd.to_numeric(labels.value_eur, errors="raise").to_numpy(float)
    require(np.isfinite(y).all() and (y >= 0).all(), "訓練目標無效。")
    expected = read_csv(args.reference)
    require({"feature", "beta"} <= set(expected) and not expected.feature.duplicated().any(),
            "參考係數必須包含唯一的feature與beta，並含__intercept__。")
    reference = expected.set_index("feature").beta.apply(pd.to_numeric)
    names = mapping.output_feature.tolist()
    require(set(reference.index) == set(names + ["__intercept__"]), "參考特徵名稱不一致。")
    means, scales, constants, rank, condition, vif = design_diagnostics(matrix)
    output = output_directory(args.output)
    diagnostics = mapping.copy()
    diagnostics["mean_x"] = means; diagnostics["sd_x"] = scales
    diagnostics["constant"] = constants; diagnostics["vif"] = vif
    diagnostics.to_csv(output / "feature_diagnostics.csv", index=False)
    notes = {"demo": args.demo, "model_spec": spec, "rank_with_intercept": rank,
             "columns_with_intercept": matrix.shape[1] + 1, "condition_number": float(condition) if np.isfinite(condition) else "infinite",
             "target_unit": "EUR", "scale": "報告原始係數及其乘訓練標準差的值；不重新縮放X並擬合Ridge，以免改變懲罰。",
             "selection": "不確定性條件於固定特徵、前處理、alpha與訓練流程；不作因果或選模後推論。",
             "independence": "假設資料列獨立；不涵蓋球會或聯賽群聚。"}
    inputs = [args.matrix, args.mapping, args.labels, args.matrix_indices, args.spec, args.reference]
    if spec["kind"] == "ols":
        if constants.any() or rank != matrix.shape[1] + 1 or len(y) <= matrix.shape[1] + 1:
            notes["status"] = "blocked_unidentified_ols"
            provenance(output, inputs, notes)
            raise ValueError(f"OLS效果不可辨識或沒有殘差自由度。秩={rank}, p+1={matrix.shape[1]+1}。請見 {output}/feature_diagnostics.csv；不能未說明就移除欄位。")
        covariance = args.covariance
        result = sm.OLS(y, sm.add_constant(matrix, has_constant="add")).fit(cov_type=covariance, use_t=True)
        beta, intercept = result.params[1:], result.params[0]
        interval, pvalues = result.conf_int()[1:], result.pvalues[1:]
        se = result.bse[1:]
        method = "OLS傳統t檢定" if covariance == "nonrobust" else "OLS採HC3穩健共變異數與近似t推論"
    else:
        alpha = spec.get("alpha")
        require(isinstance(alpha, (int, float)) and np.isfinite(alpha) and alpha > 0,
                "Ridge需要由訓練交叉驗證選定的固定正alpha。")
        require(args.bootstrap >= 200, "重抽樣至少200次。")
        model = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(matrix, y)
        beta, intercept = model.coef_, model.intercept_
        # 對既定轉換後設計做訓練列重抽樣；不重做特徵選擇或調參。
        draws = np.empty((args.bootstrap, matrix.shape[1]))
        rng = np.random.default_rng(args.seed)
        for b in range(args.bootstrap):
            idx = rng.integers(len(y), size=len(y))
            draws[b] = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(matrix[idx], y[idx]).coef_
        np.save(output / "bootstrap_coefficients.npy", draws, allow_pickle=False)
        interval = np.quantile(draws, [0.025, 0.975], axis=0).T
        se = draws.std(axis=0, ddof=1)
        # 使用重抽樣標準誤的常態／Wald近似，並標示其非精確重抽樣檢定。
        pvalues = np.full(len(beta), np.nan)
        if rank == matrix.shape[1] + 1:
            valid = se > 0
            pvalues[valid] = 2 * stats.norm.sf(np.abs(beta[valid] / se[valid]))
        method = "Ridge訓練列重抽樣百分位數信賴區間；以重抽樣標準誤計算近似 Wald p值"
        notes.update(bootstrap=args.bootstrap, seed=args.seed,
                     p_value_caveat="固定alpha懲罰估計量的近似Wald p值；不是精確重抽樣p值或無懲罰母體效果。設計不滿秩時不提供p值；百分位數區間與Wald p值可能不一致。")
    actual = np.r_[intercept, beta]
    target = reference.loc[["__intercept__"] + names].to_numpy(float)
    require(np.allclose(actual, target, rtol=1e-6, atol=1e-5),
            "重現係數與組員提供的最終係數不一致；請核對矩陣、目標轉換與超參數。")
    table = pd.DataFrame({"feature": names, "beta_eur_per_x_unit": beta,
                          "ci_low_eur_per_x_unit": interval[:, 0], "ci_high_eur_per_x_unit": interval[:, 1],
                          "beta_per_sd_eur": beta * scales, "ci_low_per_sd_eur": interval[:, 0] * scales,
                          "ci_high_per_sd_eur": interval[:, 1] * scales, "se_eur_per_x_unit": se,
                          "p_value": pvalues, "vif": vif, "training_sd": scales, "method": method})
    table["p_bonferroni"] = (table.p_value * len(table)).clip(upper=1)
    table["significant_after_correction"] = table.p_bonferroni.lt(0.05)
    table.to_csv(output / "coefficients.csv", index=False)
    coefficient_chart(table, output, args.demo)
    notes.update(status="completed", reference_match=True, intercept_eur=float(intercept), method=method,
                 ci="95%逐項信賴區間，未進行多重比較校正。", all_features_in_csv=True,
                 plotted_features="圖顯示絕對標準化係數前20項；CSV保留全部特徵。")
    provenance(output, inputs, notes)
    (output / "README.md").write_text(
        "# " + ("合成資料示範｜" if args.demo else "") + "係數分析\n\n"
        + method + "。全部特徵存於coefficients.csv；圖最多顯示20項。\n\n"
        + "每一標準差係數是報告換算，不是更換Ridge懲罰後重新擬合。"
        + "獨熱編碼每一單位係數是0→1的條件對比，取決於參考組編碼。"
        + "平方與交互項須共同解讀；不作因果解釋。"
        + "選模、群聚與p值限制請見provenance.json。\n", encoding="utf-8")
    return output


def demo_inputs(output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_train, n_test = 240, 90
    x = rng.normal(size=(n_train + n_test, 3))
    x[:, 1] = 0.35 * x[:, 0] + 0.94 * x[:, 1]
    target = 10e6 + x @ np.array([1.2e6, -0.3e6, 0.8e6]) + rng.normal(0, 250000, len(x))
    target[n_train:n_train + 3] += [7e6, 5e6, 4e6]  # 虛構未觀測衝擊，明確標示為示範
    names = ["ability_z", "age_z", "reputation_z"]
    mapping = pd.DataFrame({"output_index": range(3), "output_feature": names,
                            "source_feature": names, "transformer": "synthetic"})
    mapping.to_csv(output / "feature_mapping.csv", index=False)
    sparse.save_npz(output / "train.npz", sparse.csr_matrix(x[:n_train]))
    labels = pd.DataFrame({"original_index": range(n_train), "player_id": np.arange(n_train) + 10000,
                           "value_eur": target[:n_train]})
    labels.to_csv(output / "training_labels.csv", index=False)
    labels[["original_index"]].to_csv(output / "train_indices.csv", index=False)
    pd.DataFrame({"original_index": range(n_train, len(x)), "player_id": np.arange(n_train, len(x)) + 10000}).to_csv(output / "test_indices.csv", index=False)
    meta = pd.DataFrame({"original_index": range(len(x)), "player_id": np.arange(len(x)) + 10000,
                         "short_name": [f"虛構球員 {i}" for i in range(len(x))],
                         "age": rng.integers(17, 39, len(x)), "international_reputation": rng.integers(1, 4, len(x)),
                         "synthetic_omitted_shock": [n_train <= i < n_train + 3 for i in range(len(x))]})
    meta.to_csv(output / "metadata.csv", index=False)
    models = {"OLS": LinearRegression(), "Ridge": Ridge(alpha=3, solver="svd")}
    for name, model in models.items():
        model.fit(x[:n_train], target[:n_train])
        pd.DataFrame({"feature": ["__intercept__"] + names, "beta": np.r_[model.intercept_, model.coef_]}).to_csv(output / f"{name.lower()}_reference.csv", index=False)
        spec = {"kind": name.lower(), "fit_intercept": True, "target_unit": "EUR", "target_transform": "identity"}
        if name == "Ridge": spec["alpha"] = 3
        (output / f"{name.lower()}_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    rows = []
    def append_rows(name, indices, predicted, split, fold):
        for idx, pred in zip(indices, predicted):
            rows.append({"original_index": idx, "player_id": idx + 10000, "model": name,
                         "variant": "full", "split": split, "fold": fold, "seed": seed,
                         "y_true": target[idx], "y_pred": pred})
    test_indices = np.arange(n_train, len(x))
    for name, model in models.items(): append_rows(name, test_indices, model.predict(x[n_train:]), "test", "")
    append_rows("Mean", test_indices, np.repeat(target[:n_train].mean(), n_test), "test", "")
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(x[:n_train]), 1):
        for name, model in {"OLS": LinearRegression(), "Ridge": Ridge(alpha=3, solver="svd")}.items():
            model.fit(x[tr], target[tr]); append_rows(name, va, model.predict(x[va]), "oof", fold)
        append_rows("Mean", va, np.repeat(target[tr].mean(), len(va)), "oof", fold)
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame({"model_a": [f"Mean|full|{seed}", f"OLS|full|{seed}"],
                  "model_b": [f"Ridge|full|{seed}"] * 2}).to_csv(output / "comparisons.csv", index=False)
    (output / "README.md").write_text("僅供合成資料示範。所有列與結果皆為虛構，未讀取FC26資料。\n")


def run_demo(args: argparse.Namespace) -> Path:
    root = output_directory(args.output)
    inputs = root / "inputs"; inputs.mkdir(); demo_inputs(inputs, args.seed)
    for split in ["test", "oof"]:
        run_predictions(argparse.Namespace(predictions=inputs / "predictions.csv", comparisons=inputs / "comparisons.csv",
            indices=inputs / ("test_indices.csv" if split == "test" else "train_indices.csv"),
            metadata=inputs / "metadata.csv", split=split, final_model=f"Ridge|full|{args.seed}",
            bootstrap=args.bootstrap, seed=args.seed, top=10, output=root / split, demo=True))
    for kind in ["ols", "ridge"]:
        run_coefficients(argparse.Namespace(matrix=inputs / "train.npz", mapping=inputs / "feature_mapping.csv",
            labels=inputs / "training_labels.csv", matrix_indices=inputs / "train_indices.csv",
            spec=inputs / f"{kind}_spec.json", reference=inputs / f"{kind}_reference.csv", max_cells=25_000_000,
            covariance="HC3", bootstrap=args.bootstrap, seed=args.seed, output=root / f"coefficients_{kind}", demo=True))
    (root / "README.md").write_text("# 合成資料示範：不是作業結果\n\n"
        "虛構資料用來驗證流程；未讀取FC26原始、訓練或測試資料。"
        "前三個虛構測試球員刻意加入未觀測衝擊。"
        "test/提供固定模型成對比較；oof/不提供正式推論。"
        "coefficients_ols/與coefficients_ridge/使用不同且明確標示的方法。\n", encoding="utf-8")
    return root


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    predictions = commands.add_parser("predictions", help="分析明確提供且已對齊的歐元預測。")
    for field in ["predictions", "comparisons", "indices", "output"]:
        predictions.add_argument("--" + field, required=True)
    predictions.add_argument("--metadata")
    predictions.add_argument("--split", choices=["test", "oof"], default="oof")
    predictions.add_argument("--final-model", required=True, help="model|variant|seed")
    predictions.add_argument("--top", type=int, default=10)
    coefficients = commands.add_parser("coefficients", help="以訓練資料推論既定OLS或Ridge係數。")
    for field in ["matrix", "mapping", "labels", "matrix-indices", "spec", "reference", "output"]:
        coefficients.add_argument("--" + field, required=True)
    coefficients.add_argument("--max-cells", type=int, default=25_000_000)
    coefficients.add_argument("--covariance", choices=["HC3", "nonrobust"], default="HC3")
    demo = commands.add_parser("demo", help="只用虛構資料驗證全部流程。")
    demo.add_argument("--output", required=True)
    for command in [predictions, coefficients, demo]:
        command.add_argument("--bootstrap", type=int, default=2000)
        command.add_argument("--seed", type=int, default=42)
    predictions.add_argument("--demo", action="store_true", help="將輸入標示為合成資料示範。")
    coefficients.add_argument("--demo", action="store_true", help="將輸入標示為合成資料示範。")
    return root


def main() -> None:
    args = parser().parse_args()
    require(args.bootstrap >= 200, "--bootstrap必須至少為200。")
    if args.command == "predictions":
        require(args.top >= 3, "至少需要三個錯誤案例。")
    handlers = {"predictions": run_predictions, "coefficients": run_coefficients, "demo": run_demo}
    try:
        output = handlers[args.command](args)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"分析已停止： {exc}") from exc
    print(f"輸出資料夾： {output.resolve()}")


if __name__ == "__main__":
    main()
