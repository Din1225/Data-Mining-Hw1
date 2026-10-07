"""
程式用途：針對 FC 26 原始球員資料執行可重現的資料品質檢查與基本統計。

主要執行流程：
1. 從 FC26_DATA_PATH 環境變數讀取 repository 內的原始 CSV。
2. 建立資料集概覽、缺失值、數值欄位與類別欄位統計。
3. 檢查重複資料、特殊缺失標記、日期／位置評分格式及保守的合理值域。
4. 輸出資料集概覽、缺失值、數值欄位與類別欄位四張 CSV。

本程式只做描述性稽核，不刪除資料、不切分資料、不做 feature selection，也不訓練模型。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd


TARGET = "value_eur"
RARE_CATEGORY_MAX_COUNT = 9
TOP_CATEGORY_COUNT = 5
SPECIAL_MISSING_TOKENS = {
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    "missing",
    "unknown",
    "-",
    "--",
}

DATE_COLUMNS = ["fifa_update_date", "dob", "club_joined_date"]
MULTI_VALUE_COLUMNS = ["player_positions", "player_tags", "player_traits"]
POSITION_RATING_COLUMNS = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram",
    "lm", "lcm", "cm", "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb",
    "lb", "lcb", "cb", "rcb", "rb", "gk",
]
SEMANTIC_CATEGORICAL_COLUMNS = [
    "player_id", "fifa_version", "fifa_update", "league_id", "league_level",
    "club_team_id", "club_jersey_number", "nationality_id", "nation_team_id",
    "nation_jersey_number", "weak_foot", "skill_moves", "international_reputation",
    "work_rate",
]
SUMMARY_RATING_COLUMNS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
DETAILED_RATING_COLUMNS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
    "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
    "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking_awareness",
    "defending_standing_tackle", "defending_sliding_tackle", "goalkeeping_diving",
    "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
    "goalkeeping_reflexes", "goalkeeping_speed",
]


# 從環境變數取得資料路徑，並限制輸入必須位於 repository 內，避免意外外部 I/O。
def resolve_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    input_value = os.environ.get("FC26_DATA_PATH")
    if not input_value:
        raise RuntimeError(
            "請設定 FC26_DATA_PATH，例如："
            "FC26_DATA_PATH=data/FC26_20250921.csv python src/data_audit.py"
        )

    input_path = Path(input_value)
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    input_path = input_path.resolve()

    try:
        input_path.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError("FC26_DATA_PATH 必須指向 repository 內的檔案") from exc

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到資料檔：{input_path}")

    result_dir = repo_root / "results" / "data_summary"
    result_dir.mkdir(parents=True, exist_ok=True)
    return input_path, result_dir


# 同時讀取推論型別版本與保留原始字串版本，才能區分一般缺值、空字串與特殊 token。
def load_data(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = pd.read_csv(input_path, low_memory=False)
    raw_strings = pd.read_csv(input_path, dtype=str, keep_default_na=False, low_memory=False)
    if dataframe.shape != raw_strings.shape:
        raise RuntimeError("兩種讀取方式得到不同資料形狀，無法安全進行稽核")
    if TARGET not in dataframe.columns:
        raise KeyError(f"資料缺少已確認的 target：{TARGET}")
    return dataframe, raw_strings


# 將特殊欄位的字串 representation 明確標註，避免後續把多選或評分字串當一般類別。
def representation_type(column: str) -> str:
    if column in MULTI_VALUE_COLUMNS:
        return "comma_separated_multi_value"
    if column in POSITION_RATING_COLUMNS:
        return "base_rating_plus_modifier_string"
    if column in DATE_COLUMNS:
        return "date_string_YYYY-MM-DD"
    if column in {"player_url", "player_face_url"}:
        return "url"
    if column in SEMANTIC_CATEGORICAL_COLUMNS:
        return "numeric_coded_categorical"
    return "single_categorical_value"


# 建立逐欄缺失表，並保留空字串和特殊 missing token 的原始檢查結果。
def build_missing_summary(
    dataframe: pd.DataFrame, raw_strings: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    total_rows = len(dataframe)

    for column in dataframe.columns:
        raw_trimmed = raw_strings[column].str.strip()
        empty_count = int(raw_trimmed.eq("").sum())
        normalized = raw_trimmed.str.lower()
        special_count = int(
            (normalized.isin(SPECIAL_MISSING_TOKENS) & raw_trimmed.ne("")).sum()
        )
        missing_count = int(dataframe[column].isna().sum())
        unique_count = int(dataframe[column].nunique(dropna=True))
        rows.append(
            {
                "feature_name": column,
                "dtype": str(dataframe[column].dtype),
                "non_missing_count": total_rows - missing_count,
                "missing_count": missing_count,
                "missing_ratio": missing_count / total_rows,
                "empty_string_count_raw": empty_count,
                "special_missing_token_count_raw": special_count,
                "unique_count": unique_count,
                "constant_non_null": unique_count <= 1,
            }
        )

    return pd.DataFrame(rows)


# 對所有由 pandas 判定為數值的欄位產生完整描述統計；IQR 只標記極端值，不代表錯誤。
def build_numerical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = len(dataframe)

    for column in dataframe.select_dtypes(include="number").columns:
        series = dataframe[column]
        non_missing = series.dropna()
        count = int(non_missing.count())
        missing_count = total_rows - count

        if count:
            q1 = float(non_missing.quantile(0.25))
            median = float(non_missing.quantile(0.50))
            q3 = float(non_missing.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(((non_missing < lower_bound) | (non_missing > upper_bound)).sum())
            mean = float(non_missing.mean())
            std = float(non_missing.std()) if count > 1 else None
            minimum = float(non_missing.min())
            maximum = float(non_missing.max())
            zero_count = int(non_missing.eq(0).sum())
            negative_count = int(non_missing.lt(0).sum())
        else:
            q1 = median = q3 = iqr = lower_bound = upper_bound = None
            outlier_count = 0
            mean = std = minimum = maximum = None
            zero_count = negative_count = 0

        rows.append(
            {
                "feature_name": column,
                "dtype": str(series.dtype),
                "count": count,
                "missing_count": missing_count,
                "missing_ratio": missing_count / total_rows,
                "mean": mean,
                "std": std,
                "min": minimum,
                "25%": q1,
                "median": median,
                "75%": q3,
                "max": maximum,
                "unique_count": int(non_missing.nunique()),
                "zero_count": zero_count,
                "negative_count": negative_count,
                "iqr": iqr,
                "iqr_lower_bound": lower_bound,
                "iqr_upper_bound": upper_bound,
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_ratio": outlier_count / count if count else 0.0,
            }
        )

    return pd.DataFrame(rows)


# 類別摘要提供 top categories 與稀有類別規模；低於 10 筆的類別只標記、不合併。
def build_categorical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = len(dataframe)
    categorical_columns = [
        column
        for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
        or column in SEMANTIC_CATEGORICAL_COLUMNS
    ]

    for column in categorical_columns:
        series = dataframe[column]
        non_missing = series.dropna()
        counts = non_missing.value_counts(dropna=True)
        non_missing_count = int(non_missing.count())
        missing_count = total_rows - non_missing_count

        if counts.empty:
            most_frequent = None
            most_frequent_count = 0
            top_categories = []
        else:
            most_frequent = str(counts.index[0])
            most_frequent_count = int(counts.iloc[0])
            top_categories = [
                {
                    "value": str(value),
                    "count": int(count),
                    "ratio_non_missing": round(int(count) / non_missing_count, 6),
                }
                for value, count in counts.head(TOP_CATEGORY_COUNT).items()
            ]

        rare_counts = counts[counts <= RARE_CATEGORY_MAX_COUNT]
        rare_observation_count = int(rare_counts.sum())
        rows.append(
            {
                "feature_name": column,
                "dtype": str(series.dtype),
                "representation": representation_type(column),
                "non_missing_count": non_missing_count,
                "missing_count": missing_count,
                "missing_ratio": missing_count / total_rows,
                "unique_count": int(counts.size),
                "unique_ratio_non_missing": (
                    counts.size / non_missing_count if non_missing_count else 0.0
                ),
                "most_frequent_category": most_frequent,
                "most_frequent_count": most_frequent_count,
                "most_frequent_ratio_non_missing": (
                    most_frequent_count / non_missing_count if non_missing_count else 0.0
                ),
                "top_categories": json.dumps(top_categories, ensure_ascii=False),
                "rare_category_max_count": RARE_CATEGORY_MAX_COUNT,
                "rare_category_count": int(rare_counts.size),
                "rare_observation_count": rare_observation_count,
                "rare_observation_ratio_non_missing": (
                    rare_observation_count / non_missing_count if non_missing_count else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# 執行保守的格式與合理範圍檢查；規則刻意只抓明顯問題，避免把真實極端球員誤判為錯誤。
def collect_diagnostics(
    dataframe: pd.DataFrame, raw_strings: pd.DataFrame, missing_summary: pd.DataFrame
) -> dict[str, object]:
    total_rows = len(dataframe)
    target = dataframe[TARGET]

    date_invalid_counts = {}
    for column in DATE_COLUMNS:
        invalid = pd.to_datetime(dataframe[column], errors="coerce").isna() & dataframe[column].notna()
        date_invalid_counts[column] = int(invalid.sum())

    # 格式為 base + signed modifier；例如 86+3 與 69+-1 都是合法表示。
    position_pattern = re.compile(r"^\d{1,3}\+(?:-)?\d{1,2}$")
    position_invalid_counts = {}
    for column in POSITION_RATING_COLUMNS:
        values = dataframe[column].dropna().astype(str)
        position_invalid_counts[column] = int(
            (~values.str.fullmatch(position_pattern.pattern)).sum()
        )

    rating_columns = ["overall", "potential", *SUMMARY_RATING_COLUMNS, *DETAILED_RATING_COLUMNS]
    rating_out_of_range_count = 0
    for column in rating_columns:
        rating_out_of_range_count += int(
            ((dataframe[column] < 0) | (dataframe[column] > 100)).fillna(False).sum()
        )

    update_date = pd.to_datetime(dataframe["fifa_update_date"], errors="coerce")
    dob = pd.to_datetime(dataframe["dob"], errors="coerce")
    calculated_age = (
        update_date.dt.year
        - dob.dt.year
        - (
            (update_date.dt.month < dob.dt.month)
            | ((update_date.dt.month == dob.dt.month) & (update_date.dt.day < dob.dt.day))
        ).astype(int)
    )

    special_missing_total = int(missing_summary["special_missing_token_count_raw"].sum())
    blank_total = int(missing_summary["empty_string_count_raw"].sum())
    nbsp_cell_count = int(
        sum(raw_strings[column].str.contains("\u00a0", regex=False).sum() for column in raw_strings.columns)
    )

    return {
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "duplicate_player_id_count": int(dataframe["player_id"].duplicated().sum()),
        "all_missing_columns": missing_summary.loc[
            missing_summary["missing_ratio"].eq(1), "feature_name"
        ].tolist(),
        "constant_non_null_columns": missing_summary.loc[
            missing_summary["constant_non_null"] & missing_summary["missing_ratio"].lt(1),
            "feature_name",
        ].tolist(),
        "blank_cell_count_raw": blank_total,
        "special_missing_token_count_raw": special_missing_total,
        "nonbreaking_space_cell_count": nbsp_cell_count,
        "date_invalid_counts": date_invalid_counts,
        "position_invalid_counts": position_invalid_counts,
        "position_invalid_cell_count": int(sum(position_invalid_counts.values())),
        "position_invalid_row_count": int(
            dataframe[POSITION_RATING_COLUMNS]
            .apply(lambda col: ~col.astype(str).str.fullmatch(position_pattern.pattern))
            .any(axis=1)
            .sum()
        ),
        "rating_out_of_range_count": rating_out_of_range_count,
        "age_outside_15_60_count": int(((dataframe["age"] < 15) | (dataframe["age"] > 60)).sum()),
        "height_outside_140_230_count": int(
            ((dataframe["height_cm"] < 140) | (dataframe["height_cm"] > 230)).sum()
        ),
        "weight_outside_35_200_count": int(
            ((dataframe["weight_kg"] < 35) | (dataframe["weight_kg"] > 200)).sum()
        ),
        "potential_below_overall_count": int(
            (dataframe["potential"] < dataframe["overall"]).sum()
        ),
        "age_snapshot_mismatch_count": int(
            (calculated_age.notna() & calculated_age.ne(dataframe["age"])).sum()
        ),
        "target_zero_count": int(target.eq(0).sum()),
        "target_negative_count": int(target.lt(0).sum()),
        "wage_zero_count": int(dataframe["wage_eur"].eq(0).sum()),
        "currency_negative_count": int(
            sum(
                dataframe[column].lt(0).fillna(False).sum()
                for column in ["value_eur", "wage_eur", "release_clause_eur"]
            )
        ),
        "target_count": int(target.count()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "target_min": float(target.min()),
        "target_q1": float(target.quantile(0.25)),
        "target_median": float(target.median()),
        "target_q3": float(target.quantile(0.75)),
        "target_max": float(target.max()),
        "row_count": total_rows,
        "column_count": dataframe.shape[1],
    }


# 將全資料集的重要稽核結果整理成 key-value 表，便於其他程式或 paper 組引用。
def build_dataset_overview(
    dataframe: pd.DataFrame,
    numerical_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    diagnostics: dict[str, object],
) -> pd.DataFrame:
    target_numeric = numerical_summary.set_index("feature_name").loc[TARGET]
    high_missing_columns = int((dataframe.isna().mean() >= 0.5).sum())
    high_cardinality_categorical = int(
        (categorical_summary["unique_ratio_non_missing"] >= 0.5).sum()
    )

    rows = [
        ("row_count", diagnostics["row_count"], "原始資料列數"),
        ("column_count", diagnostics["column_count"], "原始資料欄數"),
        ("numeric_column_count", len(numerical_summary), "pandas 推論為數值的欄數"),
        ("categorical_summary_column_count", len(categorical_summary), "含數字編碼類別；可與數值摘要重疊"),
        ("target", TARGET, "已確認的 prediction target"),
        ("target_unit", "EUR", "FC／SoFIFA 估計球員身價"),
        ("target_missing_count", int(dataframe[TARGET].isna().sum()), "一般缺值"),
        ("target_zero_count", diagnostics["target_zero_count"], "零值語意待 schema 確認"),
        ("target_negative_count", diagnostics["target_negative_count"], "負值應為不可能值"),
        ("target_min", diagnostics["target_min"], "observed minimum"),
        ("target_max", diagnostics["target_max"], "observed maximum"),
        ("target_iqr_outlier_count", int(target_numeric["iqr_outlier_count"]), "描述性 IQR 標記；不代表錯誤"),
        ("duplicate_row_count", diagnostics["duplicate_row_count"], "完全重複資料列"),
        ("duplicate_player_id_count", diagnostics["duplicate_player_id_count"], "重複球員 ID"),
        ("all_missing_column_count", len(diagnostics["all_missing_columns"]), "100% 缺失欄位"),
        ("high_missing_column_count_ge_50pct", high_missing_columns, "missing ratio 至少 50%"),
        ("constant_non_null_column_count", len(diagnostics["constant_non_null_columns"]), "非缺值唯一值數為 1"),
        ("raw_blank_cell_count", diagnostics["blank_cell_count_raw"], "原始 CSV 空字串／純空白 cell"),
        ("raw_special_missing_token_count", diagnostics["special_missing_token_count_raw"], "不含空字串的特殊 token"),
        ("invalid_date_cell_count", sum(diagnostics["date_invalid_counts"].values()), "非缺值但無法解析的日期"),
        ("invalid_position_rating_cell_count", diagnostics["position_invalid_cell_count"], "不符合 base±modifier 格式"),
        ("invalid_position_rating_row_count", diagnostics["position_invalid_row_count"], "至少一個位置評分格式異常的列"),
        ("rating_out_of_0_100_count", diagnostics["rating_out_of_range_count"], "評分欄位保守值域檢查"),
        ("negative_currency_cell_count", diagnostics["currency_negative_count"], "三個 EUR 欄位的負值數"),
        ("age_snapshot_mismatch_count", diagnostics["age_snapshot_mismatch_count"], "age 與 update date/dob 足歲算法不一致"),
        ("nonbreaking_space_cell_count", diagnostics["nonbreaking_space_cell_count"], "至少含一個 non-breaking space 的 cell"),
        ("high_cardinality_categorical_count_ge_50pct", high_cardinality_categorical, "unique/non-missing ratio 至少 50%"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


# 主流程只負責依序建立並寫出結果，保持單一路徑以利重現與維護。
def main() -> None:
    input_path, result_dir = resolve_paths()
    dataframe, raw_strings = load_data(input_path)

    missing_summary = build_missing_summary(dataframe, raw_strings)
    numerical_summary = build_numerical_summary(dataframe)
    categorical_summary = build_categorical_summary(dataframe)
    diagnostics = collect_diagnostics(dataframe, raw_strings, missing_summary)
    dataset_overview = build_dataset_overview(
        dataframe, numerical_summary, categorical_summary, diagnostics
    )
    dataset_overview.to_csv(result_dir / "dataset_overview.csv", index=False)
    missing_summary.to_csv(result_dir / "missing_summary.csv", index=False, float_format="%.6f")
    numerical_summary.to_csv(
        result_dir / "numerical_summary.csv", index=False, float_format="%.6f"
    )
    categorical_summary.to_csv(
        result_dir / "categorical_summary.csv", index=False, float_format="%.6f"
    )
    print(f"Data audit completed: {len(dataframe):,} rows × {dataframe.shape[1]} columns")
    print(f"Target: {TARGET}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
