"""
程式用途：建立 FC 26 球員身價預測的正式 sklearn preprocessing pipeline。

主要執行流程：
1. 從環境變數讀取已固定的 train.csv 與 test.csv。
2. 建立 overall 平方項與 overall × international_reputation 交互作用。
3. 只在 training portion fit imputer、rare grouping、encoder 與 scaler。
4. 使用同一個 fitted preprocessor transform train 與 held-out test。
5. 輸出 sparse matrices、逐欄決策、encoding mapping 與 training-only outlier 摘要。

本程式不刪除資料、不使用 test target 做決策、不做 feature selection，也不訓練模型。
後續 Cross-Validation 必須將 build_preprocessor() 建立的物件包在模型 Pipeline 中，
並在每個 fold 的 training subset 重新 fit，不能直接使用本程式輸出的全 training fitted matrix 做 CV。
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


TARGET = "value_eur"
REFERENCE_DATE = "2025-09-19"
RARE_MIN_FREQUENCY = 10

IDENTIFIER_COLUMNS = [
    "original_index",
    "player_id",
    "player_url",
    "short_name",
    "long_name",
    "player_face_url",
]
LEAKAGE_COLUMNS = ["wage_eur", "release_clause_eur"]
CONSTANT_COLUMNS = ["fifa_version", "fifa_update", "fifa_update_date"]
ALL_MISSING_COLUMNS = ["work_rate"]
REDUNDANT_COLUMNS = ["dob", "league_id", "club_team_id", "nationality_id"]

NUMERIC_COLUMNS = [
    "overall",
    "potential",
    "age",
    "height_cm",
    "weight_kg",
    "league_level",
    "club_contract_valid_until_year",
    "weak_foot",
    "skill_moves",
    "international_reputation",
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]

STRUCTURAL_NUMERIC_COLUMNS = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "goalkeeping_speed",
]

GENERAL_CATEGORICAL_COLUMNS = [
    "league_name",
    "club_name",
    "club_position",
    "club_jersey_number",
    "nationality_name",
    "preferred_foot",
    "body_type",
    "real_face",
]

NATIONAL_TEAM_COLUMNS = [
    "nation_team_id",
    "nation_position",
    "nation_jersey_number",
]

MULTI_VALUE_COLUMNS = ["player_positions", "player_tags", "player_traits"]

POSITION_RATING_COLUMNS = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram",
    "lm", "lcm", "cm", "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb",
    "lb", "lcb", "cb", "rcb", "rb", "gk",
]

ENGINEERED_SOURCE_COLUMNS = ["overall", "international_reputation"]
ENGINEERED_FEATURE_SOURCES = {
    "overall_squared": "overall",
    "overall_x_reputation": "overall + international_reputation",
}


# 將一般類別值統一 Unicode 與空白，不改變大小寫或虛構順序。
class CategoricalCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, missing_label: str = "__MISSING__") -> None:
        self.missing_label = missing_label

    def fit(self, X: object, y: object = None) -> "CategoricalCleaner":
        frame = self._to_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: object) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        frame = self._to_frame(X)
        cleaned = frame.apply(lambda column: column.map(self._clean_value))
        return cleaned.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_

    @staticmethod
    def _to_frame(X: object) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return pd.DataFrame(array)

    def _clean_value(self, value: object) -> str:
        if pd.isna(value) or str(value).strip() == "":
            return self.missing_label
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        text = unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ")
        return re.sub(r"\s+", " ", text).strip()


# 對逗號分隔的多選欄位做 training-only vocabulary 與 rare-token grouping。
class MultiValueEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, min_frequency: int = RARE_MIN_FREQUENCY) -> None:
        self.column_name = column_name
        self.min_frequency = min_frequency

    def fit(self, X: object, y: object = None) -> "MultiValueEncoder":
        values = self._to_series(X)
        counts: Counter[str] = Counter()
        for value in values:
            counts.update(set(self._parse_tokens(value)))

        self.frequent_tokens_ = {
            token for token, count in counts.items() if count >= self.min_frequency
        }
        classes = sorted(self.frequent_tokens_) + ["__MISSING__", "__RARE__"]
        self.classes_ = np.asarray(classes, dtype=object)
        self.class_to_index_ = {value: index for index, value in enumerate(self.classes_)}
        return self

    def transform(self, X: object) -> sparse.csr_matrix:
        check_is_fitted(self, "class_to_index_")
        values = self._to_series(X)
        row_indices: list[int] = []
        column_indices: list[int] = []

        for row_index, value in enumerate(values):
            tokens = self._parse_tokens(value)
            if not tokens:
                encoded_tokens = {"__MISSING__"}
            else:
                encoded_tokens = {
                    token if token in self.frequent_tokens_ else "__RARE__"
                    for token in tokens
                }
            for token in encoded_tokens:
                row_indices.append(row_index)
                column_indices.append(self.class_to_index_[token])

        data = np.ones(len(row_indices), dtype=np.float64)
        return sparse.csr_matrix(
            (data, (row_indices, column_indices)),
            shape=(len(values), len(self.classes_)),
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "classes_")
        return np.asarray(
            [f"{self.column_name}__{token}" for token in self.classes_],
            dtype=object,
        )

    @staticmethod
    def _to_series(X: object) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        array = np.asarray(X, dtype=object).reshape(-1)
        return pd.Series(array)

    @staticmethod
    def _parse_tokens(value: object) -> list[str]:
        if pd.isna(value) or str(value).strip() == "":
            return []
        normalized = unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ")
        tokens = []
        for raw_token in normalized.split(","):
            token = re.sub(r"\s+", " ", raw_token).strip()
            token = token.removeprefix("#").strip()
            if token:
                tokens.append(token)
        return tokens


# 將 86+3 或 69+-1 解析成兩個數值欄位，不從資料學習任何參數。
class PositionRatingParser(BaseEstimator, TransformerMixin):
    _pattern = re.compile(r"^(\d{1,3})\+(-?\d{1,2})$")

    def fit(self, X: object, y: object = None) -> "PositionRatingParser":
        frame = self._to_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: object) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        frame = self._to_frame(X)
        output = np.full((len(frame), len(frame.columns) * 2), np.nan, dtype=float)

        for column_index, column in enumerate(frame.columns):
            parsed = frame[column].astype(str).str.extract(self._pattern)
            output[:, column_index * 2] = pd.to_numeric(parsed[0], errors="coerce")
            output[:, column_index * 2 + 1] = pd.to_numeric(parsed[1], errors="coerce")
        return output

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = []
        for column in self.feature_names_in_:
            names.extend([f"{column}_base", f"{column}_modifier"])
        return np.asarray(names, dtype=object)

    @staticmethod
    def _to_frame(X: object) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        array = np.asarray(X, dtype=object)
        return pd.DataFrame(array)


# 把加入球會日期轉為截至固定 snapshot 的年資，避免直接把日期字串做 ordinal encoding。
class ClubTenureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date: str = REFERENCE_DATE) -> None:
        self.reference_date = reference_date

    def fit(self, X: object, y: object = None) -> "ClubTenureTransformer":
        self.reference_date_ = pd.Timestamp(self.reference_date)
        return self

    def transform(self, X: object) -> np.ndarray:
        check_is_fitted(self, "reference_date_")
        if isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            values = X
        else:
            values = pd.Series(np.asarray(X, dtype=object).reshape(-1))
        parsed = pd.to_datetime(values, errors="coerce")
        tenure_years = (self.reference_date_ - parsed).dt.days / 365.25
        tenure_years = tenure_years.where(tenure_years >= 0, np.nan)
        return tenure_years.to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(["club_tenure_years"], dtype=object)


# 由已知的球員能力與聲望建立平方項及交互作用，讓線性模型表達曲線與聯合效果。
class PlayerValueFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: object, y: object = None) -> "PlayerValueFeatureEngineer":
        self.feature_names_in_ = np.asarray(ENGINEERED_SOURCE_COLUMNS, dtype=object)
        return self

    def transform(self, X: object) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if isinstance(X, pd.DataFrame):
            frame = X.loc[:, ENGINEERED_SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        else:
            frame = pd.DataFrame(np.asarray(X), columns=ENGINEERED_SOURCE_COLUMNS).apply(
                pd.to_numeric, errors="coerce"
            )

        overall = frame["overall"].to_numpy(dtype=float)
        reputation = frame["international_reputation"].to_numpy(dtype=float)
        return np.column_stack(
            [
                overall**2,
                overall * reputation,
            ]
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(list(ENGINEERED_FEATURE_SOURCES), dtype=object)


# 建立 unfitted preprocessor；CV 時每個模型 Pipeline 都應呼叫此函式取得新 instance。
def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    structural_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    def categorical_pipeline(missing_label: str) -> Pipeline:
        return Pipeline(
            steps=[
                ("cleaner", CategoricalCleaner(missing_label=missing_label)),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=RARE_MIN_FREQUENCY,
                        sparse_output=True,
                    ),
                ),
            ]
        )

    tenure_pipeline = Pipeline(
        steps=[
            ("tenure", ClubTenureTransformer(reference_date=REFERENCE_DATE)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    position_rating_pipeline = Pipeline(
        steps=[
            ("parser", PositionRatingParser()),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    engineered_feature_pipeline = Pipeline(
        steps=[
            ("feature_engineer", PlayerValueFeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("engineered", engineered_feature_pipeline, ENGINEERED_SOURCE_COLUMNS),
            ("structural_numeric", structural_numeric_pipeline, STRUCTURAL_NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline("__MISSING__"), GENERAL_CATEGORICAL_COLUMNS),
            ("loan_status", categorical_pipeline("__NOT_ON_LOAN__"), ["club_loaned_from"]),
            ("national_team", categorical_pipeline("__NOT_SELECTED__"), NATIONAL_TEAM_COLUMNS),
            ("club_tenure", tenure_pipeline, ["club_joined_date"]),
            (
                "player_positions",
                MultiValueEncoder("player_positions", RARE_MIN_FREQUENCY),
                ["player_positions"],
            ),
            (
                "player_tags",
                MultiValueEncoder("player_tags", RARE_MIN_FREQUENCY),
                ["player_tags"],
            ),
            (
                "player_traits",
                MultiValueEncoder("player_traits", RARE_MIN_FREQUENCY),
                ["player_traits"],
            ),
            ("position_ratings", position_rating_pipeline, POSITION_RATING_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


# 大型輸入檔必須由環境變數指定，且限制於 repository 內。
def resolve_paths() -> tuple[Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    train_value = os.environ.get("FC26_TRAIN_PATH")
    test_value = os.environ.get("FC26_TEST_PATH")
    if not train_value or not test_value:
        raise RuntimeError(
            "請設定 FC26_TRAIN_PATH 與 FC26_TEST_PATH，例如：\n"
            "FC26_TRAIN_PATH=data/splits/train.csv "
            "FC26_TEST_PATH=data/splits/test.csv python src/preprocess.py"
        )

    def resolve_input(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError("輸入資料必須位於 repository 內") from exc
        if not path.is_file():
            raise FileNotFoundError(f"找不到資料檔：{path}")
        return path

    train_path = resolve_input(train_value)
    test_path = resolve_input(test_value)
    result_dir = repo_root / "results" / "preprocessing"
    processed_dir = repo_root / "data" / "processed_splits"
    result_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return train_path, test_path, result_dir, processed_dir


# 檢查實體 split 與欄位，避免誤把 test 當 train 或兩者交疊。
def load_and_validate_splits(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    if train.columns.tolist() != test.columns.tolist():
        raise RuntimeError("raw train/test 欄位不一致")
    required = {TARGET, "original_index", *NUMERIC_COLUMNS, *POSITION_RATING_COLUMNS}
    missing = sorted(required - set(train.columns))
    if missing:
        raise KeyError(f"split 缺少 preprocessing 必要欄位：{missing}")
    if train["original_index"].duplicated().any() or test["original_index"].duplicated().any():
        raise RuntimeError("split 內出現重複 original_index")
    if set(train["original_index"]) & set(test["original_index"]):
        raise RuntimeError("train/test original_index 發生重疊")
    return train, test


# 逐一記錄每個 raw 欄位的角色與處理理由，避免只在程式碼中留下隱含決策。
def build_input_decisions(columns: list[str]) -> pd.DataFrame:
    decisions = []
    groups = {
        **{column: ("exclude", "identifier", "不進模型", "識別／記憶風險") for column in IDENTIFIER_COLUMNS},
        **{column: ("exclude", "leakage", "不進模型", "prediction time 不可取得的經濟結果") for column in LEAKAGE_COLUMNS},
        **{column: ("exclude", "constant", "不進模型", "training data 中為常數 metadata") for column in CONSTANT_COLUMNS},
        **{column: ("exclude", "all_missing", "不進模型", "training data 中 100% 缺失") for column in ALL_MISSING_COLUMNS},
        **{column: ("exclude", "redundant", "不進模型", "已有較直接欄位；避免重複表示") for column in REDUNDANT_COLUMNS},
        **{column: ("feature", "numeric", "median imputation + StandardScaler", "數值／序位特徵；median 對極端值較穩健") for column in NUMERIC_COLUMNS},
        **{column: ("feature", "structural_numeric", "0 表示不適用 + StandardScaler", "位置造成的結構性缺值，不視為一般隨機缺失") for column in STRUCTURAL_NUMERIC_COLUMNS},
        **{column: ("feature", "categorical", "清理 + missing category + rare grouping + one-hot", "名目類別無自然順序") for column in GENERAL_CATEGORICAL_COLUMNS},
        **{column: ("feature", "national_team", "not-selected category + rare grouping + one-hot", "缺值代表未入選國家隊") for column in NATIONAL_TEAM_COLUMNS},
        **{column: ("feature", "multi_value", "tokenize + rare token grouping + multi-hot", "逗號分隔多選欄位") for column in MULTI_VALUE_COLUMNS},
        **{column: ("feature", "position_rating", "解析 base/modifier + median imputation + scaling", "特殊數值字串，不作名目類別") for column in POSITION_RATING_COLUMNS},
        "club_loaned_from": ("feature", "loan_status", "not-on-loan category + rare grouping + one-hot", "缺值主要代表沒有外借"),
        "club_joined_date": ("feature", "date", "轉 club tenure + median imputation + missing indicator + scaling", "日期改為相對 snapshot 的可解釋 duration"),
        TARGET: ("target", "target", "不進 X；保持原始 EUR", "後續可在 CV 內另行評估 log1p target"),
    }

    for column in columns:
        if column not in groups:
            raise RuntimeError(f"尚未記錄 preprocessing decision 的欄位：{column}")
        role, transformer, method, reason = groups[column]
        decisions.append(
            {
                "feature_name": column,
                "role": role,
                "transformer": transformer,
                "method": method,
                "reason": reason,
            }
        )
    return pd.DataFrame(decisions)


# 只使用 training portion 計算 IQR 警示；不據此刪除、clip 或 winsorize。
def build_train_outlier_summary(train: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in [TARGET, *feature_columns]:
        if column not in train.columns or not pd.api.types.is_numeric_dtype(train[column]):
            continue
        values = train[column].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (values < lower) | (values > upper)
        rows.append(
            {
                "feature_name": column,
                "count": len(values),
                "min": values.min(),
                "q1": q1,
                "median": values.median(),
                "q3": q3,
                "max": values.max(),
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "iqr_outlier_count": int(outliers.sum()),
                "iqr_outlier_ratio": float(outliers.mean()),
                "action": "keep_no_clipping",
                "reason": "值域檢查未顯示資料錯誤；極端球員可能是真實觀測",
            }
        )
    return pd.DataFrame(rows)


# 將 encoding 後欄名對回來源 feature 與 transformer，供解釋與除錯使用。
def build_feature_mapping(preprocessor: ColumnTransformer) -> pd.DataFrame:
    output_names = preprocessor.get_feature_names_out()
    transformer_sources = {
        name: list(columns) if not isinstance(columns, str) else [columns]
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
    }
    rows = []

    for output_index, output_name in enumerate(output_names):
        transformer, encoded_name = output_name.split("__", 1)
        source_candidates = transformer_sources[transformer]
        source_feature = ENGINEERED_FEATURE_SOURCES.get(encoded_name, "")
        if not source_feature:
            for candidate in sorted(source_candidates, key=len, reverse=True):
                normalized = encoded_name.removeprefix("missingindicator_")
                if normalized == candidate or normalized.startswith(f"{candidate}_") or normalized.startswith(f"{candidate}__"):
                    source_feature = candidate
                    break
        if not source_feature and len(source_candidates) == 1:
            source_feature = source_candidates[0]

        rows.append(
            {
                "output_index": output_index,
                "output_feature": output_name,
                "source_feature": source_feature,
                "transformer": transformer,
            }
        )
    return pd.DataFrame(rows)


# 主流程只在 train fit，test 僅 transform，最後輸出可追蹤的 mapping 與驗證資訊。
def main() -> None:
    train_path, test_path, result_dir, processed_dir = resolve_paths()
    train, test = load_and_validate_splits(train_path, test_path)
    decisions = build_input_decisions(train.columns.tolist())

    X_train = train.drop(columns=[TARGET])
    X_test = test.drop(columns=[TARGET])
    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    transformed_train = sparse.csr_matrix(transformed_train)
    transformed_test = sparse.csr_matrix(transformed_test)
    feature_mapping = build_feature_mapping(preprocessor)
    used_source_features = sorted(
        set(NUMERIC_COLUMNS)
        | set(STRUCTURAL_NUMERIC_COLUMNS)
        | set(GENERAL_CATEGORICAL_COLUMNS)
        | set(NATIONAL_TEAM_COLUMNS)
        | set(MULTI_VALUE_COLUMNS)
        | set(POSITION_RATING_COLUMNS)
        | {"club_loaned_from", "club_joined_date"}
    )
    train_outliers = build_train_outlier_summary(train, used_source_features)

    if transformed_train.shape[1] != transformed_test.shape[1]:
        raise RuntimeError("processed train/test feature 數量不一致")
    if transformed_train.shape[1] != len(feature_mapping):
        raise RuntimeError("feature mapping 與 processed matrix 欄數不一致")
    if np.isnan(transformed_train.data).any() or np.isnan(transformed_test.data).any():
        raise RuntimeError("processed matrix 仍含 NaN")
    if not np.isfinite(transformed_train.data).all() or not np.isfinite(transformed_test.data).all():
        raise RuntimeError("processed matrix 含 infinite value")
    if transformed_train.shape[0] != len(train) or transformed_test.shape[0] != len(test):
        raise RuntimeError("preprocessing 意外改變 row count")

    sparse.save_npz(processed_dir / "train.npz", transformed_train)
    sparse.save_npz(processed_dir / "test.npz", transformed_test)
    pd.DataFrame({"original_index": train["original_index"]}).to_csv(
        processed_dir / "train_indices.csv", index=False
    )
    pd.DataFrame({"original_index": test["original_index"]}).to_csv(
        processed_dir / "test_indices.csv", index=False
    )
    feature_mapping.to_csv(processed_dir / "feature_mapping.csv", index=False)

    decisions.to_csv(result_dir / "input_feature_decisions.csv", index=False)
    feature_mapping.to_csv(result_dir / "feature_mapping.csv", index=False)
    train_outliers.to_csv(result_dir / "train_outlier_summary.csv", index=False)

    validation_rows: list[dict[str, object]] = [
        {
            "scope": "full_training_to_held_out_test",
            "fold": None,
            "fit_scope": "training_portion_only",
            "train_rows": transformed_train.shape[0],
            "validation_rows": transformed_test.shape[0],
            "feature_count": transformed_train.shape[1],
            "train_nan_count": int(np.isnan(transformed_train.data).sum()),
            "validation_nan_count": int(np.isnan(transformed_test.data).sum()),
            "train_infinite_count": int((~np.isfinite(transformed_train.data)).sum()),
            "validation_infinite_count": int((~np.isfinite(transformed_test.data)).sum()),
            "rows_deleted": 0,
            "rare_min_frequency": RARE_MIN_FREQUENCY,
            "reference_date": REFERENCE_DATE,
        }
    ]

    # 每一折建立獨立 preprocessor，絕不使用完整 training portion 的 fitted statistics。
    raw_cv_dir = train_path.parent / "train_5-fold_cv"
    processed_cv_dir = processed_dir / "train_5-fold_cv"
    processed_cv_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        raw_fold_dir = raw_cv_dir / f"fold_{fold}"
        fold_train = pd.read_csv(raw_fold_dir / "train.csv", low_memory=False)
        fold_validation = pd.read_csv(raw_fold_dir / "validation.csv", low_memory=False)
        if fold_train.columns.tolist() != train.columns.tolist() or fold_validation.columns.tolist() != train.columns.tolist():
            raise RuntimeError(f"Fold {fold} raw 欄位與完整 training 不一致")

        fold_preprocessor = build_preprocessor()
        fold_train_matrix = sparse.csr_matrix(
            fold_preprocessor.fit_transform(fold_train.drop(columns=[TARGET]))
        )
        fold_validation_matrix = sparse.csr_matrix(
            fold_preprocessor.transform(fold_validation.drop(columns=[TARGET]))
        )
        fold_mapping = build_feature_mapping(fold_preprocessor)

        if fold_train_matrix.shape[1] != fold_validation_matrix.shape[1]:
            raise RuntimeError(f"Fold {fold} train/validation feature 數不一致")
        if fold_train_matrix.shape[1] != len(fold_mapping):
            raise RuntimeError(f"Fold {fold} mapping 欄數不一致")
        if not np.isfinite(fold_train_matrix.data).all() or not np.isfinite(fold_validation_matrix.data).all():
            raise RuntimeError(f"Fold {fold} processed matrix 含 NaN 或 infinite")

        output_fold_dir = processed_cv_dir / f"fold_{fold}"
        output_fold_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(output_fold_dir / "train.npz", fold_train_matrix)
        sparse.save_npz(output_fold_dir / "validation.npz", fold_validation_matrix)
        pd.DataFrame({"original_index": fold_train["original_index"]}).to_csv(
            output_fold_dir / "train_indices.csv", index=False
        )
        pd.DataFrame({"original_index": fold_validation["original_index"]}).to_csv(
            output_fold_dir / "validation_indices.csv", index=False
        )
        fold_mapping.to_csv(output_fold_dir / "feature_mapping.csv", index=False)

        fold_row = {
            "scope": "cv_fold",
            "fold": fold,
            "fit_scope": f"fold_{fold}_training_only",
            "train_rows": fold_train_matrix.shape[0],
            "validation_rows": fold_validation_matrix.shape[0],
            "feature_count": fold_train_matrix.shape[1],
            "train_nan_count": int(np.isnan(fold_train_matrix.data).sum()),
            "validation_nan_count": int(np.isnan(fold_validation_matrix.data).sum()),
            "train_infinite_count": int((~np.isfinite(fold_train_matrix.data)).sum()),
            "validation_infinite_count": int((~np.isfinite(fold_validation_matrix.data)).sum()),
            "rows_deleted": 0,
            "rare_min_frequency": RARE_MIN_FREQUENCY,
            "reference_date": REFERENCE_DATE,
        }
        validation_rows.append(fold_row)

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(result_dir / "validation_summary.csv", index=False)
    print(f"Fit rows: {len(train):,}; transform-only test rows: {len(test):,}")
    print(f"Processed features: {transformed_train.shape[1]:,}")
    print(f"Train shape: {transformed_train.shape}; test shape: {transformed_test.shape}")
    print("Rows deleted: 0")


if __name__ == "__main__":
    main()
