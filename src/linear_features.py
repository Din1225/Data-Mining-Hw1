"""
程式用途：為最終線性迴歸模型建立精簡、可解釋的特徵集（Set C），並提供消融實驗用的 Set B（相關性前 k 名）。

設計原則：
1. Set C 由原始欄位直接建構，約 30 個特徵，讓 §8.1 能為每個特徵報告 β、95% CI、p 值與 VIF。
   類別欄位只保留少數指示變數並設參考組，避免 one-hot 加總恆為 1 造成與截距完全共線。
2. 工程特徵分成四組，每組都可在消融實驗中單獨移除：
   - nonlinear：overall 與 age 的中心化平方項。（potential − overall 是 overall 與 potential 的線性組合，
     放進線性模型不會增加資訊，還會讓 VIF 無限大，因此不使用。）
   - interaction：overall × international_reputation（中心化後相乘），對應 H2。
   - domain_rules：自由球員、40 歲以上非門將、五大聯賽；前兩者對應遊戲中身價為 0 的規則。
   - aggregates：同球會、同聯賽球員的平均 overall（球隊與聯賽實力）。CV 消融顯示這組沒有任何改善
     （RMSLE 差 < 0.0001，5 個 fold 中只有 3 個略好），因此最終模型不使用，只在消融實驗中以
     「Set C + aggregates」保留比較。
3. 所有需要從資料學習的量（中心化的平均、補值中位數、球會／聯賽平均、標準化）都只在各 fold 的
   training subset 上計算；training 列的球會／聯賽平均採 leave-one-out，避免包含自己。
4. Set B 在每個 fold 的 training subset 上，依各欄與 value_eur 的 |Pearson r| 選出前 k 個 Set A 欄位；
   另有改依 log1p(value_eur) 排序的版本，檢查結論是否只是 raw 身價高度偏態造成的。

本模組不讀取 held-out test set。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from experiment_utils import ID_COLUMNS, TARGET, FoldData


REFERENCE_DATE = pd.Timestamp("2025-09-19")
SNAPSHOT_YEAR = 2025
# Premier League、La Liga、Serie A、Bundesliga、Ligue 1。
TOP5_LEAGUE_IDS = {13, 53, 31, 19, 16}
DEFENDER_POSITIONS = {"CB", "LB", "RB", "LWB", "RWB"}
FORWARD_POSITIONS = {"ST", "CF", "LW", "RW", "LF", "RF"}
SUMMARY_RATINGS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

# 特徵名稱 → 群組；群組名稱即 feature_mapping 的 transformer 欄，供 drop_features 移除整組。
FEATURE_GROUPS = {
    "overall": "base",
    "potential": "base",
    "age": "base",
    "height_cm": "base",
    "weight_kg": "base",
    "weak_foot": "base",
    "skill_moves": "base",
    "international_reputation": "base",
    "league_level": "base",
    "contract_years_left": "base",
    "club_tenure_years": "base",
    **{rating: "summary_ratings" for rating in SUMMARY_RATINGS},
    "is_goalkeeper": "position",
    "is_defender": "position",
    "is_forward": "position",
    "left_footed": "profile",
    "real_face": "profile",
    "unique_body_type": "profile",
    "in_national_team": "profile",
    "on_loan": "profile",
    "overall_sq_centered": "nonlinear",
    "age_sq_centered": "nonlinear",
    "overall_x_reputation": "interaction",
    "is_free_agent": "domain_rules",
    "age40_outfield": "domain_rules",
    "top5_league": "domain_rules",
    "club_mean_overall": "aggregates",
    "league_mean_overall": "aggregates",
}
# 最終模型使用的工程特徵組；aggregates 經消融確認無效，不納入最終模型。
ENGINEERED_GROUPS = ["nonlinear", "interaction", "domain_rules"]
FINAL_EXCLUDED_GROUPS = ("aggregates",)
RAW_COLUMNS = [
    *ID_COLUMNS, TARGET, "overall", "potential", "age", "height_cm", "weight_kg", "weak_foot",
    "skill_moves", "international_reputation", "league_id", "league_level", "club_name",
    "club_contract_valid_until_year", "club_joined_date", "club_loaned_from", "player_positions",
    "preferred_foot", "real_face", "body_type", "nation_team_id", *SUMMARY_RATINGS,
]


# 在 training subset 上學習中心化平均、補值中位數與球會／聯賽平均，再套用到任意列。
class CompactFeatureBuilder:
    def fit(self, frame: pd.DataFrame) -> "CompactFeatureBuilder":
        self.overall_mean_ = frame["overall"].mean()
        self.age_mean_ = frame["age"].mean()
        self.reputation_mean_ = frame["international_reputation"].mean()
        self.league_level_median_ = frame["league_level"].median()
        self.tenure_median_ = self._tenure(frame).median()
        self.club_stats_ = frame.groupby("club_name")["overall"].agg(["sum", "count"])
        self.league_stats_ = frame.groupby("league_id")["overall"].agg(["sum", "count"])
        return self

    # training 列的球會／聯賽平均不含自己（leave-one-out），與 validation 列「只看其他人」的情況一致。
    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame)._build(frame, leave_one_out=True)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._build(frame, leave_one_out=False)

    @staticmethod
    def _tenure(frame: pd.DataFrame) -> pd.Series:
        joined = pd.to_datetime(frame["club_joined_date"], errors="coerce")
        return (REFERENCE_DATE - joined).dt.days / 365.25

    def _group_mean(self, keys: pd.Series, overall: pd.Series, stats: pd.DataFrame, leave_one_out: bool) -> pd.Series:
        total = keys.map(stats["sum"])
        count = keys.map(stats["count"])
        if leave_one_out:
            total, count = total - overall, count - 1
        mean = total / count
        return mean.where(count > 0, self.overall_mean_).fillna(self.overall_mean_)

    def _build(self, frame: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
        positions = frame["player_positions"].str.split(",")
        primary = positions.str[0].str.strip()
        plays_goalkeeper = positions.apply(lambda tokens: "GK" in [t.strip() for t in tokens])
        free_agent = frame["club_name"].isna()
        overall_c = frame["overall"] - self.overall_mean_

        features = pd.DataFrame(index=frame.index)
        for column in ["overall", "potential", "age", "height_cm", "weight_kg", "weak_foot",
                       "skill_moves", "international_reputation"]:
            features[column] = frame[column].astype(float)
        features["league_level"] = frame["league_level"].fillna(self.league_level_median_)
        features["contract_years_left"] = (frame["club_contract_valid_until_year"] - SNAPSHOT_YEAR).fillna(0)
        features["club_tenure_years"] = self._tenure(frame).fillna(self.tenure_median_)
        # 門將沒有六項摘要能力值；填 0 表示不適用，差異由 is_goalkeeper 承擔。
        for rating in SUMMARY_RATINGS:
            features[rating] = frame[rating].fillna(0)
        features["is_goalkeeper"] = (primary == "GK").astype(float)
        features["is_defender"] = primary.isin(DEFENDER_POSITIONS).astype(float)
        features["is_forward"] = primary.isin(FORWARD_POSITIONS).astype(float)
        features["left_footed"] = (frame["preferred_foot"] == "Left").astype(float)
        features["real_face"] = (frame["real_face"] == "Yes").astype(float)
        features["unique_body_type"] = (frame["body_type"] == "Unique").astype(float)
        features["in_national_team"] = frame["nation_team_id"].notna().astype(float)
        features["on_loan"] = frame["club_loaned_from"].notna().astype(float)
        features["overall_sq_centered"] = overall_c**2
        features["age_sq_centered"] = (frame["age"] - self.age_mean_) ** 2
        features["overall_x_reputation"] = overall_c * (frame["international_reputation"] - self.reputation_mean_)
        features["is_free_agent"] = free_agent.astype(float)
        features["age40_outfield"] = ((frame["age"] >= 40) & ~plays_goalkeeper).astype(float)
        features["top5_league"] = frame["league_id"].isin(TOP5_LEAGUE_IDS).astype(float)
        features["club_mean_overall"] = self._group_mean(
            frame["club_name"], frame["overall"], self.club_stats_, leave_one_out
        )
        features["league_mean_overall"] = self._group_mean(
            frame["league_id"], frame["overall"], self.league_stats_, leave_one_out
        )
        if list(features.columns) != list(FEATURE_GROUPS):
            raise RuntimeError("Set C 特徵順序與 FEATURE_GROUPS 不一致")
        if features.isna().any().any():
            raise RuntimeError(f"Set C 特徵含缺值：{features.columns[features.isna().any()].tolist()}")
        return features


# 在一份 training 資料上 fit Set C 的特徵建構與標準化，再套用到其他資料；各 CV fold 與最終 test 共用。
class SetCTransformer:
    def __init__(self, exclude_groups: tuple[str, ...] = FINAL_EXCLUDED_GROUPS) -> None:
        self.exclude_groups = tuple(exclude_groups)
        self.columns = [name for name, group in FEATURE_GROUPS.items() if group not in self.exclude_groups]

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.builder_ = CompactFeatureBuilder()
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(self.builder_.fit_transform(frame)[self.columns])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(self.builder_.transform(frame)[self.columns])

    # feature_mapping 的 transformer 欄記錄群組，讓消融實驗可依群組移除。
    def mapping(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "output_index": range(len(self.columns)),
                "output_feature": self.columns,
                "source_feature": self.columns,
                "transformer": [FEATURE_GROUPS[name] for name in self.columns],
            }
        )


# 由 raw fold CSV 建立 Set C folds；目標、ID 與列順序沿用 Set A，並逐 fold 標準化（只在 training fit）。
def build_set_c_folds(
    split_dir: Path, set_a_folds: list[FoldData], exclude_groups: tuple[str, ...] = FINAL_EXCLUDED_GROUPS
) -> list[FoldData]:
    folds = []
    for data in set_a_folds:
        fold_dir = split_dir / "train_5-fold_cv" / f"fold_{data.fold}"
        train = pd.read_csv(fold_dir / "train.csv", usecols=RAW_COLUMNS, low_memory=False)
        valid = pd.read_csv(fold_dir / "validation.csv", usecols=RAW_COLUMNS, low_memory=False)
        if not valid["original_index"].equals(data.valid_ids["original_index"]):
            raise RuntimeError(f"Fold {data.fold}：Set C 與 Set A 的 validation 列順序不一致")
        if not np.array_equal(train[TARGET].to_numpy(dtype=float), data.y_train):
            raise RuntimeError(f"Fold {data.fold}：Set C 與 Set A 的 training target 不一致")

        transformer = SetCTransformer(exclude_groups)
        folds.append(
            FoldData(
                fold=data.fold,
                X_train=transformer.fit_transform(train),
                y_train=data.y_train,
                X_valid=transformer.transform(valid),
                y_valid=data.y_valid,
                valid_ids=data.valid_ids,
                feature_mapping=transformer.mapping(),
            )
        )
    return folds


# 稀疏矩陣每一欄與 y 的 Pearson r；常數欄回傳 0。
def column_pearson(X: sparse.csr_matrix | np.ndarray, y: np.ndarray) -> np.ndarray:
    X = sparse.csc_matrix(X)
    n = X.shape[0]
    mean_x = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    std_x = np.sqrt(np.maximum(mean_sq - mean_x**2, 0))
    y_centered = y - y.mean()
    cov = (X.T @ y_centered) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / (std_x * y_centered.std())
    return np.nan_to_num(r)


# Set B：在每個 fold 的 training subset 上，依 |Pearson r(feature, value_eur)| 選出前 k 個 Set A 欄位。
# log_target=True 時改用 log1p(value_eur) 計算相關係數（robustness 版本）。
def build_set_b_folds(set_a_folds: list[FoldData], k: int, log_target: bool = False) -> list[FoldData]:
    folds = []
    for data in set_a_folds:
        r = column_pearson(data.X_train, np.log1p(data.y_train) if log_target else data.y_train)
        selected = np.sort(np.argsort(-np.abs(r))[:k])
        mapping = data.feature_mapping.iloc[selected].reset_index(drop=True)
        mapping = mapping.assign(output_index=range(len(selected)), pearson_r=r[selected])
        folds.append(
            FoldData(
                fold=data.fold,
                X_train=data.X_train[:, selected],
                y_train=data.y_train,
                X_valid=data.X_valid[:, selected],
                y_valid=data.y_valid,
                valid_ids=data.valid_ids,
                feature_mapping=mapping,
            )
        )
    return folds
