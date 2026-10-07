"""
程式用途：建立全專案共用且可重現的 held-out test set 與 5-fold CV assignments。

主要執行流程：
1. 從 FC26_DATA_PATH 環境變數載入 repository 內的原始資料。
2. 使用固定參數切出 training portion 與完全獨立的 held-out test set。
3. 只在 training portion 上建立固定 K-Fold validation assignments。
4. 驗證 indices 完整、互斥且可重現，再保存資料、indices、manifest、統計與示意圖。

本程式不做 EDA 決策、preprocessing、feature selection 或模型訓練，也不讀取 test performance。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import KFold, train_test_split


# 所有切分參數集中在此處，後續實驗不得自行建立另一組 random split。
TARGET = "value_eur"
TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_SEED = 42
SHUFFLE = True
MEAN_SHIFT_WARNING_THRESHOLD = 0.10
MEDIAN_SHIFT_WARNING_THRESHOLD = 0.10
STD_RATIO_WARNING_LOWER = 0.80
STD_RATIO_WARNING_UPPER = 1.25
FOLD_MEAN_SHIFT_WARNING_THRESHOLD = 0.10


# 使用 Pillow 內建字型，讓切分示意圖不依賴 repository 外的字型檔。
def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# 畫出 training 內的 5-fold validation 與完全隔離的 held-out test。
def write_split_diagram(output_path: Path) -> None:
    image = Image.new("RGB", (1120, 360), "white")
    draw = ImageDraw.Draw(image)
    training_color = "#C7D5E0"
    validation_color = "#D45B43"
    test_color = "#183047"
    x_start, y_start = 150, 92
    cell_width, cell_height, gap = 34, 30, 3
    training_blocks, test_blocks = 20, 5

    draw.text((35, 22), "5-Fold Cross-Validation within Training Portion", fill="#102A43", font=get_font(22))
    training_end = x_start + training_blocks * (cell_width + gap) - gap
    test_start = x_start + training_blocks * (cell_width + gap)
    test_end = test_start + test_blocks * (cell_width + gap) - gap
    draw.text(((x_start + training_end) / 2, 62), "Training portion (80%)", fill="#40566B", font=get_font(14), anchor="mm")
    draw.text(((test_start + test_end) / 2, 62), "Held-out test (20%)", fill="#183047", font=get_font(14), anchor="mm")

    for fold in range(N_SPLITS):
        y = y_start + fold * (cell_height + 8)
        draw.text((55, y + 6), f"Fold {fold + 1}", fill="#40566B", font=get_font(14))
        validation_start = fold * 4
        validation_end = validation_start + 4
        for block in range(training_blocks + test_blocks):
            x = x_start + block * (cell_width + gap)
            if block >= training_blocks:
                color = test_color
            elif validation_start <= block < validation_end:
                color = validation_color
            else:
                color = training_color
            draw.rectangle((x, y, x + cell_width, y + cell_height), fill=color)

    legend_y = 305
    legend_items = [
        (training_color, "Fold training", 250),
        (validation_color, "Validation fold", 475),
        (test_color, "Held-out test (never used in CV)", 700),
    ]
    for color, label, x in legend_items:
        draw.rectangle((x, legend_y, x + 24, legend_y + 16), fill=color)
        draw.text((x + 34, legend_y - 1), label, fill="#333333", font=get_font(13))

    image.save(output_path, format="PNG", optimize=True)


# 從環境變數取得資料路徑，並限制輸入必須位於 repository 內。
def resolve_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    input_value = os.environ.get("FC26_DATA_PATH")
    if not input_value:
        raise RuntimeError(
            "請設定 FC26_DATA_PATH，例如："
            "FC26_DATA_PATH=data/FC26_20250921.csv python src/split_data.py"
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

    split_dir = repo_root / "data" / "splits"
    result_dir = repo_root / "results" / "data_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return input_path, split_dir, result_dir


# 對原始 CSV 計算內容指紋，防止資料內容改變後仍誤用舊 indices。
def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 確認資料符合目前一般隨機切分的前提；明確依賴存在時直接停止而非偷偷換方法。
def validate_dataset(dataframe: pd.DataFrame) -> dict[str, int]:
    required_columns = {
        "player_id",
        TARGET,
        "fifa_version",
        "fifa_update",
        "fifa_update_date",
        "club_team_id",
        "league_id",
    }
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise KeyError(f"資料缺少必要欄位：{missing_columns}")
    if dataframe.empty:
        raise ValueError("資料集是空的")
    if dataframe[TARGET].isna().any():
        raise ValueError("target 含缺值，建立 split 前必須先確認處理規則")

    duplicate_rows = int(dataframe.duplicated().sum())
    duplicate_entities = int(dataframe["player_id"].duplicated().sum())
    version_count = int(dataframe["fifa_version"].nunique(dropna=False))
    update_count = int(dataframe["fifa_update"].nunique(dropna=False))
    date_count = int(dataframe["fifa_update_date"].nunique(dropna=False))

    if duplicate_entities:
        raise RuntimeError(
            "同一 player_id 出現多筆資料，可能有 entity dependency；"
            "依需求停止，不自動改用 group split"
        )
    if version_count > 1 or update_count > 1 or date_count > 1:
        raise RuntimeError(
            "資料包含多版本或多時間點，可能有 temporal dependency；"
            "依需求停止，不自動改用 temporal split"
        )

    return {
        "duplicate_row_count": duplicate_rows,
        "duplicate_player_id_count": duplicate_entities,
        "fifa_version_count": version_count,
        "fifa_update_count": update_count,
        "fifa_update_date_count": date_count,
        "club_count": int(dataframe["club_team_id"].nunique(dropna=True)),
        "league_count": int(dataframe["league_id"].nunique(dropna=True)),
        "max_players_per_club": int(dataframe["club_team_id"].value_counts().max()),
        "max_players_per_league": int(dataframe["league_id"].value_counts().max()),
    }


# 先固定 held-out test，再只對 training indices 建立 validation fold；輸出皆使用原始 row index。
def create_splits(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    original_indices = np.arange(len(dataframe), dtype=int)
    train_indices, test_indices = train_test_split(
        original_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=SHUFFLE,
    )
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    fold_assignments = np.zeros(len(train_indices), dtype=int)
    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=SHUFFLE,
        random_state=RANDOM_SEED,
    )
    for fold, (_, validation_positions) in enumerate(kfold.split(train_indices), start=1):
        fold_assignments[validation_positions] = fold

    train_table = pd.DataFrame(
        {
            "original_index": train_indices,
            "player_id": dataframe.iloc[train_indices]["player_id"].to_numpy(),
        }
    )
    test_table = pd.DataFrame(
        {
            "original_index": test_indices,
            "player_id": dataframe.iloc[test_indices]["player_id"].to_numpy(),
        }
    )
    fold_table = train_table.copy()
    fold_table["fold"] = fold_assignments
    return train_table, test_table, fold_table


# 用集合與 player_id 雙重驗證 split，避免 row order 或寫檔錯誤造成資料交疊。
def validate_splits(
    dataframe: pd.DataFrame,
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    fold_table: pd.DataFrame,
) -> None:
    full_indices = set(range(len(dataframe)))
    train_indices = set(train_table["original_index"])
    test_indices = set(test_table["original_index"])

    if train_indices & test_indices:
        raise RuntimeError("training portion 與 held-out test indices 發生重疊")
    if train_indices | test_indices != full_indices:
        raise RuntimeError("train/test indices 沒有完整覆蓋原始資料")
    if len(train_table) + len(test_table) != len(dataframe):
        raise RuntimeError("train/test sample count 不正確")
    if train_table["player_id"].duplicated().any() or test_table["player_id"].duplicated().any():
        raise RuntimeError("split 中出現重複 player_id")
    if set(train_table["player_id"]) & set(test_table["player_id"]):
        raise RuntimeError("同一 player_id 同時出現在 train 與 test")
    if not fold_table["original_index"].equals(train_table["original_index"]):
        raise RuntimeError("fold assignments 與 training indices 順序不一致")
    if set(fold_table["fold"].unique()) != set(range(1, N_SPLITS + 1)):
        raise RuntimeError("fold labels 不完整")
    if fold_table["original_index"].duplicated().any():
        raise RuntimeError("同一 training sample 被重複指派 validation fold")


# 統計 full/train/test 及每個 validation fold 的 target 分布，不據此重新抽樣。
def build_split_summary(
    dataframe: pd.DataFrame,
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    fold_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add_row(
        partition: str,
        indices: np.ndarray,
        fold: int | None = None,
        fold_training_count: int | None = None,
    ) -> None:
        target = dataframe.iloc[indices][TARGET]
        rows.append(
            {
                "partition": partition,
                "fold": fold,
                "sample_count": len(indices),
                "fold_training_count": fold_training_count,
                "fold_validation_count": len(indices) if fold is not None else None,
                "target_mean": target.mean(),
                "target_std": target.std(),
                "target_q1": target.quantile(0.25),
                "target_median": target.median(),
                "target_q3": target.quantile(0.75),
                "target_min": target.min(),
                "target_max": target.max(),
                "target_zero_count": int(target.eq(0).sum()),
            }
        )

    full_indices = np.arange(len(dataframe), dtype=int)
    train_indices = train_table["original_index"].to_numpy()
    test_indices = test_table["original_index"].to_numpy()
    add_row("full_dataset", full_indices)
    add_row("training_portion", train_indices)
    add_row("held_out_test", test_indices)

    for fold in range(1, N_SPLITS + 1):
        validation_indices = fold_table.loc[
            fold_table["fold"].eq(fold), "original_index"
        ].to_numpy()
        add_row(
            "cv_validation",
            validation_indices,
            fold=fold,
            fold_training_count=len(train_indices) - len(validation_indices),
        )

    return pd.DataFrame(rows)


# 以事先固定的門檻描述分布差異；只回報結果，不反覆抽樣挑選較漂亮的 split。
def assess_distribution(summary: pd.DataFrame) -> dict[str, float | bool]:
    indexed = summary.set_index("partition")
    full = indexed.loc["full_dataset"]
    train = indexed.loc["training_portion"]
    test = indexed.loc["held_out_test"]

    full_std = float(full["target_std"])
    full_iqr_reference = float(full["target_q3"]) - float(full["target_q1"])
    standardized_mean_difference = abs(
        float(train["target_mean"]) - float(test["target_mean"])
    ) / full_std
    normalized_median_difference = abs(
        float(train["target_median"]) - float(test["target_median"])
    ) / full_iqr_reference
    std_ratio = float(test["target_std"]) / float(train["target_std"])

    fold_rows = summary.loc[summary["partition"].eq("cv_validation")]
    max_fold_mean_deviation = (
        (fold_rows["target_mean"] - float(train["target_mean"])).abs().max() / full_std
    )
    obvious_shift = bool(
        standardized_mean_difference >= MEAN_SHIFT_WARNING_THRESHOLD
        or normalized_median_difference >= MEDIAN_SHIFT_WARNING_THRESHOLD
        or not STD_RATIO_WARNING_LOWER <= std_ratio <= STD_RATIO_WARNING_UPPER
        or max_fold_mean_deviation >= FOLD_MEAN_SHIFT_WARNING_THRESHOLD
    )
    return {
        "standardized_train_test_mean_difference": standardized_mean_difference,
        "normalized_train_test_median_difference": normalized_median_difference,
        "test_train_std_ratio": std_ratio,
        "max_standardized_fold_mean_deviation": float(max_fold_mean_deviation),
        "obvious_distribution_shift": obvious_shift,
    }


# Manifest 固定資料內容與 split 設定；不同資料或參數不得靜默覆寫既有 split。
def build_manifest(
    input_path: Path,
    dataframe: pd.DataFrame,
    dataset_checks: dict[str, int],
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "source_file": str(input_path.relative_to(repo_root)),
        "source_sha256": calculate_sha256(input_path),
        "row_count": len(dataframe),
        "column_count": dataframe.shape[1],
        "target": TARGET,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
        "shuffle": SHUFFLE,
        "index_definition": "zero-based pandas original row index after reading the source CSV",
        "entity_key": "player_id",
        "dataset_checks": dataset_checks,
    }


# 若 split 已存在，只允許寫入完全相同的內容，確保 final test set 不會被重新抽樣。
def verify_or_write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        expected = dataframe.reset_index(drop=True)
        if not existing.equals(expected):
            raise RuntimeError(f"既有固定 split 與本次結果不同，拒絕覆寫：{path}")
        return
    dataframe.to_csv(path, index=False)


# Manifest 採相同保護策略，資料指紋或參數一旦不同就停止。
def verify_or_write_manifest(manifest: dict[str, object], path: Path) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"資料或 split 參數已改變，拒絕覆寫既有 manifest：{path}")
        return
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# 依固定 indices 輸出可直接讀取的 train、test 與五個互斥 CV fold 資料檔。
def write_split_datasets(
    dataframe: pd.DataFrame,
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    fold_table: pd.DataFrame,
    split_dir: Path,
) -> None:
    def export_rows(indices: np.ndarray, path: Path) -> None:
        subset = dataframe.iloc[indices].copy()
        subset.insert(0, "original_index", indices)
        subset.to_csv(path, index=False)

    export_rows(train_table["original_index"].to_numpy(), split_dir / "train.csv")
    export_rows(test_table["original_index"].to_numpy(), split_dir / "test.csv")

    cv_dir = split_dir / "train_5-fold_cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, N_SPLITS + 1):
        validation_indices = fold_table.loc[
            fold_table["fold"].eq(fold), "original_index"
        ].to_numpy()
        fold_train_indices = fold_table.loc[
            fold_table["fold"].ne(fold), "original_index"
        ].to_numpy()
        fold_dir = cv_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        export_rows(fold_train_indices, fold_dir / "train.csv")
        export_rows(validation_indices, fold_dir / "validation.csv")
        pd.DataFrame({"original_index": fold_train_indices}).to_csv(
            fold_dir / "train_indices.csv", index=False
        )
        pd.DataFrame({"original_index": validation_indices}).to_csv(
            fold_dir / "validation_indices.csv", index=False
        )


# 主流程依固定順序完成檢查、切分、鎖定輸出與示意圖產生。
def main() -> None:
    input_path, split_dir, result_dir = resolve_paths()
    dataframe = pd.read_csv(input_path, low_memory=False)
    dataset_checks = validate_dataset(dataframe)
    train_table, test_table, fold_table = create_splits(dataframe)
    validate_splits(dataframe, train_table, test_table, fold_table)

    summary = build_split_summary(dataframe, train_table, test_table, fold_table)
    distribution = assess_distribution(summary)
    manifest = build_manifest(input_path, dataframe, dataset_checks)

    verify_or_write_manifest(manifest, result_dir / "split_manifest.json")
    verify_or_write_csv(train_table, split_dir / "train_indices.csv")
    verify_or_write_csv(test_table, split_dir / "test_indices.csv")
    verify_or_write_csv(fold_table, result_dir / "cv_fold_assignments.csv")
    write_split_datasets(dataframe, train_table, test_table, fold_table, split_dir)
    write_split_diagram(split_dir / "split_diagram.png")

    summary.to_csv(result_dir / "split_summary.csv", index=False, float_format="%.6f")
    print(f"Full dataset: {len(dataframe):,}")
    print(f"Training portion: {len(train_table):,}")
    print(f"Held-out test: {len(test_table):,}")
    print(f"CV folds: {N_SPLITS}")
    print(f"Random seed: {RANDOM_SEED}; shuffle={SHUFFLE}")
    print(f"Obvious target distribution shift: {distribution['obvious_distribution_shift']}")


if __name__ == "__main__":
    main()
