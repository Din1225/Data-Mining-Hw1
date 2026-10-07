"""
程式用途：只使用固定 training split，產生作業報告需要的精簡 Feature–Target 圖。

主要執行流程：
1. 從環境變數讀取 training CSV，並核對固定的 training indices。
2. 計算報告使用的 feature-target 與 feature-feature Pearson correlation。
3. 將全部 Pearson 結果印在終端，讓報告中的數值可以重現與核對。
4. 畫 overall、age 的散點圖，以及重要特徵的 Pearson correlation 排名。
5. 將三張獨立圖片輸出至 png/，每張圖各自包含圖例與資料範圍說明。

本程式不讀取 held-out test set，也不訓練任何模型。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TARGET = "value_eur"
RANDOM_SEED = 42
MAX_SCATTER_POINTS = 4_000
DISPLAY_QUANTILE = 0.995
RANKED_FEATURES = [
    "international_reputation",
    "overall",
    "potential",
    "movement_reactions",
    "mentality_composure",
    "attacking_short_passing",
]
REDUNDANCY_PAIRS = [
    ("pace", "movement_sprint_speed"),
    ("pace", "movement_acceleration"),
    ("overall", "movement_reactions"),
    ("overall", "potential"),
]


# 使用 Pillow 內建字型，避免依賴 repository 外的字型檔。
def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# 僅接受 repository 內的固定 training split，並逐列核對 indices。
def load_training_data() -> tuple[pd.DataFrame, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    train_value = os.environ.get("FC26_TRAIN_PATH")
    if not train_value:
        raise RuntimeError(
            "請設定 FC26_TRAIN_PATH，例如："
            "FC26_TRAIN_PATH=data/splits/train.csv python src/eda_feature_report.py"
        )

    train_path = Path(train_value)
    if not train_path.is_absolute():
        train_path = repo_root / train_path
    train_path = train_path.resolve()
    try:
        train_path.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError("FC26_TRAIN_PATH 必須位於 repository 內") from exc

    train = pd.read_csv(train_path, low_memory=False)
    expected_indices = pd.read_csv(repo_root / "data" / "splits" / "train_indices.csv")
    if not train["original_index"].equals(expected_indices["original_index"]):
        raise RuntimeError("輸入資料與固定 training indices 不一致，拒絕執行")
    return train, repo_root


# 集中計算報告引用的 Pearson 數值，避免圖表與表格各自使用手動結果。
def calculate_pearson_results(train: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    target_features = RANKED_FEATURES + ["age"]
    target_correlations = train[target_features].corrwith(train[TARGET])

    rows = []
    for feature in target_features:
        valid = train[[feature, TARGET]].dropna()
        rows.append(
            {
                "section": "feature_target",
                "feature_1": feature,
                "feature_2": TARGET,
                "pearson_r": target_correlations[feature],
                "complete_pair_count": len(valid),
            }
        )

    for feature_1, feature_2 in REDUNDANCY_PAIRS:
        valid = train[[feature_1, feature_2]].dropna()
        rows.append(
            {
                "section": "feature_redundancy",
                "feature_1": feature_1,
                "feature_2": feature_2,
                "pearson_r": valid[feature_1].corr(valid[feature_2]),
                "complete_pair_count": len(valid),
            }
        )

    results = pd.DataFrame(rows)
    ranked_correlations = target_correlations[RANKED_FEATURES].sort_values(ascending=False)
    return ranked_correlations, results


# 畫一個 feature 對 target 的散點圖，並用橘線呈現分箱後的 target median。
def draw_scatter_panel(
    draw: ImageDraw.ImageDraw,
    frame: pd.DataFrame,
    feature: str,
    bounds: tuple[int, int, int, int],
    panel_label: str,
    y_limit_million: float,
) -> None:
    left, top, right, bottom = bounds
    x = pd.to_numeric(frame[feature], errors="coerce")
    y = frame[TARGET] / 1_000_000
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    pearson = x.corr(y)

    x_min, x_max = float(x.min()), float(x.max())

    def x_pixel(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def y_pixel(value: float) -> float:
        value = min(max(value, 0), y_limit_million)
        return bottom - value / y_limit_million * (bottom - top)

    draw.text((left, 24), f"({panel_label}) {feature} vs. value_eur", fill="#102A43", font=get_font(22))
    draw.text((left, 55), f"Pearson r = {pearson:+.3f}", fill="#D3543C", font=get_font(17))
    draw.text((left, top - 22), "value_eur (million EUR)", fill="#333333", font=get_font(13))
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)

    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(len(x), size=min(MAX_SCATTER_POINTS, len(x)), replace=False)
    x_values, y_values = x.to_numpy(), y.to_numpy()
    for index in selected:
        px, py = x_pixel(x_values[index]), y_pixel(y_values[index])
        color = "#D62728" if y_values[index] > y_limit_million else "#7899B8"
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)

    # 離散序位欄位直接依取值分組；連續欄位以分位數分箱，避免單一直線掩蓋曲線。
    trend_frame = pd.DataFrame({"x": x, "y": y})
    if x.nunique() <= 10:
        trend = trend_frame.groupby("x", observed=True).median()
        trend["x"] = trend.index
    else:
        bins = pd.qcut(x, q=18, duplicates="drop")
        trend = trend_frame.groupby(bins, observed=True).median()
    points = [(x_pixel(row.x), y_pixel(row.y)) for row in trend.itertuples()]
    if len(points) >= 2:
        draw.line(points, fill="#F28E2B", width=4)

    for fraction in np.linspace(0, 1, 5):
        xv = x_min + fraction * (x_max - x_min)
        px = x_pixel(xv)
        draw.line((px, bottom, px, bottom + 6), fill="#333333")
        draw.text((px - 18, bottom + 10), f"{xv:.0f}", fill="#333333", font=get_font(13))

        yv = fraction * y_limit_million
        py = y_pixel(yv)
        draw.line((left - 6, py, left, py), fill="#333333")
        draw.text((left - 48, py - 7), f"{yv:.0f}M", fill="#333333", font=get_font(12))

    x_axis_labels = {
        "overall": "overall (rating points)",
        "age": "age (years)",
    }
    x_axis_label = x_axis_labels.get(feature, f"{feature} (unitless)")
    draw.text(
        ((left + right) / 2, bottom + 46),
        x_axis_label,
        fill="#222222",
        font=get_font(15),
        anchor="mm",
    )


# 以水平長條圖呈現重要 features 的 Pearson correlation。
def draw_ranking_panel(
    draw: ImageDraw.ImageDraw,
    correlations: pd.Series,
    bounds: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = bounds
    draw.text((left, 24), "(c) Pearson association ranking", fill="#102A43", font=get_font(22))
    draw.text((left, 72), "Feature", fill="#333333", font=get_font(13))
    maximum = 1.0
    row_height = (bottom - top) / len(correlations)
    label_width = 205
    bar_left = left + label_width
    for index, (feature, value) in enumerate(correlations.items()):
        y = top + index * row_height + 8
        bar_width = max(0, value) / maximum * (right - bar_left - 55)
        draw.text((left, y + 4), feature, fill="#333333", font=get_font(14))
        draw.rectangle((bar_left, y, bar_left + bar_width, y + 24), fill="#CE5A43")
        draw.text((bar_left + bar_width + 7, y + 3), f"{value:+.3f}", fill="#333333", font=get_font(14))
    draw.line((bar_left, bottom, right - 55, bottom), fill="#333333", width=2)
    for value in np.linspace(0, 1, 6):
        x = bar_left + value * (right - bar_left - 55)
        draw.line((x, bottom, x, bottom + 6), fill="#333333", width=1)
        draw.text((x - 10, bottom + 10), f"{value:.1f}", fill="#333333", font=get_font(12))
    draw.text(
        ((bar_left + right - 55) / 2, bottom + 48),
        "Pearson r with value_eur",
        fill="#222222",
        font=get_font(14),
        anchor="mm",
    )


# 每張 scatter 都放自己的圖例與 training-only 說明。
def draw_scatter_legend(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle(
        (70, 650, 930, 735),
        radius=14,
        fill="#F5F7FA",
        outline="#D8DEE9",
        width=2,
    )
    draw.text((95, 668), "Legend", fill="#102A43", font=get_font(16))
    draw.ellipse((190, 674, 198, 682), fill="#7899B8")
    draw.text((208, 668), "Training samples", fill="#333333", font=get_font(14))
    draw.line((355, 678, 395, 678), fill="#F28E2B", width=5)
    draw.text((405, 668), "Binned target median", fill="#333333", font=get_font(14))
    draw.ellipse((595, 673, 605, 683), fill="#D62728")
    draw.text((615, 668), "Above q99.5 limit", fill="#333333", font=get_font(14))
    draw.text(
        (95, 706),
        "Training split only; y display capped at q99.5; Pearson r uses all training rows.",
        fill="#555555",
        font=get_font(13),
    )


# Pearson ranking 使用獨立圖例，避免沿用 scatter 的點與趨勢線說明。
def draw_ranking_legend(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle(
        (70, 650, 930, 735),
        radius=14,
        fill="#F5F7FA",
        outline="#D8DEE9",
        width=2,
    )
    draw.text((95, 668), "Legend", fill="#102A43", font=get_font(16))
    draw.rectangle((190, 673, 225, 685), fill="#CE5A43")
    draw.text((238, 668), "Pearson correlation with value_eur", fill="#333333", font=get_font(14))
    draw.text(
        (95, 706),
        "Training split only; held-out test was not used; Pearson r uses all training rows.",
        fill="#555555",
        font=get_font(13),
    )


# 主流程輸出三張獨立的 training-only 圖。
def main() -> None:
    train, repo_root = load_training_data()
    correlations, pearson_results = calculate_pearson_results(train)
    y_limit_million = float(train[TARGET].quantile(DISPLAY_QUANTILE) / 1_000_000)

    output_dir = repo_root / "png"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for feature, panel_label, filename in [
        ("overall", "a", "feature_target_overall.png"),
        ("age", "b", "feature_target_age.png"),
    ]:
        image = Image.new("RGB", (1000, 760), "white")
        draw = ImageDraw.Draw(image)
        draw_scatter_panel(draw, train, feature, (100, 100, 940, 580), panel_label, y_limit_million)
        draw_scatter_legend(draw)
        output_path = output_dir / filename
        image.save(output_path, format="PNG", optimize=True)
        output_paths.append(output_path)

    ranking_image = Image.new("RGB", (1000, 760), "white")
    ranking_draw = ImageDraw.Draw(ranking_image)
    draw_ranking_panel(ranking_draw, correlations, (80, 100, 930, 560))
    draw_ranking_legend(ranking_draw)
    ranking_path = output_dir / "feature_target_pearson_ranking.png"
    ranking_image.save(ranking_path, format="PNG", optimize=True)
    output_paths.append(ranking_path)

    print(f"Training rows: {len(train):,}")
    print(pearson_results.to_string(index=False))
    for output_path in output_paths:
        print(f"Output: {output_path.relative_to(repo_root)}")
    print("Held-out test accessed: False")


if __name__ == "__main__":
    main()
