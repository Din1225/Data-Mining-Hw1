# Task Definition

| Item | Definition |
|---|---|
| Target | `value_eur`：球員估計市場身價，單位為 EUR；觀測範圍 €0–€174,500,000，合理值域為非負值 |
| One row | 一位球員在 FC 26 單一資料快照中的一筆紀錄 |
| Inputs | 球員基本資料、球會／聯賽／國家隊與合約資訊、能力及位置評分 |
| Known at prediction time | 使用同一資料快照中當下已知的球員非經濟資料與評分，並沒有使用未來資訊 |

# Data Summary

| Item | Summary |
|---|---|
| Source | [FC 26 — FIFA 26 Player Data, Kaggle](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data/data) |
| Collection method | 從 Kaggle 下載公開的 FC 26 球員資料 CSV |
| Period | 單次資料快照；FC update date 為 2025-09-19 |
| Size | 18,405 rows × 110 columns；每列代表一位球員 |
| Target count | 18,405 |
| Target mean | €2,931,633.36 |
| Target standard deviation | €7,947,786.78 |
| Target minimum | €0 |
| Target 25% | €475,000 |
| Target median | €1,000,000 |
| Target 75% | €2,100,000 |
| Target maximum | €174,500,000 |

# Data Split

完整資料先使用 `train_test_split` 切成 training portion（14,724 筆，80%）與 held-out test（3,681 筆，20%），再只對 training portion 使用 5-fold `KFold`。切分設定為 `shuffle=True`、`random_state=42`，所有模型共用相同 indices。

Held-out test 不參與 EDA、feature decision、preprocessing fitting、model selection 或 hyperparameter tuning，只保留在最後完成時進行一次最終評估。

![Held-out test and 5-fold cross-validation diagram](../data/splits/split_diagram.png)

# Preprocessing and the Reason for Each Step

| Step | Processing | Reason |
|---|---|---|
| Missing values | • 一般數值：用 training median 補值<br>• 位置限定數值：缺值填 0，表示該位置不適用<br>• 類別欄位：改成有明確意思的 missing category<br>• `work_rate`：直接移除 | • Median 比平均值不容易被少數極端球員拉動<br>• 位置限定欄位的空值填 0 表示不適用<br>• 類別缺值不使用眾數補值，而是補成具有明確意義的獨立類別(像是__MISSING__)，保留原本缺值所代表的狀態。<br>• `work_rate` 全部都是空值，留下來也沒有資訊 |
| Outliers | • 保留合理範圍內的極端值<br>• 不刪除也不 clipping | 高身價或高能力值通常是真實的頂級球員；只因為數值很大就刪掉，反而會少掉重要案例 |
| Scaling | • 數值欄位使用 `StandardScaler`<br>• 只在 training data 或各 CV training fold 上 fit | 讓不同單位的數值落在相近尺度，模型比較好處理；只用 training fit 是為了不要偷看到 test／validation 的資料分布 |
| Encoding | • 無序類別：one-hot encoding<br>• 低頻類別：依 training vocabulary 合併<br>• 多選欄位：拆開後做 multi-hot encoding<br>• 位置評分：解析成數值<br>• 聯賽：使用 `league_id` 編碼，不使用 `league_name` | • 有些類別沒有大小順序，不適合直接編成 1、2、3；例如 `preferred_foot` 會拆成 `preferred_foot_Left` 和 `preferred_foot_Right` 欄位<br>• 合併太少見的類別可以避免產生太多零散欄位；例如 `nationality_name = Namibia` 在 training data 只出現 1 次，因此會和其他出現少於 10 次的國籍合併到 `nationality_name_infrequent_sklearn` 欄位<br>• 多選欄位要讓一位球員可以同時對應多個標籤；例如 `player_positions = "ST, RW"` 會拆成 `player_positions_ST = 1` 和 `player_positions_RW = 1`<br>• 位置評分要先轉成模型能使用的數值；例如 `86+3` 會拆成基礎分數 86 與修正值 3<br>• `league_name` 不是唯一的：51 個聯賽只有 42 個名稱，例如 `Premier League` 同時代表英超與烏克蘭超、`Super League` 同時代表希臘、瑞士、中國與印度的聯賽；用名稱編碼會把身價水準差很多的聯賽混成同一欄 |
