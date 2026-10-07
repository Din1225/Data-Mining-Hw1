# Processed Splits

本目錄由 `data/splits/` 經 `src/preprocess.py` 轉換而來。所有矩陣都是 SciPy CSR sparse matrix，可使用 `scipy.sparse.load_npz()` 讀取。

## Preprocessing 摘要

- 排除 target、薪資、解約金、常數及重複表示欄位。
- 一般數值以 training median 補值；位置限定數值缺值填 0。
- 數值欄位使用 `StandardScaler`。
- 無序類別使用 one-hot；低頻類別合併；多選字串使用 multi-hot。
- `club_joined_date` 轉為 `club_tenure_years`；`ls`–`gk` 位置評分拆成 base 與 modifier。
- 新增 `overall_squared` 與 `overall_x_reputation`，讓線性模型表達非線性及交互作用。
- 保留極端但合理的球員資料。


## `feature_mapping.csv`

此檔用來把 `.npz` 矩陣的每一欄對回原始 feature：

| 欄位 | 用途 |
|---|---|
| `output_index` | `.npz` 矩陣中的 zero-based column index |
| `output_feature` | preprocessing 後的欄位名稱 |
| `source_feature` | 對應的原始資料欄位 |
| `transformer` | 產生該欄位的 preprocessing 分支 |

根目錄 mapping 只搭配根目錄的 `train.npz`／`test.npz`；每個 `fold_k/feature_mapping.csv` 只搭配同一 fold 的 `train.npz`／`validation.npz`。

## 重現

此目錄由 `src/preprocess.py` 產生：

```bash
FC26_TRAIN_PATH=data/splits/train.csv \
FC26_TEST_PATH=data/splits/test.csv \
python src/preprocess.py
```
