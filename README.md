# Data Mining HW1

本專案使用 FC 26 球員資料，以 `value_eur`（球員市場身價）作為 prediction target。

原始資料來源：[FC 26 — FIFA 26 Player Data, Kaggle](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data/data)。

## 目錄結構

```text
HW1/
├── data/                       # 原始資料、固定切分與前處理後資料
│   ├── FC26_20250921.csv       # Kaggle 原始資料
│   ├── splits/                 # 80/20 train-test 與 training portion 的 5-fold 資料
│   └── processed_splits/       # 前處理後的 sparse matrices、indices 與 feature mapping
├── doc/                        # 實作細節的文件 (可參考來做書面報告或上台報告)
├── png/                        # 使用到的圖表和圖片
├── src/                        # 資料檢查、切分、前處理與 EDA 的可重現程式
└── README.md                   # 專案與目錄說明
```

## 目前執行環境
```bash
conda create --name dm-hw1 python=3.12 pip -y
conda activate dm-hw1
python -m pip install -r requirment.txt
```

## 主要程式

| 程式 | 用途 |
|---|---|
| `src/data_audit.py` | 檢查原始資料品質與摘要統計 |
| `src/split_data.py` | 建立固定 80/20 split 與 training portion 的 5-fold 資料 |
| `src/preprocess.py` | 以 training data fit preprocessing，輸出前處理後資料 |
| `src/eda_feature_report.py` | 只使用 training split 重現 EDA 圖表與 Pearson correlation |

## 文件

- `doc/Task_Definition_and_Data.md`：任務定義、資料摘要、資料切分與 preprocessing 說明。
- `doc/Exploratory_Feature_Analysis_and_Feature_Engineering.md`：feature–target 圖表、Pearson correlation、feature redundancy 與後續假設。

