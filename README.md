# Data Mining HW1

本專案使用 FC 26 球員資料，以 `value_eur`（球員市場身價）作為 prediction target。

原始資料來源：[FC 26 — FIFA 26 Player Data, Kaggle](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data/data)。

## 目錄結構

```text
HW1/
├── data/                       # 原始資料、固定切分與前處理後資料
│   ├── FC26_20250921.csv       # Kaggle 原始資料
│   ├── splits/                 # 80/20 train-test 與 training portion 的 5-fold 資料
│   └── processed_splits/       # 前處理後的 train-test 與 training portion 的 5-fold 資料 與 feature mapping
├── doc/                        # 實作細節的文件 (可參考來做書面報告或上台報告)
├── png/                        # 使用到的圖表和圖片
├── results/                    # 實驗結果
│   ├── tuning/                 # strong baseline（Random Forest）超參數搜尋
│   ├── linear/main/            # 線性變體選擇、VIF、主實驗表（5-fold CV）
│   ├── linear/ablation/        # 消融實驗與分群指標（5-fold CV）
│   ├── final_test/             # held-out test 唯一一次評估、逐筆預測與係數表
│   ├── threshold/              # decision threshold（±20%）的指標與 test 每位球員的誤差
│   └── report/                 # 報告用表格（Markdown、LaTeX）與圖（PDF）
├── src/                        # 資料處理、EDA 與實驗的可重現程式
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
| `src/run_tuning.py` | strong baseline（Random Forest）的超參數搜尋 |
| `src/run_linear_models.py` | 比較 OLS／Ridge／Lasso／ElasticNet、計算 VIF，與三層 baseline 產生主實驗表 |
| `src/run_linear_ablation.py` | 特徵集消融（Set A／B／C、移除特徵群組、H1／H2）與分群指標 |
| `src/run_final_test.py` | 以完整 training portion 重新訓練，在 held-out test 上評估唯一一次 |
| `src/threshold_metrics.py` | 以 decision threshold（±20% 相對誤差）評估 CV 與 test 的每個結果 |
| `src/make_report_tables.py` | 由結果檔產生報告用表格 |
| `src/make_report_figures.py` | 由結果檔產生報告用圖（tuning curve、消融長條圖） |

共用模組：`src/experiment_utils.py`（fold 讀取、指標、CV 流程）、`src/linear_features.py`（Set B／Set C 特徵）、`src/model_registry.py`（模型與搜尋範圍）。

實驗程式依序執行（需先完成前處理）：

```bash
export FC26_SPLIT_DIR=data/splits FC26_PROCESSED_DIR=data/processed_splits
python src/run_tuning.py
python src/run_linear_models.py
python src/run_linear_ablation.py
python src/run_final_test.py
python src/threshold_metrics.py
python src/make_report_tables.py
python src/make_report_figures.py
```

`run_final_test.py` 在 `results/final_test/test_scores.csv` 已存在時會拒絕執行，確保 test 只使用一次。

## 文件

- `doc/Task_Definition_and_Data.md`：任務定義、資料摘要、資料切分與 preprocessing 說明。
- `doc/Exploratory_Feature_Analysis_and_Feature_Engineering.md`：feature–target 圖表、Pearson correlation、feature redundancy 與後續假設。
- `doc/Experiments_and_Ablation.md`：主實驗、超參數選擇、消融實驗、分群分析與最終模型係數的解讀。
- `doc/Decision_Threshold.md`：decision threshold（±20% 相對誤差）的定義、選擇理由、比較結果，以及給 part 3 與報告撰寫的使用說明。

