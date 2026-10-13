# 第三部分：交付與執行指南

本次交付使用 part 2 的既有主實驗結果。最新分析位於 [Analysis_Summary.md](../results/part3/943351b_v4/Analysis_Summary.md)。這些是供報告整合的分析資料；正式 ACL 報告仍需排版。

說明文字使用繁體中文。專有名詞、feature 名稱、模型名稱與指標沿用組員原文，不額外加中文解釋。原始 dataset、組員資料及程式識別碼保持原文。

## 模型與分析範圍

本文沿用 part 2 的名稱：Ours（OLS, Set C）、Strong（Random Forest）；表格金額使用 €K 或 €M，RMSLE 使用三位小數，原始 CSV 保留完整精度。

主模型是 Set C 的 OLS，共 31 個 features，training target 為 `log1p(value_eur)`。組員先在 OLS、Ridge、Lasso、ElasticNet 中選出最終線性模型；另外訓練 Random Forest 作為 strong baseline。

本分析重建同一個 OLS，核對組員匯出的係數及 test 預測，再計算 HC3 推論與錯誤案例。顯著性比較目前針對 OLS／Random Forest；其他 baseline 的主實驗表現沿用 `results/final_test/test_scores.csv`。這次沒有新增模型、重新選模或重新評估 test 來調參。

## 報告組員先看哪些檔案

| 檔案 | 用途 |
|---|---|
| `Analysis_Summary.md` | 模型差距、係數／H1／H2、EUR 實務意義、三個最大誤差案例與限制 |
| `coefficients_hc3.csv` | 截距及全部 31 個 features 的 β、95% CI、p-value、VIF；同時保留 classical／HC3 |
| `model_comparisons.csv` | OLS／Random Forest 的 MAE、RMSLE、±20% 比例與檢定 |
| `hypothesis_block_tests.csv`、`marginal_effects.csv` | H1／H2 整組檢定，含平方／交互項的有限對比 |
| `practical_effects_eur.csv` | 固定參考情境的 EUR 預測差與 CI |
| `largest_error_cases.csv` | 最大 EUR 誤差案例、實際值、預測值與背景 |
| `case_training_evidence.csv`、`nearest_training_cases.csv`、`shared_pattern_oof.csv` | 案例的 training／OOF 查證資料 |
| `all_residuals.csv`、`largest_log_error_cases.csv` | 完整誤差及另一種排序，供追溯 |
| `ablation_descriptive.csv` | 沿用組員的 CV 消融結果，只作描述性分析 |
| `*.png` | 係數、實際／預測身價與誤差圖 |
| `provenance.json` | 輸入及程式 SHA-256、套件版本、設定與假設；路徑以 repository 根目錄為基準 |

正文與誤差圖統一使用 `signed error = actual − predicted`，負值表示高估。原始 CSV 的 `residual_eur`、`mean_signed_error_ols_eur` 等欄位仍按原定義保留 `predicted − actual`；正文呈現時換成相反方向，原始數值不修改。

以上檔案均位於 `results/part3/943351b_v4/`。正式報告需整理成完整係數表、三案例表與精簡段落，補上表／圖編號、caption、正文引用和單位。

## 建立分析環境

在完整 repository 根目錄使用 Python 3.12。根目錄的 `requirements-analysis.txt` 會先載入組員原本的 `requirements.txt`，再安裝 statsmodels。

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
```

Windows 可將 `.venv-analysis/bin/python` 改為 `.venv-analysis\Scripts\python.exe`。程式需要可用的中文字型：macOS 通常有 PingFang TC；Windows 可使用 Microsoft JhengHei；Linux 可安裝 Noto Sans CJK TC。缺少字型時程式會明確停止，請安裝後再執行。

## 重現分析

輸入資料及主實驗結果已在完整 repository 中，不需要另找模型或下載其他 dataset。使用新的輸出目錄，保留目前 v4：

```bash
.venv-analysis/bin/python src/analyze_current_results.py \
  --output analysis_outputs/part3_reproduced --bootstrap 2000 --seed 42
.venv-analysis/bin/python -m unittest discover -s tests -v
```

分析會核對固定 split、player IDs、target、OOF fold、全部係數與既有 test 預測。重新執行時不呼叫 `run_final_test.py`；既有 test 預測只供核對與分析。

`src/analysis_part3.py` 提供共用功能，另保留通用 OLS／Ridge 分析入口；其通用係數入口只支援 identity target。當前 log1p target 模型請使用上述 `analyze_current_results.py`。

## 解讀時保留的限制

- OLS 的係數尺度是 `log1p(value_eur)`；平方／交互項須共同解讀，不能直接把 β 當作 EUR 差。EUR 實務效果使用明確的固定參考預測，CI 不包含參考預測的不確定性。
- HC3 推論條件於既定模型、特徵及 preprocessing，不涵蓋特徵選擇的不確定性或 club／league 群聚；係數與消融都不能直接證明因果。
- OLS／Random Forest 的 paired tests 將模型視為固定、球員視為獨立。league cluster bootstrap 只作敏感度檢查。CI 是逐項 95% 區間，p-value 另作 Bonferroni 校正。
- ±20% 門檻在 test 評估後選定，必須標為事後分析。RMSLE 不等同每位球員都在 ±20% 內。
- CV folds 的 training sets 重疊，不能當作五個獨立實驗；消融及 OOF 群體查證只作描述性分析。
- 三個案例的原因仍是候選解釋。相似 training 群體及共同模式的樣本數小，改善方向留作未來實驗。
- p-value 若儲存為 0，是數值下溢，不是數學上精確的 0。正式報告以適當的小於號表示。

方法與課程依據：[課程規格](https://docs.google.com/presentation/d/1xmiuV3wiYCCg5r3_tMDF5b5uF7VQweaWSDrZ6DDazG0/edit)、[HC3](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hc3.html)、[McNemar](https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html)。
