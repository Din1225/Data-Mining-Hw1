# Decision_Threshold

> **決定（2026-10-09）：身價 > 0 的球員，預測與實際身價的相對誤差在 ±20% 以內，視為「不會改變使用者的決策」。**
> 也就是 |預測 − 實際| ≤ 20% × 實際身價。對應到整體指標 RMSLE，門檻是 ln(1.2) ≈ 0.182。

作業要求定義一個 decision threshold（誤差大到多少，使用者的決策就會改變），並把每個結果與門檻比較。本文件說明門檻的定義、選擇理由、目前的比較結果，以及接下來的人要怎麼使用。

## 定義

| Item | Definition |
|---|---|
| 使用者與決策 | 球探在轉會前，用預測身價判斷一位球員的報價是否合理 |
| 門檻 | \|預測 − 實際\| ≤ 20% × 實際身價（單位為歐元，依每位球員的實際身價換算） |
| 主要比較 | **在門檻內的球員比例**：身價 > 0 的球員中，預測落在門檻內的比例（越高越好） |
| 整體比較 | RMSLE ≤ ln(1.2) ≈ 0.182，等價於「典型相對誤差」exp(RMSLE) − 1 ≤ 20%；報告表格中以 † 標示 |
| 身價為 0 的球員 | 相對誤差沒有定義（分母為 0），不納入比例；另外回報人數與預測值的最大值 |
| 程式中的定義 | `src/experiment_utils.py` 的 `DECISION_THRESHOLD = 0.20`，所有程式都從這裡讀取 |

## 為什麼選 ±20%

1. **使用情境。** 我們把模型定位成球探的估價參考：轉會前看一位球員的報價，用預測身價判斷合不合理。我們假設預測偏差在兩成以內時，球探的判斷不會改變；偏差超過兩成，就可能多付錢，或放棄一筆其實合理的交易。

2. **用相對誤差，不用固定金額。** 身價從 €0 到 €1.745 億都有，training 的中位數是 €100 萬。同樣 €50 萬的誤差，對 €1 億的球員只有 0.5%，對 €50 萬的球員卻是 100%，所以任何固定金額的門檻，都會對球星太嚴格、對一般球員太寬鬆。我們的 target 是 log1p(身價)、主要指標是 RMSLE，兩者衡量的本來就是相對誤差，用相對門檻也和它們一致。

3. **選 20%，不選 10% 或 30%。**
   - ±10% 幾乎要求預測等於實際身價，對一個輔助判斷的參考值來說過於嚴格。
   - ±30% 以上時，€1,000 萬的球員預測成 €700 萬或 €1,300 萬都算合格，範圍寬到無法支持出價決策。
   - ±20% 介於兩者之間，也是好溝通的整數。對中位數 €100 萬的球員，可接受範圍是 €80 萬–€120 萬，比 training 身價的四分位距（€47.5 萬–€210 萬）窄得多，所以這個門檻仍然能區分不同價位的球員。

4. **RMSLE 的門檻為什麼是 0.182。** 高估 20% 的 log 誤差是 ln(1.2) = 0.182，低估 20% 是 |ln(0.8)| = 0.223，取較嚴格的 0.182。身價遠大於 €1 時，log1p 與 log 幾乎相同，所以 RMSLE 可以直接和 0.182 比，或換算成典型相對誤差 exp(RMSLE) − 1 再和 20% 比。

5. **身價為 0 的球員。** 這些是自由球員和 40 歲以上的非門將（遊戲規則），相對誤差沒有定義。它們不納入「在門檻內的比例」，但仍包含在整體 RMSLE、MAE 等指標中，並另外回報預測值。例如 CV 中最終模型對 93 位 €0 球員的預測最高為 €68。

## 門檻的選擇不影響結論

門檻是在 held-out test 評估之後（2026-10-09）才決定的。為了確認這個選擇沒有讓結論偏向任何模型，我們也計算了其他候選門檻下，在門檻內的球員比例（5-fold CV，身價 > 0 的球員）：

| Model | ±10% | **±20%（採用）** | ±30% | ±50% |
|---|---:|---:|---:|---:|
| Trivial (training mean) | 3.8% | 7.4% | 10.6% | 16.8% |
| Simple (OLS on overall) | 13.1% | 26.3% | 40.2% | 77.4% |
| Strong (Random Forest) | 92.4% | 97.8% | 99.0% | 99.7% |
| Ours (OLS, Set C) | 54.6% | 86.2% | 95.8% | 98.7% |

在每個候選門檻下，排名都是 RF > Ours > Simple > Trivial，所以「在門檻內的比例」這個主要比較不受門檻選擇影響。

整體 RMSLE 的 † 標記則會隨門檻改變。例如改用 ±10% 時，門檻變成 ln(1.1) = 0.095，最終模型的 CV RMSLE 0.171 就不再達到門檻。因此報告中的主要比較請用「在門檻內的比例」，† 只作為輔助。

## 每個結果與門檻的比較

### 主實驗與 test

| Model | CV：在門檻內 | CV RMSLE（典型相對誤差） | Test：在門檻內 | Test RMSLE（典型相對誤差） |
|---|---:|---:|---:|---:|
| Trivial (training mean) | 7.4% | 2.002（640%） | 8.2% | 1.867（547%） |
| Simple (OLS on overall) | 26.3% | 1.293（264%） | 25.3% | 1.109（203%） |
| Strong (Random Forest) | **97.8%** | 0.269（30.8%） | **98.2%** | 0.078 †（8.1%） |
| Ours (OLS, Set C) | 86.2% | 0.171 †（18.7%） | 85.8% | 0.160 †（17.3%） |

- 最終模型對約 86% 身價 > 0 的球員，預測誤差在 ±20% 以內，CV 與 test 一致；RF 約 98%。
- RF 的 CV RMSLE（0.269）沒有達到門檻，是被少數 40 歲以上的 €0 球員拉高；只看身價 > 0 的球員，RF 的 RMSLE 是 0.098（典型誤差 10.3%），在門檻內。原因見 `doc/Experiments_and_Ablation.md`。
- Simple 和 Trivial 在 CV 與 test 都遠超過門檻。

### 消融實驗（CV）

| Feature set | 在門檻內 | RMSLE |
|---|---:|---:|
| A: processed raw columns | 61.6% | 0.507 |
| B: top-31 by \|Pearson r\| with value | 54.3% | 1.143 |
| B': top-31 by \|Pearson r\| with log1p(value) | 56.9% | 0.613 |
| **C: ours (final)** | **86.2%** | 0.171 † |
| C − nonlinear | 72.2% | 0.228 |
| C − interaction | 85.5% | 0.174 † |
| C − domain rules | 60.7% | 1.041 |
| C − overall and derived (H1) | 62.2% | 0.283 |
| C − reputation and derived (H2) | 85.2% | 0.175 † |
| C + club/league mean overall | 86.2% | 0.171 † |

- 只有 Set C 和三個只改動少量特徵的變體（移除 interaction、移除 reputation、加入 aggregates）達到 RMSLE 門檻。
- 移除 interaction 或 reputation 後，RMSLE 仍在門檻內，但在門檻內的比例在 5 個 fold 都下降（−0.7 與 −1.0 個百分點，paired t-test p = 0.02 與 p < 0.001）。
- 移除 domain rules 不只影響 €0 球員：身價 > 0 的球員在門檻內的比例也從 86.2% 降到 60.7%。下降幾乎全部來自兩個 €0 規則（`is_free_agent`、`age40_outfield`），而不是 `top5_league`：用同樣的 CV 設定檢查，只移除這兩個特徵時為 59.8%，只移除 `top5_league` 時為 86.0%。少了 €0 規則，OLS 必須用一般特徵去擬合 93 位 log 身價為 0 的極端值，連帶扭曲了其他球員的係數。

### 分群（CV）

RF 在每一個分群（身價、聯賽等級、位置、年齡）在門檻內的比例都比較高。最終模型最弱的三群是 ≥30 歲（73.7%）、門將（74.1%）與身價 ≥ €1,000 萬（74.2%），最好的是 ≤21 歲（92.1%）。完整數字在 `results/threshold/cv_slices.csv` 與 `results/report/slices.tex`。

## 給接下來做的人

### 檔案在哪裡

| File | Content |
|---|---|
| `src/experiment_utils.py`（`DECISION_THRESHOLD`） | 門檻數值，唯一的定義處 |
| `src/threshold_metrics.py` | 計算所有門檻指標；只讀取既有的預測，不重新訓練 |
| `results/threshold/cv_main_summary.csv` | 主實驗各模型（CV）在門檻內的比例（mean ± std）、中位數相對誤差 |
| `results/threshold/cv_ablation_summary.csv` | 消融各特徵集，含與 Set C 的配對差異與 paired t-test |
| `results/threshold/cv_slices.csv` | 各分群的結果 |
| `results/threshold/cv_sensitivity.csv` | ±10%／±30%／±50% 的比較 |
| `results/threshold/test_summary.csv` | test 各模型在門檻內的比例 |
| `results/threshold/test_player_errors.csv` | **test 每位球員**在各模型下的相對誤差（`*_rel_error`）與是否在門檻內（`*_within_threshold`）；身價為 0 的球員兩欄皆為空值；可用 `original_index` 對回 `results/final_test/test_predictions.csv` |
| `results/report/*.tex`、`results/report/tables.md` | 報告用表格，已含「Within ±20% (%)」欄與 † 標記 |

### Part 3：顯著性檢定

「是否在門檻內」是每位球員的成對二元結果（同一位球員、兩個模型），適合用 McNemar test 比較兩個模型在門檻內的比例：

```python
import pandas as pd
from scipy import stats

errors = pd.read_csv("results/threshold/test_player_errors.csv")
pair = errors.dropna(subset=["ours_ols_within_threshold"])        # 只留身價 > 0 的球員
rf = pair["strong_random_forest_within_threshold"].astype(bool)
ours = pair["ours_ols_within_threshold"].astype(bool)
b, c = int((rf & ~ours).sum()), int((~rf & ours).sum())           # 只有其中一個模型在門檻內的球員數
print(b, c, stats.binomtest(b, b + c, 0.5).pvalue)                 # exact McNemar test
```

把欄位換成其他模型，就能比較任意兩個模型。CV 的逐 fold 結果在 `cv_main_fold_scores.csv`，可以做 5-fold paired t-test。

### Part 3：錯誤案例

作業要求至少 3 個 test 上誤差最大的案例。可以從最終模型超出門檻最多的球員挑：

```python
preds = pd.read_csv("results/final_test/test_predictions.csv")[["original_index", "ours_ols"]]
worst = (errors.dropna(subset=["ours_ols_rel_error"])
         .merge(preds, on="original_index")
         .sort_values("ours_ols_rel_error", ascending=False)
         [["short_name", "y_true", "ours_ols", "ours_ols_rel_error"]]
         .head(10))
```

身價為 0 的 16 位球員沒有相對誤差，如果要分析，請改用歐元誤差或 log 誤差，並說明它們是另一類情況。

### 寫報告的人

- **Setup（§6）：** 寫出門檻的定義與理由。可以改寫下面這段英文：

  > We frame the model as a pricing aid for a scout who checks whether a player's asking price is reasonable before a transfer. We assume that an error within ±20% of the true market value would not change this judgment, whereas a larger error could lead the scout to overpay or to walk away from a fair deal. We therefore define the decision threshold as a relative error of 20% (|ŷ − y| ≤ 0.2y, in euros relative to each player's value) and report, for every model, the share of players with a positive value whose prediction falls within it. We use a relative rather than an absolute threshold because values span €0–€174.5M: a €0.5M error is negligible for a €100M player but equals the entire value of a €0.5M player. On the log scale, the threshold corresponds to RMSLE ≤ ln 1.2 ≈ 0.182. Players valued at €0 (free agents and outfield players aged 40 or older) have no defined relative error and are reported separately.

- **Results（§7）：** `results/report/` 的主表、消融表、分群表與 test 表都已經有「Within ±20% (%)」欄與 † 標記，直接引用即可。`threshold_sensitivity.tex` 可以放在正文或附錄，用來說明結論不取決於 20% 這個選擇。
- **時間點：** 門檻是在 test 評估之後才決定的，建議在報告中照實說明，並引用敏感度分析。

### 注意事項

- 門檻已經定案，請不要自行更改。若組內決定修改，改 `src/experiment_utils.py` 的 `DECISION_THRESHOLD`，重新執行下面兩個程式，並更新本文件。
- 計算門檻指標不需要、也不可以重跑 `src/run_final_test.py`；`threshold_metrics.py` 只讀取已經存在的 test 預測。

```bash
export FC26_SPLIT_DIR=data/splits FC26_PROCESSED_DIR=data/processed_splits
python src/threshold_metrics.py     # → results/threshold/
python src/make_report_tables.py    # → results/report/
```
