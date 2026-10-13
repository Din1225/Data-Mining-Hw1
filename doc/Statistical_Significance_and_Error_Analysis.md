# Statistical_Significance_and_Error_Analysis

**本節接續 part 2 的既有結果，分析最終線性模型 Ours（OLS, Set C）與 strong baseline（Random Forest）。** 固定 training portion 為 14,724 筆，held-out test 為 3,681 筆。
模型使用 31 個 features，target 為 `log1p(value_eur)`；預測以 `expm1` 還原，再截在 training 身價範圍。以下使用組員已匯出的 test 預測，並重建同一個 OLS 核對係數。

組員在 OLS、Ridge、Lasso、ElasticNet 中選出最終線性模型，另外訓練 Random Forest 作為 strong baseline。本節的顯著性比較針對 OLS／Random Forest；其他 baseline 分數見 [test_scores.csv](../results/final_test/test_scores.csv)。模型、features 與 test 預測均沿用 part 2 的既定結果。

## Model Comparison (§7.1)

| Metric | Ours (OLS, Set C) | Strong (Random Forest) | RF 改善幅度 | 95% bootstrap CI | Bonferroni p |
|---|---:|---:|---:|---|---:|
| MAE (€K) | 365 | 155 | 210 | [168, 248] | < 0.001 |
| RMSLE | 0.160 | 0.078 | 0.082 | [0.063, 0.098] | < 0.001 |
| Within ±20% (%) | 85.8 | 98.2 | 12.4 | [11.3, 13.5] | < 0.001 |

**Held-out test 上，RF 的 MAE、RMSLE 與 Within ±20% 均較佳。** MAE／RMSLE 的改善幅度是 Ours − RF；Within ±20% 是 RF − Ours，差值單位為百分點。後者只包含身價 > 0 的 3,665 位球員。
主模型保留 OLS 是為了符合最終 predictor 為線性模型的要求，而不是因為它在 test 上勝過 RF。RF 使用原始前處理後的大型特徵集，OLS 使用 31-feature Set C，模型類型與特徵集同時不同；較彈性的非線性擬合及較多輸入都是候選原因，不能將差距只歸因於單一因素。主實驗與特徵設計的解讀見 [Experiments_and_Ablation](Experiments_and_Ablation.md)。
MAE 使用每位球員 absolute loss 的 paired t-test；RMSLE 的 paired t-test 使用 squared log1p loss，CI 則直接 bootstrap RMSLE 差；Within ±20% 使用 exact McNemar。p-value 對三項指標檢定作 Bonferroni 校正，CI 為逐項區間。
這些推論假設球員獨立、模型固定。league cluster bootstrap 的敏感度結果保留在 [model_comparisons.csv](../results/statistical_analysis/model_comparisons.csv)。±20% 門檻於 test 評估後決定，屬事後分析；它與「每位球員都符合某個 RMSLE 門檻」不是同一判準。

## Final Model Coefficients (§8.1)

以下 β 是 feature 增加 1 training SD 時，`log1p(value_eur)` 尺度的係數。CI 與逐項 p-value 採 HC3 covariance 的 OLS t-test（t 分布近似）；正文 p-value 未作多重校正，31-feature Bonferroni 結果另見 CSV。組員原本的係數表採 classical OLS，兩種方法均保留在 [coefficients_hc3.csv](../results/statistical_analysis/coefficients_hc3.csv)。截距及 Bonferroni 校正結果另見同檔。

| Feature | β | HC3 95% CI | HC3 p | VIF |
|---|---:|---|---:|---:|
| `overall` | 0.9396 | [0.9248, 0.9543] | < 0.001 | 19.8 |
| `potential` | 0.3144 | [0.3024, 0.3264] | < 0.001 | 13.2 |
| `age` | -0.2380 | [-0.2485, -0.2276] | < 0.001 | 11.9 |
| `height_cm` | 0.0019 | [-0.0033, 0.0072] | 0.473 | 3.2 |
| `weight_kg` | -0.0028 | [-0.0078, 0.0022] | 0.269 | 2.9 |
| `weak_foot` | 0.0036 | [0.0002, 0.0071] | 0.040 | 1.2 |
| `skill_moves` | 0.0096 | [0.0038, 0.0155] | 0.001 | 3.5 |
| `international_reputation` | 0.1638 | [0.1443, 0.1834] | < 0.001 | 19.1 |
| `league_level` | 0.0021 | [-0.0005, 0.0046] | 0.112 | 1.1 |
| `contract_years_left` | 0.0052 | [0.0012, 0.0092] | 0.010 | 1.4 |
| `club_tenure_years` | 0.0007 | [-0.0033, 0.0047] | 0.717 | 1.1 |
| `pace` | 0.0026 | [-0.0079, 0.0132] | 0.627 | 11.1 |
| `shooting` | 0.0238 | [0.0125, 0.0352] | < 0.001 | 16.3 |
| `passing` | 0.0406 | [0.0238, 0.0574] | < 0.001 | 38.0 |
| `dribbling` | 0.0063 | [-0.0136, 0.0261] | 0.538 | 51.6 |
| `defending` | -0.0062 | [-0.0151, 0.0027] | 0.172 | 11.1 |
| `physic` | 0.0212 | [0.0089, 0.0335] | < 0.001 | 18.3 |
| `is_goalkeeper` | -0.0041 | [-0.0213, 0.0132] | 0.646 | 33.4 |
| `is_defender` | -0.0138 | [-0.0192, -0.0084] | < 0.001 | 3.1 |
| `is_forward` | 0.0121 | [0.0079, 0.0162] | < 0.001 | 2.6 |
| `left_footed` | -0.0008 | [-0.0038, 0.0022] | 0.607 | 1.1 |
| `real_face` | 0.0030 | [-0.0014, 0.0075] | 0.181 | 1.8 |
| `unique_body_type` | -0.0049 | [-0.0136, 0.0038] | 0.270 | 1.2 |
| `in_national_team` | -0.0014 | [-0.0061, 0.0033] | 0.555 | 1.4 |
| `on_loan` | -0.0011 | [-0.0036, 0.0015] | 0.403 | 1.2 |
| `overall_sq_centered` | 0.1255 | [0.1198, 0.1312] | < 0.001 | 2.2 |
| `age_sq_centered` | -0.2225 | [-0.2318, -0.2133] | < 0.001 | 2.9 |
| `overall_x_reputation` | -0.1531 | [-0.1706, -0.1357] | < 0.001 | 18.6 |
| `is_free_agent` | -1.0237 | [-1.0409, -1.0066] | < 0.001 | 1.2 |
| `age40_outfield` | -0.4200 | [-0.4472, -0.3928] | < 0.001 | 1.1 |
| `top5_league` | 0.0111 | [0.0067, 0.0154] | < 0.001 | 1.6 |

- **VIF：** 12 個 features 的 VIF > 10，與 part 2 一致；共線性會影響個別係數的解讀。
- **方法差異：** HC3 下有 13 個 features 的 p > 0.05，classical OLS 為 12 個。`unique_body_type` 的 p 從約 0.0024 變為 0.270，需在報告中明確標示使用哪種方法。
- **解讀：** 平方項與交互項須共同考慮。HC3 處理異質變異，但 CI／p 仍條件於既定模型與特徵選擇，不涵蓋 club／league 群聚；不顯著不等於沒有作用，係數也不代表因果效果。

### 係數方向與量級

§4 只事先提出 H1／H2；其餘 features 以下作分組描述，沒有補造事前方向假設。每個 feature 的正負、大小與 CI 均列於上表。β 範圍是每 1 training SD 的 log1p(value_eur) 尺度，平方／交互項仍須共同判讀。

| Feature group | β 範圍 | 條件方向與解讀 |
|---|---|---|
| base (11) | [-0.2380, 0.9396] | overall、potential 為較大的正係數；age 為負，但需與平方項合併。height_cm、weight_kg、league_level、club_tenure_years 的 CI 含零；weak_foot、skill_moves、contract_years_left 的 β 較小。international_reputation 的效果受 interaction 改變，另於 H2 討論。 |
| summary_ratings (6) | [-0.0062, 0.0406] | shooting、passing、physic 為正；pace、dribbling、defending 的 CI 含零。這些能力欄位與 overall 的資訊重疊，個別係數是控制其他能力後的條件關聯。 |
| position (3) | [-0.0138, 0.0121] | is_defender 為負、is_forward 為正，is_goalkeeper 的 CI 含零；是相對參考編碼的條件差異，不能推論某種位置普遍較有價值。 |
| profile (5) | [-0.0049, 0.0030] | left_footed、real_face、unique_body_type、in_national_team、on_loan 的 CI 均含零；目前沒有足夠證據支持各個條件係數異於零，不等於證明沒有作用。 |
| nonlinear (2) | [-0.2225, 0.1255] | overall_sq_centered 為正、age_sq_centered 為負，與 §4 的能力凸性及年齡先升後降觀察一致；不能單看一次項推論所有年齡或能力區間。 |
| interaction (1) | [-0.1531, -0.1531] | overall_x_reputation 為負，讓 international_reputation 的條件效果隨 overall 增加而下降；需與兩個主項合併，不代表 reputation 的整體關聯必為負。 |
| domain_rules (3) | [-1.0237, 0.0111] | is_free_agent、age40_outfield 為較大的負係數，與資料中的 €0 規則一致；top5_league 為小的正係數。這些是資料／遊戲規則下的條件關聯，不是現實轉會的因果溢價。 |

### Part 1 假設的對照（H1、H2）

| Hypothesis | 移除的 features | CV RMSLE | 結果 |
|---|---|---|---|
| H1 (`overall`) | `overall`、`overall_sq_centered`、`overall_x_reputation` | 0.171 → 0.283 | 5/5 folds 均變差 |
| H2 (`international_reputation`) | `international_reputation`、`overall_x_reputation` | 0.171 → 0.175 | 5/5 folds 均變差 |

這與 part 2 對預測價值的解讀一致。因為 CV folds 的 training sets 重疊，這裡使用消融結果作描述性證據；不把五個 folds 當成五個獨立實驗。H1／H2 的整組 HC3 檢定也拒絕零係數（均 p < 0.001），完整結果見 [hypothesis_block_tests.csv](../results/statistical_analysis/hypothesis_block_tests.csv)。這是條件關聯的檢定，與預測消融的問題不同。
消融數值直接引用 [組員的 CV 消融結果](../results/linear/ablation/cv_summary.csv)，不採用其中的 fold paired t-test p-value。

`international_reputation` 的條件效果會隨 `overall` 改變，例如 `overall` = 80 時 +1 約對應 +8.5%，85 時約 −3.7%。局部 training 支持與完整有限對比見 [marginal_effects.csv](../results/statistical_analysis/marginal_effects.csv)。

H1／H2 有兩個問題，需分開回答：移除特徵後的 CV 表現支持它們有預測價值；方向預期則不一定在所有條件下成立。H1 的正向一次項及平方項與 §4 觀察一致，實際對比仍依 reputation 改變。H2 的 raw Pearson r = 0.721 為正，但 overall = 85 時的條件對比為 −3.7%，沒有支持「任何能力水準下 reputation 越高都越貴」的解讀。

H2 的符號反轉在模型中直接來自負的 overall_x_reputation，這與 raw correlation 的未控制關聯不同。international_reputation 的 VIF 為 19.1、interaction 為 18.6；共線性、控制其他能力後的關聯及未包含資訊都是候選解釋，不能由係數證明原因。overall = 85、reputation = 3 的局部 training 支持有 172 位，reputation = 1 僅 1 位；有限對比不是同一球員 reputation 改變的真實實驗，稀疏情境須保守解讀。

### EUR 尺度的實務意義

固定 `overall` = 70、`international_reputation` = 1、其他輸入不變，並假設參考預測為 €1M。共同計入平方項／交互項後，得到以下裁切前預測差：

| Feature +1 | 預測差 (€K) | HC3 95% CI (€K) | 與示例 €200K 門檻比較 |
|---|---:|---|---|
| `overall` | 165.6 | [162.8, 168.3] | 低於 |
| `international_reputation` | 379.2 | [325.5, 435.0] | 超過 |

兩個 CI 都排除零，但統計顯著不直接等於實務價值。€200K 是沿用事後 ±20% 的示例門檻，尚未驗證為使用者容忍度；上述情境也不是觀測到的實驗結果或因果效果。CI 固定參考預測，不包含其不確定性，也不代表 EUR 條件平均。€10M 情境與精確金額見 [practical_effects_eur.csv](../results/statistical_analysis/practical_effects_eur.csv)。

## Error Analysis (§8.2)

**依最終 OLS 的最大 absolute EUR error 選出三例。** 下表及圖使用 `signed error = actual − predicted`，負值表示高估；所有金額單位為 €M。

| Player | Actual (€M) | Ours prediction (€M) | Signed error (€M) | RF absolute error (€M) |
|---|---:|---:|---:|---:|---:|
| A. Hakimi | 111.000 | 137.490 | -26.490 | 10.522 |
| R. Gravenberch | 66.500 | 84.729 | -18.229 | 0.076 |
| M. Caicedo | 93.000 | 110.164 | -17.164 | 17.988 |

最大金額誤差不等於超出 decision threshold；依每位球員實際身價計算：

| Player | Absolute relative error (%) | Within ±20% |
|---|---:|---|
| A. Hakimi | 23.86 | 否 |
| R. Gravenberch | 27.41 | 否 |
| M. Caicedo | 18.46 | 是 |

M. Caicedo 仍在 ±20% 內，但他的 absolute EUR error 位居第三，因此依 §8.2 的最大金額誤差選例保留。


### A. Hakimi

- **背景：** `player_id` 235212，26 歲，`overall` 89、`potential` 90、`international_reputation` 4，位置 RB, RM，聯賽 Ligue 1。
- **查證：** 相同主要位置、`overall` ±3、`age` ±3 的 training 球員有 2 位；OOF MAE 為 Ours €0.822M、RF €7.091M，Ours 平均 signed error 為 −€0.822M。
- **可能原因：** 相似 training 球員僅兩位，局部樣本稀少可能使頂尖 RB 的預測不穩定。不過這兩位的 OLS OOF MAE 小於 RF，不能據此推論此類球員普遍被 OLS 高估。

### R. Gravenberch

- **背景：** `player_id` 246104，23 歲，`overall` 85、`potential` 88、`international_reputation` 3，位置 CDM, CM，聯賽 Premier League。
- **查證：** 相同主要位置、`overall` ±3、`age` ±3 的 training 球員有 9 位；OOF MAE 為 Ours €7.940M、RF €3.927M，Ours 平均 signed error 為 −€7.940M。
- **可能原因：** 相似 training 群體的 OLS OOF 平均也為高估，MAE 約為 RF 的兩倍，支持局部校準偏差的候選解釋。RF 對此例接近 actual，可能與更彈性的局部擬合有關。

### M. Caicedo

- **背景：** `player_id` 256079，23 歲，`overall` 87、`potential` 89、`international_reputation` 4，位置 CDM, CM，聯賽 Premier League。
- **查證：** 相同主要位置、`overall` ±3、`age` ±3 的 training 球員有 3 位；OOF MAE 為 Ours €11.369M、RF €7.577M，Ours 平均 signed error 為 −€11.369M。
- **可能原因：** 相似 training 群體的 OLS OOF 平均也為高估，支持局部校準偏差的候選解釋。但 RF 對此例是低估，而且 absolute error 略大，原因仍需考慮高身價映射與未包含的資訊。

相似球員的匹配只控制主要位置、`overall` 與 `age`，沒有涵蓋所有因素。上述原因是候選解釋；完整證據見 [case_training_evidence.csv](../results/statistical_analysis/case_training_evidence.csv) 與 [nearest_training_cases.csv](../results/statistical_analysis/nearest_training_cases.csv)。

### 共同模式與改善方向

三例都是高身價 RB／CDM，且 OLS 高估。Training OOF 中，身價 ≥€50M 的 RB／CDM 只有 6 位，OLS 對 6 位高估；平均 signed error 為 −€7.285M，MAE 為 Ours €7.285M、RF €8.015M。這支持進一步檢查高身價球員的校準，但不能推論所有後衛都高估，或 RF 在所有高身價球員都更好。
這個小群體是在觀察 test 案例後選定，只作 OOF 探索；詳見 [shared_pattern_oof.csv](../results/statistical_analysis/shared_pattern_oof.csv)。

第一個要嘗試的改善是在 Set C 加入 RB／CDM indicator × overall 的交互項，檢查模型能否較好地表達高能力右後衛／防守中場的身價映射。這仍是線性於 β 的模型。後續實驗只使用 training CV，預先固定群體定義（主要位置 RB／CDM、overall ≥ 85），比較整體 RMSLE、MAE 及該群體 OOF signed error／MAE；若局部改善卻使整體表現變差，就不保留。這是待驗證的第一項實驗，不是已完成的改善。構想由本次 test 錯誤診斷提出；後續只用 training CV 比較候選，不再用目前 test 調參或選模型。若要宣稱泛化改善，需另以未使用的資料確認。

![誤差分布](../png/statistical_analysis_residuals.png)

圖中 signed error 的定義與案例表相同；完整逐筆結果見 [all_residuals.csv](../results/statistical_analysis/all_residuals.csv)。原始 CSV 同時保留兩個方向的誤差欄位，閱讀時依欄名判讀。

## Analysis Settings

本次 bootstrap 使用 B=2,000、seed=42。分析使用既有預測 CSV 的兩位小數 EUR；RF 的 RMSLE 與未捨入主表有微小差異，捨入到三位小數後同為 0.078。
來源版本、輸入檔案 SHA、套件版本與設定保留在 [provenance.json](../results/statistical_analysis/provenance.json)。CSV 保留完整精度；正文使用 part 2 的模型名稱、金額單位與小數格式。


## Result Files

| File | 用途 |
|---|---|
| [model_comparisons.csv](../results/statistical_analysis/model_comparisons.csv) | OLS／Random Forest 的三項指標、檢定與 bootstrap CI |
| [coefficients_hc3.csv](../results/statistical_analysis/coefficients_hc3.csv) | 截距及 31 個 features；同時保留 classical／HC3 與 VIF |
| [hypothesis_block_tests.csv](../results/statistical_analysis/hypothesis_block_tests.csv) | H1／H2 整組 HC3 檢定 |
| [marginal_effects.csv](../results/statistical_analysis/marginal_effects.csv) | 含平方／交互項的有限對比與局部 training 支持 |
| [practical_effects_eur.csv](../results/statistical_analysis/practical_effects_eur.csv) | 固定參考情境的 EUR 預測差與 CI |
| [largest_error_cases.csv](../results/statistical_analysis/largest_error_cases.csv) | 按 absolute EUR error 排序的案例與背景 |
| [case_training_evidence.csv](../results/statistical_analysis/case_training_evidence.csv) | 案例的相似 training 群體及 OOF 誤差 |
| [nearest_training_cases.csv](../results/statistical_analysis/nearest_training_cases.csv) | 相似 training 球員明細 |
| [shared_pattern_oof.csv](../results/statistical_analysis/shared_pattern_oof.csv) | 事後選定共同模式的 OOF 查證 |
| [all_residuals.csv](../results/statistical_analysis/all_residuals.csv) | 完整逐筆誤差，供追溯 |
| [provenance.json](../results/statistical_analysis/provenance.json) | 輸入及程式 SHA-256、套件版本、設定與推論假設 |

## Reproduction

沿用 README 的 Python 3.12 執行環境。從完整 repository 根目錄安裝第三部分需求；requirements-analysis.txt 載入組員原本的 requirements.txt，再加入 statsmodels：

```bash
python -m pip install -r requirements-analysis.txt
```

使用新的輸出根目錄，程式會建立 doc/、results/statistical_analysis/ 與 png/；既有輸出不會被覆蓋：

```bash
python src/analyze_current_results.py --output analysis_outputs/part3_reproduced --bootstrap 2000 --seed 42
python -m unittest discover -s tests -v
```

src/analyze_current_results.py 是正式入口；src/analysis_part3.py 提供共用檢查、設計矩陣診斷、來源紀錄及圖表。tests/test_current_results.py 是選擇性程式驗證，不參與正式分析計算。

輸入沿用 data/splits/、results/final_test/、results/linear/main/ 與 results/linear/ablation/。分析會核對 player IDs、固定 split、OOF folds、features、全部係數及既有 test 預測；重建同一個 OLS 是為了係數推論與核對，沒有重新選模或用 test 調參。

程式需要中文字型：macOS 通常有 PingFang TC，Windows 可使用 Microsoft JhengHei，Linux 可使用 Noto Sans CJK TC。缺少字型時會停止，請安裝後再執行。

原始 CSV 的 residual_eur、mean_signed_error_ols_eur 等欄位保留 predicted − actual；正文與圖使用 actual − predicted。閱讀時依欄名判讀。p-value 若儲存為 0，表示數值下溢，不是數學上的精確 0。

正式 ACL 報告仍需整合各部分內容，補上表／圖編號、caption 與正文引用。

方法與課程依據：[課程規格](https://docs.google.com/presentation/d/1xmiuV3wiYCCg5r3_tMDF5b5uF7VQweaWSDrZ6DDazG0/edit)、[HC3](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hc3.html)、[McNemar](https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html)。
