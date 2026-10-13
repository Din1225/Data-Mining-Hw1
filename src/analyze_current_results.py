"""分析第二部分既有結果，維持既定模型與基準預測。

以訓練資料重建既定OLS，核對對數目標係數
並提供HC3推論；核對既有測試預測後進行分析。
"""
from __future__ import annotations

import argparse
from os.path import relpath
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from analysis_part3 import (
    design_diagnostics, output_directory, residual_chart, provenance, require,
)
from linear_features import FEATURE_GROUPS, SetCTransformer

IDS = ["original_index", "player_id"]
OURS, RF = "ours_ols", "strong_random_forest"
MODELS = ["trivial_mean", "simple_ols_overall", RF, OURS]


def align_saved_predictions(pred: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    require(set(IDS + ["y_true"] + MODELS) <= set(pred), "既有預測缺少必要欄位。")
    require(not pred.original_index.duplicated().any() and not pred.player_id.duplicated().any(),
            "每位球員必須恰好出現一次。")
    require(len(pred) == len(raw) and set(pred.original_index) == set(raw.original_index),
            "既有預測未涵蓋完整固定切分。")
    context = raw.drop(columns="short_name") if "short_name" in pred else raw
    result = pred.merge(context, on=IDS, validate="one_to_one")
    require(len(result) == len(raw), "球員識別碼與切分資料不一致。")
    require(np.allclose(result.y_true, result.value_eur, rtol=0, atol=0.005),
            "預測的實際值與原始歐元答案不一致。")
    require(np.isfinite(result[["y_true"] + MODELS].to_numpy()).all(), "既有預測含非有限數值。")
    require(result[["y_true"] + MODELS].ge(0).all().all(), "既有預測含負值。")
    return result


def effect_gaps(y: np.ndarray, ours: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """各項差距正值代表隨機森林較佳；比例只計算正身價球員。"""
    a, b = np.log1p(ours)-np.log1p(y), np.log1p(rf)-np.log1p(y)
    positive = y > 0
    oa, rb = np.abs(ours[positive]/y[positive]-1) <= 0.2, np.abs(rf[positive]/y[positive]-1) <= 0.2
    return np.array([np.abs(ours-y).mean()-np.abs(rf-y).mean(),
                     np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b)),
                     (rb.astype(float)-oa.astype(float)).mean()*100])


def gap_intervals(frame: pd.DataFrame, repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y, a, b = frame.y_true.to_numpy(), frame[OURS].to_numpy(), frame[RF].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array([effect_gaps(y[idx], a[idx], b[idx])
                      for idx in rng.integers(len(y), size=(repeats, len(y)))])
    iid = np.quantile(draws, [0.025, 0.975], axis=0).T
    # 僅作敏感度檢查：整個聯賽重抽樣，保留其全部測試球員。
    keys = frame.league_id.fillna("NO_LEAGUE").astype(str).to_numpy()
    groups = [np.flatnonzero(keys == key) for key in pd.unique(keys)]
    cluster = np.empty((repeats, 3))
    for i in range(repeats):
        idx = np.concatenate([groups[g] for g in rng.integers(len(groups), size=len(groups))])
        cluster[i] = effect_gaps(y[idx], a[idx], b[idx])
    return iid, np.quantile(cluster, [0.025, 0.975], axis=0).T


def comparisons(frame: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    y, a, b = frame.y_true.to_numpy(), frame[OURS].to_numpy(), frame[RF].to_numpy()
    abs_a, abs_b = np.abs(a-y), np.abs(b-y)
    log_a, log_b = np.square(np.log1p(a)-np.log1p(y)), np.square(np.log1p(b)-np.log1p(y))
    positive = y > 0
    oa, rb = np.abs(a[positive]/y[positive]-1) <= 0.2, np.abs(b[positive]/y[positive]-1) <= 0.2
    rf_only, ours_only = int((rb & ~oa).sum()), int((oa & ~rb).sum())
    p_threshold = stats.binomtest(rf_only, rf_only+ours_only, 0.5).pvalue if rf_only+ours_only else 1.0
    iid, cluster = gap_intervals(frame, repeats, seed)
    rows = [
        {"metric":"MAE_EUR", "ours":abs_a.mean(), "rf":abs_b.mean(), "n":len(y),
         "method":"每位球員歐元絕對誤差的成對t檢定", "p_value":stats.ttest_rel(abs_a,abs_b).pvalue},
        {"metric":"RMSLE", "ours":np.sqrt(log_a.mean()), "rf":np.sqrt(log_b.mean()), "n":len(y),
         "method":"平方對數誤差成對t檢定；重抽樣RMSLE差的信賴區間", "p_value":stats.ttest_rel(log_a,log_b).pvalue},
        {"metric":"Within20_percent", "ours":oa.mean()*100, "rf":rb.mean()*100, "n":int(positive.sum()),
         "method":"精確雙尾McNemar檢定（不一致配對的二項檢定）", "p_value":p_threshold,
         "rf_only_pass":rf_only, "ours_only_pass":ours_only},
    ]
    result = pd.DataFrame(rows)
    result["gap_rf_better"] = effect_gaps(y,a,b)
    result["ci95_iid_low"], result["ci95_iid_high"] = iid[:,0], iid[:,1]
    result["ci95_league_low"], result["ci95_league_high"] = cluster[:,0], cluster[:,1]
    result["p_bonferroni_3"] = (result.p_value*3).clip(upper=1)
    return result


def coefficient_inference(train: pd.DataFrame, transformer: SetCTransformer,
                          X: np.ndarray, reference: pd.DataFrame) -> tuple:
    design = sm.add_constant(X, has_constant="add")
    require(np.linalg.matrix_rank(design) == design.shape[1], "最終OLS設計矩陣不滿秩。")
    require(reference.feature.tolist() == ["(intercept)"] + transformer.columns,
            "匯出係數名稱或順序與最終設計不一致。")
    classical = sm.OLS(np.log1p(train.value_eur.to_numpy()), design).fit(use_t=True)
    require(np.allclose(classical.params, reference.beta_per_sd, rtol=5e-6, atol=5e-7),
            "最終係數與匯出模型在六位有效數字精度下不一致。")
    robust = classical.get_robustcov_results(cov_type="HC3", use_t=True)
    _, _, _, rank, condition, vif = design_diagnostics(X)
    interval, ordinary = robust.conf_int(), classical.conf_int()
    table = pd.DataFrame({"feature":["(intercept)"]+transformer.columns,
        "group":["intercept"]+[FEATURE_GROUPS[f] for f in transformer.columns],
        "beta_per_sd_log1p_eur":classical.params, "se_hc3":robust.bse,
        "ci95_low_hc3":interval[:,0], "ci95_high_hc3":interval[:,1], "p_hc3":robust.pvalues,
        "se_classical":classical.bse, "p_classical":classical.pvalues,
        "ci95_low_classical":ordinary[:,0], "ci95_high_classical":ordinary[:,1],
        "vif":np.r_[np.nan,vif], "raw_feature_sd":np.r_[np.nan,transformer.scaler_.scale_]})
    table["p_hc3_bonferroni_31"] = np.nan
    table.loc[1:,"p_hc3_bonferroni_31"] = (table.loc[1:,"p_hc3"]*len(transformer.columns)).clip(upper=1)
    # 整組條件關聯檢定，不能取代預測消融檢定。
    hypotheses = {"H1_overall":["overall","overall_sq_centered","overall_x_reputation"],
                  "H2_reputation":["international_reputation","overall_x_reputation"]}
    tests = []
    for label, features in hypotheses.items():
        restriction = np.zeros((len(features), design.shape[1]))
        for i, name in enumerate(features): restriction[i,1+transformer.columns.index(name)] = 1
        result = robust.f_test(restriction)
        tests.append({"hypothesis":label, "features":";".join(features),
                      "f_hc3":float(result.fvalue), "p_hc3":float(result.pvalue),
                      "df_num":result.df_num, "df_denom":result.df_denom})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni_2"] = (tests.p_hc3*2).clip(upper=1)
    return table, tests, robust, {"rank":rank,"condition_number":condition,
        "max_coefficient_rounding_difference":float(np.max(np.abs(classical.params-reference.beta_per_sd.to_numpy())))}


def marginal_effects(train: pd.DataFrame, transformer: SetCTransformer, robust) -> pd.DataFrame:
    """共同計入平方與交互項的加一有限對比，未套用裁切。"""
    rows = []
    names, scales = transformer.columns, transformer.scaler_.scale_
    center = transformer.builder_
    critical = stats.t.ppf(0.975, robust.df_resid)
    for overall in [60,70,80,85,90]:
        for reputation in [1,3,4]:
            for changed in ['overall','international_reputation']:
                contrast = np.zeros(len(names)+1)
                def assign(name, difference):
                    i=names.index(name); contrast[i+1]=difference/scales[i]
                if changed=='overall':
                    assign('overall',1)
                    assign('overall_sq_centered',2*(overall-center.overall_mean_)+1)
                    assign('overall_x_reputation',reputation-center.reputation_mean_)
                else:
                    assign('international_reputation',1)
                    assign('overall_x_reputation',overall-center.overall_mean_)
                delta=float(contrast @ robust.params)
                se=float(np.sqrt(contrast @ robust.cov_params() @ contrast))
                support=int((train.overall.sub(overall).abs().le(3)
                             & train.international_reputation.eq(reputation)).sum())
                rows.append({'overall':overall,'reputation':reputation,'feature_incremented':changed,
                    'delta_log1p_for_plus_one':delta,'ci95_low_hc3':delta-critical*se,
                    'ci95_high_hc3':delta+critical*se,'pct_change_value_plus_one':np.expm1(delta)*100,
                    'pct_ci_low':np.expm1(delta-critical*se)*100,
                    'pct_ci_high':np.expm1(delta+critical*se)*100,'n_train_local_support':support,
                    'interpretation':'裁切前的條件有限對比；不是歐元條件平均或因果效果。'})
    return pd.DataFrame(rows)


def practical_effects(effects: pd.DataFrame) -> pd.DataFrame:
    """將聯合對數對比換算為明確固定參考情境的歐元差。"""
    selected = effects[effects.overall.eq(70) & effects.reputation.eq(1)]
    rows = []
    for row in selected.itertuples():
        for value in [1e6,1e7]:
            rows.append({'feature_incremented':row.feature_incremented,'overall':70,'reputation':1,
                'reference_prediction_eur':value,
                'delta_prediction_eur':(value+1)*np.expm1(row.delta_log1p_for_plus_one),
                'ci95_low_eur':(value+1)*np.expm1(row.ci95_low_hc3),
                'ci95_high_eur':(value+1)*np.expm1(row.ci95_high_hc3),
                'illustrative_20pct_eur':0.2*value,
                'interpretation':'固定的假設參考預測；裁切前條件對比，不是因果效果或歐元條件平均。信賴區間不含基準預測不確定性。'})
    return pd.DataFrame(rows)


def shared_pattern_evidence(train: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    frame = train.merge(oof[IDS+[OURS,RF]],on=IDS,validate='one_to_one')
    position = frame.player_positions.str.split(',').str[0].str.strip()
    masks = {'value_ge_50M':frame.value_eur.ge(50e6),
             'value_ge_50M_RB_or_CDM':frame.value_eur.ge(50e6)&position.isin(['RB','CDM']),
             'all_RB':position.eq('RB'),'all_CDM':position.eq('CDM')}
    rows=[]
    for label, mask in masks.items():
        subset=frame.loc[mask]; error=subset[OURS]-subset.value_eur
        rows.append({'group':label,'n_train':len(subset),'n_ols_overpredicted':int(error.gt(0).sum()),
            'mean_signed_error_ols_eur':error.mean(),'mae_ols_eur':error.abs().mean(),
            'mae_rf_eur':(subset[RF]-subset.value_eur).abs().mean(),
            'scope':'觀察測試案例後選定的折外探索群體，不提供正式p值。'})
    return pd.DataFrame(rows)


def case_evidence(frame: pd.DataFrame, train: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    frame = frame.copy()
    frame["residual_eur"] = frame[OURS]-frame.y_true
    frame["actual_minus_predicted_eur"] = -frame.residual_eur
    frame["absolute_error_eur"] = np.abs(frame.residual_eur)
    frame["relative_error"] = np.where(frame.y_true>0,frame.absolute_error_eur/frame.y_true,np.nan)
    frame["abs_log1p_error"] = np.abs(np.log1p(frame[OURS])-np.log1p(frame.y_true))
    frame["rf_absolute_error_eur"] = np.abs(frame[RF]-frame.y_true)
    frame["primary_position"] = frame.player_positions.str.split(',').str[0].str.strip()
    frame["at_training_max_clip"] = np.isclose(frame[OURS],train.value_eur.max(),rtol=0,atol=0.005)
    frame = frame.sort_values(["absolute_error_eur","original_index"],ascending=[False,True])
    training = train.copy()
    training["primary_position"] = training.player_positions.str.split(',').str[0].str.strip()
    training = training.merge(oof[IDS+[OURS,RF]],on=IDS,validate="one_to_one")
    evidence, neighbors = [], []
    # 以明確的描述性匹配規則查證；沒有建立新模型。
    for _, case in frame.head(3).iterrows():
        masks = {"same_primary_position":training.primary_position.eq(case.primary_position),
                 "similar_ability_age_position":training.primary_position.eq(case.primary_position)
                      & training.overall.sub(case.overall).abs().le(3) & training.age.sub(case.age).abs().le(3),
                 "reputation_at_least_case":training.international_reputation.ge(case.international_reputation),
                 "value_at_least_50M":training.value_eur.ge(50e6)}
        for label, mask in masks.items():
            subset = training.loc[mask]
            y = subset.value_eur.to_numpy()
            evidence.append({"case_player_id":case.player_id,"group":label,"n_train":len(subset),
                "training_median_value_eur":np.median(y) if len(y) else np.nan,
                "oof_mae_ours_eur":np.abs(subset[OURS]-y).mean(),
                "oof_mae_rf_eur":np.abs(subset[RF]-y).mean(),
                "oof_mean_signed_error_ours_eur":(subset[OURS]-y).mean()})
        subset = training.loc[masks["similar_ability_age_position"]].copy()
        subset["matching_distance"] = np.square(subset.overall-case.overall)+np.square(subset.age-case.age)
        subset = subset.sort_values(["matching_distance","original_index"]).head(5)
        subset.insert(0,"case_player_id",case.player_id)
        neighbors.append(subset[["case_player_id",*IDS,"short_name","age","overall","potential",
            "international_reputation","primary_position","value_eur",OURS,RF,"matching_distance"]])
    return frame,pd.DataFrame(evidence),pd.concat(neighbors,ignore_index=True)


def summary_report(output: Path, comparison: pd.DataFrame, table: pd.DataFrame,
                   cases: pd.DataFrame, evidence: pd.DataFrame, notes: dict,
                   ablation: pd.DataFrame, blocks: pd.DataFrame,
                   practical: pd.DataFrame, patterns: pd.DataFrame) -> None:
    """沿用組員的模型名稱、單位與精度，整理第三部分正文及證據連結。"""
    def p_text(value: float) -> str:
        return '< 0.001' if value < 0.001 else f'{value:.3f}'

    def signed_money(value: float) -> str:
        return f"{'−' if value < 0 else ''}€{abs(value)/1e6:.3f}M"

    positive_n = int(comparison.loc[comparison.metric.eq('Within20_percent'),'n'].iloc[0])
    guide_path = relpath(Path(__file__).resolve().parents[1]/'doc/Part3_Analysis_Guide.md', output.resolve()).replace('\\','/')
    ablation_path = relpath(Path(__file__).resolve().parents[1]/'results/linear/ablation/cv_summary.csv', output.resolve()).replace('\\','/')
    lines = ['# 統計顯著性與錯誤案例分析', '',
        f"**本節接續 part 2 的既有結果，分析最終線性模型 Ours（OLS, Set C）與 strong baseline（Random Forest）。** 固定 training portion 為 {notes['n_train']:,} 筆，held-out test 為 {notes['n_test']:,} 筆。",
        '模型使用 31 個 features，target 為 `log1p(value_eur)`；預測以 `expm1` 還原，再截在 training 身價範圍。以下使用組員已匯出的 test 預測，並重建同一個 OLS 核對係數。', '',
        '## 模型比較（§7.1）', '',
        '| Metric | Ours (OLS, Set C) | Strong (Random Forest) | RF 改善幅度 | 95% bootstrap CI | Bonferroni p |',
        '|---|---:|---:|---:|---|---:|']
    settings = {'MAE_EUR':('MAE (€K)',1000,0), 'RMSLE':('RMSLE',1,3),
                'Within20_percent':('Within ±20% (%)',1,1)}
    for row in comparison.itertuples():
        label, scale, digits = settings[row.metric]
        fmt = lambda value: f'{value/scale:,.{digits}f}'
        lines.append(f'| {label} | {fmt(row.ours)} | {fmt(row.rf)} | {fmt(row.gap_rf_better)} | [{fmt(row.ci95_iid_low)}, {fmt(row.ci95_iid_high)}] | {p_text(row.p_bonferroni_3)} |')
    lines += ['',
        f'**Held-out test 上，RF 的 MAE、RMSLE 與 Within ±20% 均較佳。** MAE／RMSLE 的改善幅度是 Ours − RF；Within ±20% 是 RF − Ours，差值單位為百分點。後者只包含身價 > 0 的 {positive_n:,} 位球員。',
        'MAE 使用每位球員 absolute loss 的 paired t-test；RMSLE 的 paired t-test 使用 squared log1p loss，CI 則直接 bootstrap RMSLE 差；Within ±20% 使用 exact McNemar。p-value 對三項指標檢定作 Bonferroni 校正，CI 為逐項區間。'.replace('三项','三項'),
        '這些推論假設球員獨立、模型固定。league cluster bootstrap 的敏感度結果保留在 [model_comparisons.csv](model_comparisons.csv)。±20% 門檻於 test 評估後決定，屬事後分析；它與「每位球員都符合某個 RMSLE 門檻」不是同一判準。', '',
        '## 最終模型係數（§8.1）', '',
        '以下 β 是 feature 增加 1 training SD 時，`log1p(value_eur)` 尺度的係數。CI 與 p-value 採 HC3；組員原本的係數表採 classical OLS，兩種方法均保留在 [coefficients_hc3.csv](coefficients_hc3.csv)。截距及 Bonferroni 校正結果另見同檔。', '',
        '| Feature | β | HC3 95% CI | HC3 p | VIF |',
        '|---|---:|---|---:|---:|']
    for row in table.iloc[1:].itertuples():
        lines.append(f'| `{row.feature}` | {row.beta_per_sd_log1p_eur:.4f} | [{row.ci95_low_hc3:.4f}, {row.ci95_high_hc3:.4f}] | {p_text(row.p_hc3)} | {row.vif:.1f} |')
    lines += ['',
        f"- **VIF：** {int(table.vif.gt(10).sum())} 個 features 的 VIF > 10，與 part 2 一致；共線性會影響個別係數的解讀。",
        f"- **方法差異：** HC3 下有 {int(table.iloc[1:].p_hc3.gt(0.05).sum())} 個 features 的 p > 0.05，classical OLS 為 {int(table.iloc[1:].p_classical.gt(0.05).sum())} 個。`unique_body_type` 的 p 從約 0.0024 變為 0.270，需在報告中明確標示使用哪種方法。",
        '- **解讀：** 平方項與交互項須共同考慮。HC3 處理異質變異，但 CI／p 仍條件於既定模型與特徵選擇，不涵蓋 club／league 群聚；不顯著不等於沒有作用，係數也不代表因果效果。', '',
        '### Part 1 假設的對照（H1、H2）', '',
        '| Hypothesis | 移除的 features | CV RMSLE | 結果 |',
        '|---|---|---|---|']
    baseline = ablation.loc[ablation.model.eq('set_c'),'rmsle_mean'].iloc[0]
    for label, model, removed in [
        ('H1 (`overall`)','set_c_minus_overall','`overall`、`overall_sq_centered`、`overall_x_reputation`'),
        ('H2 (`international_reputation`)','set_c_minus_reputation','`international_reputation`、`overall_x_reputation`')]:
        value = ablation.loc[ablation.model.eq(model),'rmsle_mean'].iloc[0]
        lines.append(f'| {label} | {removed} | {baseline:.3f} → {value:.3f} | 5/5 folds 均變差 |')
    lines += ['',
        '這與 part 2 對預測價值的解讀一致。因為 CV folds 的 training sets 重疊，這裡使用消融結果作描述性證據；不把五個 folds 當成五個獨立實驗。H1／H2 的整組 HC3 檢定也拒絕零係數（均 p < 0.001），完整結果見 [hypothesis_block_tests.csv](hypothesis_block_tests.csv)。這是條件關聯的檢定，與預測消融的問題不同。',
        f'消融數值直接引用 [組員的 CV 消融結果]({ablation_path})，不採用其中的 fold paired t-test p-value。', '',
        '`international_reputation` 的條件效果會隨 `overall` 改變，例如 `overall` = 80 時 +1 約對應 +8.5%，85 時約 −3.7%。局部 training 支持與完整有限對比見 [marginal_effects.csv](marginal_effects.csv)。', '',
        '### EUR 尺度的實務意義', '',
        '固定 `overall` = 70、`international_reputation` = 1、其他輸入不變，並假設參考預測為 €1M。共同計入平方項／交互項後，得到以下裁切前預測差：', '',
        '| Feature +1 | 預測差 (€K) | HC3 95% CI (€K) | 與示例 €200K 門檻比較 |',
        '|---|---:|---|---|']
    for row in practical[practical.reference_prediction_eur.eq(1e6)].itertuples():
        conclusion = '低於' if abs(row.delta_prediction_eur)<row.illustrative_20pct_eur else '超過'
        lines.append(f'| `{row.feature_incremented}` | {row.delta_prediction_eur/1000:.1f} | [{row.ci95_low_eur/1000:.1f}, {row.ci95_high_eur/1000:.1f}] | {conclusion} |')
    lines += ['',
        '兩個 CI 都排除零，但統計顯著不直接等於實務價值。€200K 是沿用事後 ±20% 的示例門檻，尚未驗證為使用者容忍度；上述情境也不是觀測到的實驗結果或因果效果。CI 固定參考預測，不包含其不確定性，也不代表 EUR 條件平均。€10M 情境與精確金額見 [practical_effects_eur.csv](practical_effects_eur.csv)。', '',
        '## 錯誤案例分析（§8.2）', '',
        '**依最終 OLS 的最大 absolute EUR error 選出三例。** 下表及圖使用 `signed error = actual − predicted`，負值表示高估；所有金額單位為 €M。', '',
        '| Player | Actual (€M) | Ours prediction (€M) | Signed error (€M) | RF absolute error (€M) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in cases.head(3).itertuples():
        lines.append(f'| {row.short_name} | {row.y_true/1e6:.3f} | {getattr(row,OURS)/1e6:.3f} | {row.actual_minus_predicted_eur/1e6:.3f} | {row.rf_absolute_error_eur/1e6:.3f} |')
    for row in cases.head(3).itertuples():
        matched = evidence[evidence.case_player_id.eq(row.player_id)&evidence.group.eq('similar_ability_age_position')].iloc[0]
        if int(matched.n_train)<3:
            reason = '相似 training 球員僅兩位，局部樣本稀少可能使頂尖 RB 的預測不穩定。不過這兩位的 OLS OOF MAE 小於 RF，不能據此推論此類球員普遍被 OLS 高估。'
        elif row.rf_absolute_error_eur>row.absolute_error_eur:
            reason = '相似 training 群體的 OLS OOF 平均也為高估，支持局部校準偏差的候選解釋。但 RF 對此例是低估，而且 absolute error 略大，原因仍需考慮高身價映射與未包含的資訊。'
        else:
            reason = '相似 training 群體的 OLS OOF 平均也為高估，MAE 約為 RF 的兩倍，支持局部校準偏差的候選解釋。RF 對此例接近 actual，可能與更彈性的局部擬合有關。'
        lines += ['',f'### {row.short_name}', '',
            f'- **背景：** `player_id` {int(row.player_id)}，{int(row.age)} 歲，`overall` {int(row.overall)}、`potential` {int(row.potential)}、`international_reputation` {int(row.international_reputation)}，位置 {row.player_positions}，聯賽 {row.league_name}。',
            f'- **查證：** 相同主要位置、`overall` ±3、`age` ±3 的 training 球員有 {int(matched.n_train)} 位；OOF MAE 為 Ours €{matched.oof_mae_ours_eur/1e6:.3f}M、RF €{matched.oof_mae_rf_eur/1e6:.3f}M，Ours 平均 signed error 為 {signed_money(-matched.oof_mean_signed_error_ours_eur)}。',
            f'- **可能原因：** {reason}']
    pattern = patterns[patterns.group.eq('value_ge_50M_RB_or_CDM')].iloc[0]
    lines += ['',
        '相似球員的匹配只控制主要位置、`overall` 與 `age`，沒有涵蓋所有因素。上述原因是候選解釋；完整證據見 [case_training_evidence.csv](case_training_evidence.csv) 與 [nearest_training_cases.csv](nearest_training_cases.csv)。', '',
        '### 共同模式與改善方向', '',
        f"三例都是高身價 RB／CDM，且 OLS 高估。Training OOF 中，身價 ≥€50M 的 RB／CDM 只有 {int(pattern.n_train)} 位，OLS 對 {int(pattern.n_ols_overpredicted)} 位高估；平均 signed error 為 {signed_money(-pattern.mean_signed_error_ols_eur)}，MAE 為 Ours €{pattern.mae_ols_eur/1e6:.3f}M、RF €{pattern.mae_rf_eur/1e6:.3f}M。這支持進一步檢查高身價球員的校準，但不能推論所有後衛都高估，或 RF 在所有高身價球員都更好。",
        '這個小群體是在觀察 test 案例後選定，只作 OOF 探索；詳見 [shared_pattern_oof.csv](shared_pattern_oof.csv)。後續可在新的實驗中檢查位置 × 能力、`age` × `potential` 的交互項或分段效果，不用目前的 test 調參。', '',
        '![誤差分布](residual_vs_actual.png)', '',
        '圖中 signed error 的定義與案例表相同；完整逐筆結果見 [all_residuals.csv](all_residuals.csv)。原始 CSV 同時保留兩個方向的誤差欄位，閱讀時依欄名判讀。', '',
        '## 執行與補充資料', '',
        f"本次 bootstrap 使用 B={notes['bootstrap']:,}、seed={notes['seed']}。分析使用既有預測 CSV 的兩位小數 EUR；RF 的 RMSLE 與未捨入主表有微小差異，捨入到三位小數後同為 0.078。",
        f'套件設定、方法依據與重現方式見 [執行指南]({guide_path})；來源版本與檔案 SHA 保留在 [provenance.json](provenance.json)。CSV 保留完整精度，本文只統一閱讀格式。', '']
    (output/'Analysis_Summary.md').write_text('\n'.join(lines),encoding='utf-8')

def run(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parents[1]
    require(args.bootstrap >= 200, "重抽樣至少200次。")
    paths = {"train":root/'data/splits/train.csv',"test":root/'data/splits/test.csv',
        "test_predictions":root/'results/final_test/test_predictions.csv',
        "oof":root/'results/linear/main/oof_predictions.csv',
        "model":root/'results/final_test/final_model.json',
        "reference":root/'results/final_test/coefficients.csv'}
    paths['ablation']=root/'results/linear/ablation/cv_summary.csv'
    model = json.loads(paths['model'].read_text())
    require(model['variant']=='ols' and model['params']=={}, "此銜接程式僅適用既定OLS。")
    train,test = pd.read_csv(paths['train'],low_memory=False),pd.read_csv(paths['test'],low_memory=False)
    pred,oof = pd.read_csv(paths['test_predictions']),pd.read_csv(paths['oof'])
    frame = align_saved_predictions(pred,test)
    align_saved_predictions(oof,train)
    require(not set(train.original_index)&set(test.original_index), "訓練與測試資料重疊。")
    for fold in range(1,6):
        fold_ids=pd.read_csv(root/f'data/splits/train_5-fold_cv/fold_{fold}/validation_indices.csv')
        require(set(oof.loc[oof.fold.eq(fold),'original_index'])==set(fold_ids.original_index), "折外預測的折次指派不一致。")
        paths[f'fold_{fold}']=root/f'data/splits/train_5-fold_cv/fold_{fold}/validation_indices.csv'
    transformer = SetCTransformer(); X = transformer.fit_transform(train)
    require(transformer.columns==model['features'], "最終特徵清單已變動。")
    table,blocks,robust,notes = coefficient_inference(train,transformer,X,pd.read_csv(paths['reference']))
    effects = marginal_effects(train,transformer,robust)
    practical = practical_effects(effects)
    patterns = shared_pattern_evidence(train,oof)
    ablation=pd.read_csv(paths['ablation'])
    reconstructed = np.clip(np.expm1(robust.predict(sm.add_constant(transformer.transform(test),has_constant='add'))),0,train.value_eur.max())
    check = test[IDS].copy(); check['reconstructed']=reconstructed
    check = pred.merge(check,on=IDS,validate='one_to_one')
    require(np.allclose(check[OURS],check.reconstructed,rtol=0,atol=0.0051), "既有最終預測與重建模型不一致。")
    differences = comparisons(frame,args.bootstrap,args.seed)
    cases,evidence,neighbors = case_evidence(frame,train,oof)
    output = output_directory(args.output)
    differences.to_csv(output/'model_comparisons.csv',index=False)
    table.to_csv(output/'coefficients_hc3.csv',index=False)
    blocks.to_csv(output/'hypothesis_block_tests.csv',index=False)
    effects.to_csv(output/'marginal_effects.csv',index=False)
    practical.to_csv(output/'practical_effects_eur.csv',index=False)
    patterns.to_csv(output/'shared_pattern_oof.csv',index=False)
    cols = [*IDS,'short_name','age','overall','potential','international_reputation','player_positions',
            'league_id','league_name','y_true',OURS,RF,'residual_eur','actual_minus_predicted_eur',
            'absolute_error_eur','relative_error','abs_log1p_error','rf_absolute_error_eur','at_training_max_clip']
    cases[cols].to_csv(output/'all_residuals.csv',index=False)
    cases.head(10)[cols].to_csv(output/'largest_error_cases.csv',index=False)
    evidence.to_csv(output/'case_training_evidence.csv',index=False)
    neighbors.to_csv(output/'nearest_training_cases.csv',index=False)
    plot = cases.copy(); plot['y_pred']=plot[OURS]
    residual_chart(plot,output)
    notes.update(upstream_commit=subprocess.check_output(['git','rev-parse','origin/main'],cwd=root,text=True).strip(),
        n_train=len(train),n_test=len(test),bootstrap=args.bootstrap,seed=args.seed,
        max_prediction_rounding_difference=float(np.max(np.abs(check[OURS]-check.reconstructed))),
        target_transform='log1p',coefficient_unit='每一訓練標準差的身價對數尺度；不是歐元差。',
        inference='特徵與模型視為固定。採獨立球員檢定；聯賽重抽樣只作敏感度檢查。',
        posthoc='20%門檻於測試評估後選定，不宣稱是事先指定門檻的推論。',
        ci='95%逐項百分位數區間；三項比較的p值採Bonferroni校正。',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    code_paths=[Path(__file__),root/'src/linear_features.py',root/'src/experiment_utils.py',root/'src/analysis_part3.py']
    provenance(output,list(paths.values())+code_paths,notes)
    summary_report(output,differences,table,cases,evidence,notes,ablation,blocks,practical,patterns)
    print(f'已核對全部既有預測，最大歐元捨入差為 {notes["max_prediction_rounding_difference"]:.6f}.')
    print(f'輸出資料夾： {output.resolve()}')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    run(parser.parse_args())
