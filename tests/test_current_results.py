"""獨立驗證對數目標銜接、資料配對與二元結果比較。"""
from pathlib import Path
import sys
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analysis_part3 import provenance
from analyze_current_results import align_saved_predictions, comparisons, coefficient_inference, effect_gaps, practical_effects, MODELS


class CurrentResultsTests(unittest.TestCase):
    def fixture(self):
        y=np.array([100.,200.,300.,400.,500.,600.,700.,0.])
        ours=y*np.array([1.4,1.4,1.,1.,1.4,1.,1.4,1.])
        rf=y*np.array([1.,1.,1.4,1.,1.,1.,1.4,1.])
        pred=pd.DataFrame({'original_index':range(len(y)),'player_id':np.arange(len(y))+100,
                           'y_true':y,'league_id':[1,1,2,2,3,3,4,4]})
        for m in MODELS: pred[m]=rf if m=='strong_random_forest' else ours
        raw=pred[['original_index','player_id']].assign(value_eur=y)
        return pred,raw

    def test_alignment_is_by_identity_not_row_order(self):
        pred,raw=self.fixture()
        joined=align_saved_predictions(pred.sample(frac=1,random_state=4),raw)
        self.assertEqual(set(joined.original_index),set(raw.original_index))
        np.testing.assert_array_equal(joined.y_true,joined.value_eur)

    def test_raw_label_mismatch_is_rejected(self):
        pred,raw=self.fixture(); raw.loc[0,'value_eur']+=1
        with self.assertRaisesRegex(ValueError,'原始歐元答案不一致'):
            align_saved_predictions(pred,raw)

    def test_player_identity_mismatch_is_rejected(self):
        pred,raw=self.fixture(); raw.loc[0,'player_id']+=999
        with self.assertRaisesRegex(ValueError,'球員識別碼與切分資料不一致'):
            align_saved_predictions(pred,raw)

    def test_exact_mcnemar_matches_statsmodels_and_excludes_zero(self):
        pred,_=self.fixture()
        table=comparisons(pred,200,42)
        binary=table[table.metric.eq('Within20_percent')].iloc[0]
        # 兩者通過2人、僅RF通過3人、僅OLS通過1人、兩者未通過1人；排除零身價。
        self.assertAlmostEqual(binary.p_value,mcnemar([[2,1],[3,1]],exact=True).pvalue)
        self.assertEqual(binary.n,7)
        self.assertEqual(binary.rf_only_pass,3)
        self.assertEqual(binary.ours_only_pass,1)
        self.assertAlmostEqual(binary.gap_rf_better,2/7*100)

    def test_rmsle_gap_is_not_mean_squared_log_loss_gap(self):
        pred,_=self.fixture()
        gap=effect_gaps(pred.y_true.to_numpy(),pred.ours_ols.to_numpy(),pred.strong_random_forest.to_numpy())[1]
        ours=np.sqrt(np.mean(np.square(np.log1p(pred.ours_ols)-np.log1p(pred.y_true))))
        rf=np.sqrt(np.mean(np.square(np.log1p(pred.strong_random_forest)-np.log1p(pred.y_true))))
        self.assertAlmostEqual(gap,ours-rf)

    def test_log_target_coefficients_are_reconstructed_in_log_units(self):
        rng=np.random.default_rng(4)
        X=rng.normal(size=(150,4))
        names=['overall','overall_sq_centered','overall_x_reputation','international_reputation']
        y_log=12+X@np.array([.3,.1,-.05,.2])+rng.normal(0,.05,150)
        train=pd.DataFrame({'value_eur':np.expm1(y_log)})
        fitted=sm.OLS(y_log,sm.add_constant(X)).fit()
        reference=pd.DataFrame({'feature':['(intercept)']+names,'beta_per_sd':fitted.params})
        transform=SimpleNamespace(columns=names,scaler_=SimpleNamespace(scale_=np.ones(4)))
        table,blocks,_,_=coefficient_inference(train,transform,X,reference)
        np.testing.assert_allclose(table.beta_per_sd_log1p_eur,fitted.params)
        self.assertEqual(len(table),5)
        self.assertEqual(len(blocks),2)
        self.assertLess(table.beta_per_sd_log1p_eur.iloc[1],1)

    def test_eur_contrast_respects_log1p_offset_and_fixed_reference(self):
        effects=pd.DataFrame({'overall':[70],'reputation':[1],'feature_incremented':['overall'],
            'delta_log1p_for_plus_one':[np.log(1.2)],'ci95_low_hc3':[np.log(1.1)],'ci95_high_hc3':[np.log(1.3)]})
        table=practical_effects(effects)
        row=table[table.reference_prediction_eur.eq(1e6)].iloc[0]
        self.assertAlmostEqual(row.delta_prediction_eur,200000.2)
        self.assertAlmostEqual(row.ci95_low_eur,100000.1)
        self.assertAlmostEqual(row.ci95_high_eur,300000.3)
        self.assertEqual(row.illustrative_20pct_eur,200000)


    def test_provenance_can_be_resolved_after_repository_is_moved(self):
        root = Path(__file__).resolve().parents[1]
        source = root / 'src/analysis_part3.py'
        with tempfile.TemporaryDirectory() as directory:
            provenance(Path(directory), [source], {})
            record = json.loads((Path(directory) / 'provenance.json').read_text())
            self.assertEqual(record['path_base'], 'repository_root')
            self.assertEqual(record['inputs'][0]['path'], 'src/analysis_part3.py')
            resolved = root / record['inputs'][0]['path']
            self.assertEqual(hashlib.sha256(resolved.read_bytes()).hexdigest(),
                             record['inputs'][0]['sha256'])



if __name__=='__main__':
    unittest.main()
