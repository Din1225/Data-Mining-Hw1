"""以暫存虛構資料檢查配對、數值推論與防呆。"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis_part3 as analysis


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        analysis.demo_inputs(self.root, 42)
        self.pred = self.root / "predictions.csv"
        self.plan = analysis.read_csv(self.root / "comparisons.csv")

    def validated(self, split="test"):
        name = "test_indices.csv" if split == "test" else "train_indices.csv"
        return analysis.validate_predictions(self.pred, split, self.root / name)

    def coef_args(self, kind="ols", name="result"):
        return argparse.Namespace(matrix=self.root / "train.npz", mapping=self.root / "feature_mapping.csv",
            labels=self.root / "training_labels.csv", matrix_indices=self.root / "train_indices.csv",
            spec=self.root / f"{kind}_spec.json", reference=self.root / f"{kind}_reference.csv",
            max_cells=100000, covariance="HC3", bootstrap=200, seed=42,
            output=self.root / name, demo=True)

    def test_prediction_permutation_preserves_paired_result(self):
        original = analysis.compare_predictions(self.validated(), self.plan, "test", 200, 42)
        table = analysis.read_csv(self.pred).sample(frac=1, random_state=21)
        table.to_csv(self.pred, index=False)
        shuffled = analysis.compare_predictions(self.validated(), self.plan, "test", 200, 42)
        pd.testing.assert_frame_equal(original, shuffled)

    def test_paired_result_matches_independently_computed_difference(self):
        data = self.validated()
        row = analysis.compare_predictions(data, self.plan, "test", 200, 42).iloc[0]
        a = data[data.model.eq("Mean")].sort_values("original_index")
        b = data[data.model.eq("Ridge")].sort_values("original_index")
        d = np.abs(a.y_pred.to_numpy()-a.y_true.to_numpy()) - np.abs(b.y_pred.to_numpy()-b.y_true.to_numpy())
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        self.assertAlmostEqual(row.p_paired_t, 2*stats.t.sf(abs(t), len(d)-1), places=12)
        self.assertAlmostEqual(row.mae_reduction_b_eur, d.mean())
        self.assertEqual(row.n, 90)

    def test_missing_sample_is_rejected(self):
        table = analysis.read_csv(self.pred)
        table.drop(table[(table.model.eq("Mean")) & table.split.eq("test")].index[0]).to_csv(self.pred, index=False)
        with self.assertRaisesRegex(ValueError, "切分樣本不完整"):
            self.validated()

    def test_conflicting_targets_are_rejected(self):
        table = analysis.read_csv(self.pred)
        i = table[table.model.eq("Mean") & table.split.eq("test")].index[0]
        table.loc[i, "y_true"] = "123"
        table.to_csv(self.pred, index=False)
        with self.assertRaisesRegex(ValueError, "實際值不一致"):
            self.validated()

    def test_repeated_sample_is_not_an_independent_observation(self):
        table = analysis.read_csv(self.pred)
        pd.concat([table, table.head(1)]).to_csv(self.pred, index=False)
        with self.assertRaisesRegex(ValueError, "樣本重複"):
            self.validated()

    def test_oof_has_no_formal_inference(self):
        result = analysis.compare_predictions(self.validated("oof"), self.plan, "oof", 200, 42)
        self.assertTrue(result[["p_paired_t", "p_bonferroni", "ci_low_bootstrap_eur", "ci_low_t_eur"]].isna().all().all())

    def test_wrong_oof_fold_assignment_is_rejected(self):
        table = analysis.read_csv(self.pred)
        i = table[table.model.eq("Mean") & table.split.eq("oof")].index[0]
        table.loc[i, "fold"] = str(int(table.loc[i, "fold"]) % 5 + 1)
        table.to_csv(self.pred, index=False)
        with self.assertRaisesRegex(ValueError, "折次指派不一致"):
            self.validated("oof")

    def test_one_fold_cannot_pass_as_five_fold_oof(self):
        table = analysis.read_csv(self.pred)
        table.loc[table.split.eq("oof"), "fold"] = "1"
        table.to_csv(self.pred, index=False)
        with self.assertRaisesRegex(ValueError, "全部五折"):
            self.validated("oof")

    def test_identical_errors_do_not_create_a_fake_significance(self):
        data = self.validated()
        for key in data.model_key.unique():
            data.loc[data.model_key.eq(key), "y_pred"] = data.loc[data.model_key.eq(key), "y_true"]
        result = analysis.compare_predictions(data, self.plan, "test", 200, 42)
        self.assertTrue(result.p_paired_t.isna().all())
        self.assertTrue(result.inference_status.str.startswith("degenerate").all())

    def test_negative_prediction_is_retained(self):
        table = analysis.read_csv(self.pred)
        table.loc[0, "y_pred"] = "-500"
        table.to_csv(self.pred, index=False)
        self.assertEqual(self.validated().iloc[0].y_pred, -500)

    def test_ols_coefficients_and_hc3_interval_match_reference(self):
        args = self.coef_args()
        output = analysis.run_coefficients(args)
        table = pd.read_csv(output / "coefficients.csv")
        x, _ = analysis.read_design(args.matrix, args.mapping, args.max_cells)
        y = pd.read_csv(args.labels).value_eur
        reference = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3", use_t=True)
        np.testing.assert_allclose(table.beta_eur_per_x_unit, reference.params.iloc[1:])
        np.testing.assert_allclose(table.ci_low_eur_per_x_unit, reference.conf_int().iloc[1:, 0])
        np.testing.assert_allclose(table.beta_per_sd_eur, reference.params.iloc[1:]*x.std(axis=0))

    def test_full_dummy_coding_blocks_unidentified_ols(self):
        args = self.coef_args()
        x = np.arange(240) % 2
        np.save(self.root / "singular.npy", np.column_stack([x, 1-x]))
        args.matrix = self.root / "singular.npy"
        pd.DataFrame({"output_index":[0,1], "output_feature":["a","b"]}).to_csv(args.mapping, index=False)
        pd.DataFrame({"feature":["__intercept__","a","b"],"beta":[1,1,1]}).to_csv(args.reference,index=False)
        with self.assertRaisesRegex(ValueError, "不可辨識"):
            analysis.run_coefficients(args)
        self.assertFalse((args.output / "coefficients.csv").exists())
        self.assertEqual(json.loads((args.output / "provenance.json").read_text())["settings"]["status"], "blocked_unidentified_ols")

    def test_reference_coefficient_mismatch_is_rejected(self):
        args = self.coef_args()
        ref = pd.read_csv(args.reference)
        ref.loc[0, "beta"] += 1000000
        ref.to_csv(args.reference, index=False)
        with self.assertRaisesRegex(ValueError, "最終係數不一致"):
            analysis.run_coefficients(args)
        self.assertFalse((args.output / "coefficients.csv").exists())

    def test_ridge_bootstrap_reproducible_and_labelled(self):
        first = analysis.run_coefficients(self.coef_args("ridge", "ridge1"))
        second = analysis.run_coefficients(self.coef_args("ridge", "ridge2"))
        np.testing.assert_array_equal(np.load(first / "bootstrap_coefficients.npy"), np.load(second / "bootstrap_coefficients.npy"))
        table = pd.read_csv(first / "coefficients.csv")
        self.assertTrue(table.method.str.contains("近似 Wald").all())

    def test_training_order_mismatch_is_rejected(self):
        args = self.coef_args()
        pd.read_csv(args.labels).iloc[::-1].to_csv(args.labels, index=False)
        with self.assertRaisesRegex(ValueError, "符合matrix_indices順序"):
            analysis.run_coefficients(args)

    def test_unsupported_fit_options_are_rejected(self):
        args = self.coef_args("ridge")
        spec = json.loads(args.spec.read_text())
        spec["positive"] = True
        args.spec.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "不支援的擬合設定"):
            analysis.run_coefficients(args)


if __name__ == "__main__":
    unittest.main()
