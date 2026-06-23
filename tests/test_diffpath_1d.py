import unittest
import tempfile

import numpy as np
import torch
import torch.nn as nn

from diffpath_1d import (
    apply_diffpath_6d_calibrator,
    apply_diffpath_6d_gmm_calibrator,
    backproject_windows_to_time,
    backproject_feature_windows_to_time,
    empirical_cdf,
    fit_diffpath_6d_calibrator,
    fit_diffpath_6d_gmm_calibrator,
    fit_diffpath_calibrator,
    seed_for_save,
)
from main_model import CSDI_base
from tools.evaluate_pathB_diffpath_f1 import (
    best_fusion,
    best_point_adjusted_threshold,
    load_score_file,
    point_adjust,
    precision_recall_f1,
)


class StepOnlyEpsilon(nn.Module):
    def forward(
        self,
        x,
        _side_info,
        diffusion_step,
        _strategy_type,
    ):
        batch, _, features, length = x.shape
        values = diffusion_step.float().reshape(batch, 1, 1)
        return values.expand(batch, features, length)


class DummyDiffPathModel:
    resolve_diffpath_timesteps = CSDI_base.resolve_diffpath_timesteps
    set_input_to_diffmodel = CSDI_base.set_input_to_diffmodel
    compute_diffpath_1d = CSDI_base.compute_diffpath_1d

    def __init__(self, num_steps=50):
        self.num_steps = int(num_steps)
        self.device = torch.device("cpu")
        self.is_unconditional = True
        beta = torch.linspace(0.0001, 0.02, self.num_steps)
        alpha = torch.cumprod(1.0 - beta, dim=0)
        self.alpha_torch = alpha.unsqueeze(1).unsqueeze(1)
        self.diffmodel = StepOnlyEpsilon()


class DiffPathFormulaTests(unittest.TestCase):
    def test_default_timesteps_match_expected_spacing(self):
        model = DummyDiffPathModel(num_steps=50)
        self.assertEqual(
            model.resolve_diffpath_timesteps(10),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        )

    def test_non_ddim_spacing_is_rejected(self):
        model = DummyDiffPathModel(num_steps=50)
        with self.assertRaises(ValueError):
            model.resolve_diffpath_timesteps(12)

    def test_diffpath_statistic_uses_step_and_feature_sum(self):
        model = DummyDiffPathModel(num_steps=50)
        observed = torch.zeros(2, 3, 4)
        cond_mask = torch.zeros_like(observed)
        side_info = torch.zeros(2, 1, 3, 4)
        strategy_type = torch.zeros(2, dtype=torch.long)

        result = model.compute_diffpath_1d(
            observed,
            cond_mask,
            side_info,
            strategy_type,
            num_path_steps=10,
            return_moments=True,
        )
        timesteps = result["diffpath_timesteps"]
        differences = np.diff(timesteps).astype(np.float64) * 10.0
        expected = np.sqrt(3.0 * np.sum(differences ** 2))
        actual = result["diffpath_1d_statistic"]
        self.assertEqual(list(actual.shape), [2, 4])
        self.assertTrue(
            torch.allclose(
                actual,
                torch.full_like(actual, float(expected)),
                atol=1e-5,
            )
        )
        self.assertEqual(
            list(result["epsilon_moment_sums"].shape),
            [2, 3, 4],
        )
        self.assertEqual(
            list(result["derivative_moment_sums"].shape),
            [2, 3, 4],
        )
        self.assertEqual(
            list(result["diffpath_6d_moment_sums"].shape),
            [2, 6, 4],
        )

    def test_diffpath_is_deterministic(self):
        model = DummyDiffPathModel(num_steps=50)
        observed = torch.randn(2, 3, 4)
        cond_mask = torch.zeros_like(observed)
        side_info = torch.zeros(2, 1, 3, 4)
        strategy_type = torch.zeros(2, dtype=torch.long)
        first = model.compute_diffpath_1d(
            observed,
            cond_mask,
            side_info,
            strategy_type,
        )["diffpath_1d_statistic"]
        second = model.compute_diffpath_1d(
            observed,
            cond_mask,
            side_info,
            strategy_type,
        )["diffpath_1d_statistic"]
        self.assertTrue(torch.equal(first, second))


class CalibrationAndWindowTests(unittest.TestCase):
    def test_each_save_gets_an_independent_seed(self):
        self.assertEqual(seed_for_save(7, "save0"), 7)
        self.assertEqual(seed_for_save(7, "save1"), 8)
        self.assertEqual(seed_for_save(7, "save2"), 9)

    def test_window_backprojection_matches_existing_layout(self):
        windows = torch.arange(3 * 8).reshape(3, 8).float()
        scores, indices = backproject_windows_to_time(windows, split=4)
        expected_scores = np.concatenate(
            [
                windows[0, :2].numpy(),
                windows[:, 2:6].reshape(-1).numpy(),
            ]
        )
        expected_indices = np.concatenate(
            [
                np.arange(0, 2),
                np.arange(2, 6),
                np.arange(6, 10),
                np.arange(10, 14),
            ]
        )
        np.testing.assert_array_equal(scores, expected_scores)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_feature_window_backprojection_matches_existing_layout(self):
        windows = torch.arange(3 * 2 * 8).reshape(3, 2, 8).float()
        features, indices = backproject_feature_windows_to_time(
            windows,
            split=4,
        )
        expected_features = np.concatenate(
            [
                windows[0, :, :2].transpose(0, 1).numpy(),
                windows[:, :, 2:6].permute(0, 2, 1).reshape(-1, 2).numpy(),
            ],
            axis=0,
        )
        expected_indices = np.concatenate(
            [
                np.arange(0, 2),
                np.arange(2, 6),
                np.arange(6, 10),
                np.arange(10, 14),
            ]
        )
        np.testing.assert_array_equal(features, expected_features)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_calibrator_outputs_finite_kde_parameters(self):
        normal_recon = np.linspace(0.0, 1.0, 50)
        normal_diffpath = np.linspace(1.0, 3.0, 50)
        calibrator = fit_diffpath_calibrator(
            normal_recon,
            normal_diffpath,
            timesteps=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            bandwidths=[0.1, 0.2],
            seed=7,
        )
        self.assertIn(float(calibrator["kde_bandwidth"]), [0.1, 0.2])
        self.assertTrue(
            np.isfinite(
                calibrator[
                    "normal_diffpath_raw_score_sorted"
                ]
            ).all()
        )
        cdf = empirical_cdf(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.0, 0.5, 1.0]),
        )
        np.testing.assert_allclose(
            cdf,
            np.asarray([1.0 / 3.0, 2.0 / 3.0, 1.0]),
        )

    def test_6d_calibrator_outputs_finite_scores(self):
        normal_recon = np.linspace(0.0, 1.0, 60)
        normal_6d = np.column_stack(
            [
                np.linspace(0.0, 1.0, 60),
                np.linspace(1.0, 2.0, 60),
                np.sin(np.linspace(0.0, 1.0, 60)),
                np.cos(np.linspace(0.0, 1.0, 60)),
                np.linspace(2.0, 1.0, 60),
                np.linspace(1.0, 3.0, 60) ** 2,
            ]
        )
        calibrator = fit_diffpath_6d_calibrator(
            normal_recon,
            normal_6d,
            timesteps=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            bandwidths=[0.5, 1.0],
            seed=11,
        )
        raw_score, recon_cdf, path_cdf = apply_diffpath_6d_calibrator(
            calibrator,
            np.asarray([0.0, 0.5, 1.0]),
            normal_6d[[0, 30, 59]],
        )
        self.assertEqual(list(calibrator["kde_fit_values"].shape), [60, 6])
        self.assertTrue(np.isfinite(raw_score).all())
        self.assertTrue(np.isfinite(recon_cdf).all())
        self.assertTrue(np.isfinite(path_cdf).all())
        self.assertTrue(np.all((path_cdf >= 0.0) & (path_cdf <= 1.0)))

    def test_6d_gmm_calibrator_outputs_finite_scores(self):
        rng = np.random.default_rng(123)
        normal_recon = np.linspace(0.0, 1.0, 80)
        first = rng.normal(loc=-1.0, scale=0.2, size=(40, 6))
        second = rng.normal(loc=1.0, scale=0.3, size=(40, 6))
        normal_6d = np.concatenate([first, second], axis=0)
        calibrator = fit_diffpath_6d_gmm_calibrator(
            normal_recon,
            normal_6d,
            timesteps=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            n_components=[2, 4],
            covariance_types=["diag", "full"],
            seed=11,
        )
        raw_score, recon_cdf, path_cdf = apply_diffpath_6d_gmm_calibrator(
            calibrator,
            np.asarray([0.0, 0.5, 1.0]),
            normal_6d[[0, 40, 79]],
        )
        self.assertIn(int(calibrator["gmm_n_components"]), [2, 4])
        self.assertIn(
            str(np.asarray(calibrator["gmm_covariance_type"]).item()),
            ["diag", "full"],
        )
        self.assertEqual(list(calibrator["standard_scaler_mean"].shape), [6])
        self.assertTrue(np.isfinite(raw_score).all())
        self.assertTrue(np.isfinite(recon_cdf).all())
        self.assertTrue(np.isfinite(path_cdf).all())
        self.assertTrue(np.all((path_cdf >= 0.0) & (path_cdf <= 1.0)))


class F1SearchTests(unittest.TestCase):
    def brute_force_best(self, labels, scores):
        best = None
        for threshold in np.unique(scores):
            raw_prediction = (scores >= threshold).astype(np.int64)
            adjusted = point_adjust(raw_prediction, labels)
            precision, recall, f1 = precision_recall_f1(
                adjusted,
                labels,
            )
            candidate = (f1, precision, float(threshold))
            if best is None or candidate > best[0]:
                best = (
                    candidate,
                    {
                        "threshold": float(threshold),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    },
                )
        return best[1]

    def test_fast_threshold_search_matches_brute_force(self):
        labels = np.asarray([0, 1, 1, 1, 0, 0, 1, 1, 0])
        scores = np.asarray(
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.2, 0.6, 0.5, 0.7]
        )
        expected = self.brute_force_best(labels, scores)
        actual = best_point_adjusted_threshold(labels, scores)
        self.assertAlmostEqual(actual["threshold"], expected["threshold"])
        self.assertAlmostEqual(actual["precision"], expected["precision"])
        self.assertAlmostEqual(actual["recall"], expected["recall"])
        self.assertAlmostEqual(actual["f1"], expected["f1"])

    def test_fusion_alpha_boundaries_use_original_scores(self):
        labels = np.asarray([0, 1, 1, 0, 0, 1])
        recon = np.asarray([0.1, 0.9, 0.8, 0.2, 0.3, 0.7])
        diffpath = np.asarray([0.8, 0.1, 0.2, 0.7, 0.6, 0.3])

        alpha_zero = best_fusion(
            labels,
            recon,
            diffpath,
            [0.0],
        )
        alpha_one = best_fusion(
            labels,
            recon,
            diffpath,
            [1.0],
        )
        np.testing.assert_allclose(alpha_zero["scores"], diffpath)
        np.testing.assert_allclose(alpha_one["scores"], recon)

    def test_diffpath_f1_loader_separates_raw_and_cdf_scores(self):
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez_compressed(
                f.name,
                labels_aligned=np.asarray([0, 1, 0]),
                valid_indices=np.asarray([0, 1, 2]),
                final_recon_score=np.asarray([10.0, 20.0, 30.0]),
                final_recon_score_sum_abs=np.asarray(
                    [10.0, 20.0, 30.0]
                ),
                final_recon_score_max_abs=np.asarray(
                    [7.0, 8.0, 9.0]
                ),
                diffpath_1d_raw_score=np.asarray([1.0, 2.0, 3.0]),
                diffpath_6d_gmm_raw_score=np.asarray([4.0, 5.0, 6.0]),
                recon_cdf=np.asarray([0.1, 0.2, 0.3]),
                recon_sum_abs_cdf=np.asarray([0.1, 0.2, 0.3]),
                recon_max_abs_cdf=np.asarray([0.9, 0.8, 0.7]),
                diffpath_1d_cdf=np.asarray([0.4, 0.5, 0.6]),
                diffpath_6d_gmm_cdf=np.asarray([0.7, 0.8, 0.9]),
            )
            loaded = load_score_file(f.name)

        np.testing.assert_allclose(
            loaded["recon_raw"],
            np.asarray([10.0, 20.0, 30.0]),
        )
        np.testing.assert_allclose(
            loaded["recon_sum_raw"],
            np.asarray([10.0, 20.0, 30.0]),
        )
        np.testing.assert_allclose(
            loaded["recon_max_raw"],
            np.asarray([7.0, 8.0, 9.0]),
        )
        np.testing.assert_allclose(
            loaded["diffpath_6d_gmm_raw"],
            np.asarray([4.0, 5.0, 6.0]),
        )
        np.testing.assert_allclose(
            loaded["recon_cdf"],
            np.asarray([0.1, 0.2, 0.3]),
        )
        np.testing.assert_allclose(
            loaded["recon_sum_cdf"],
            np.asarray([0.1, 0.2, 0.3]),
        )
        np.testing.assert_allclose(
            loaded["recon_max_cdf"],
            np.asarray([0.9, 0.8, 0.7]),
        )
        np.testing.assert_allclose(
            loaded["diffpath_6d_gmm_cdf"],
            np.asarray([0.7, 0.8, 0.9]),
        )


if __name__ == "__main__":
    unittest.main()
