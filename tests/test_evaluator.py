"""
tests/test_evaluator.py
=======================
Unit tests untuk modul ModelEvaluator.
Memverifikasi kebenaran numerik formula MAE, MAPE, RMSE,
serta logika interpretasi kualitas model.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


# ---------------------------------------------------------------------------
# Test: calculate_mae
# ---------------------------------------------------------------------------

class TestCalculateMAE:
    def test_perfect_prediction(self, evaluator):
        y = np.array([10, 20, 30, 40])
        assert evaluator.calculate_mae(y, y) == 0.0

    def test_known_value(self, evaluator):
        # |10-8| + |20-25| + |30-27| = 2 + 5 + 3 = 10; mean = 10/3 ≈ 3.333
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([ 8.0, 25.0, 27.0])
        result = evaluator.calculate_mae(y_true, y_pred)
        assert abs(result - 10 / 3) < 1e-6

    def test_empty_arrays(self, evaluator):
        assert evaluator.calculate_mae(np.array([]), np.array([])) == 0.0

    def test_accepts_series(self, evaluator):
        y_true = pd.Series([5.0, 10.0, 15.0])
        y_pred = pd.Series([5.0, 10.0, 15.0])
        assert evaluator.calculate_mae(y_true, y_pred) == 0.0


# ---------------------------------------------------------------------------
# Test: calculate_mape
# ---------------------------------------------------------------------------

class TestCalculateMAPE:
    def test_perfect_prediction(self, evaluator):
        y = np.array([100.0, 200.0, 300.0])
        assert evaluator.calculate_mape(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_known_percentage(self, evaluator):
        # y_true = [100], y_pred = [110] → |100-110|/100 * 100 = 10%
        result = evaluator.calculate_mape([100.0], [110.0])
        assert abs(result - 10.0) < 1e-6

    def test_no_division_by_zero(self, evaluator):
        # y_true = 0 → harus gunakan epsilon, tidak crash
        result = evaluator.calculate_mape([0.0, 100.0], [10.0, 110.0])
        assert result >= 0.0 and not np.isinf(result)

    def test_empty(self, evaluator):
        assert evaluator.calculate_mape([], []) == 0.0

    def test_returns_percentage(self, evaluator):
        # Galat konsisten 20%
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = y_true * 1.20
        result = evaluator.calculate_mape(y_true, y_pred)
        assert abs(result - 20.0) < 1e-6


# ---------------------------------------------------------------------------
# Test: calculate_rmse
# ---------------------------------------------------------------------------

class TestCalculateRMSE:
    def test_perfect_prediction(self, evaluator):
        y = np.array([5.0, 10.0, 15.0])
        assert evaluator.calculate_rmse(y, y) == 0.0

    def test_known_value(self, evaluator):
        # (1-0)^2 + (2-0)^2 + (3-0)^2 = 1 + 4 + 9 = 14; sqrt(14/3) ≈ 2.160
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.zeros(3)
        expected = np.sqrt(14 / 3)
        result = evaluator.calculate_rmse(y_true, y_pred)
        assert abs(result - expected) < 1e-6

    def test_empty(self, evaluator):
        assert evaluator.calculate_rmse([], []) == 0.0

    def test_larger_errors_penalized(self, evaluator):
        # RMSE memberi penalti lebih besar pada galat ekstrem dibanding MAE
        y_true = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        y_pred = np.zeros(5)
        rmse = evaluator.calculate_rmse(y_true, y_pred)
        mae  = evaluator.calculate_mae(y_true, y_pred)
        assert rmse > mae


# ---------------------------------------------------------------------------
# Test: interpret_mape
# ---------------------------------------------------------------------------

class TestInterpretMAPE:
    @pytest.mark.parametrize("mape,expected_label,expected_emoji", [
        (5.0,  "Sangat Baik", "🟢"),
        (15.0, "Baik",        "🔵"),
        (35.0, "Layak",       "🟡"),
        (75.0, "Kurang",      "🔴"),
        (0.0,  "Sangat Baik", "🟢"),
        (9.99, "Sangat Baik", "🟢"),
        (10.0, "Baik",        "🔵"),
        (20.0, "Layak",       "🟡"),
        (50.0, "Kurang",      "🔴"),
    ])
    def test_threshold_boundaries(self, evaluator, mape, expected_label, expected_emoji):
        label, emoji = evaluator.interpret_mape(mape)
        assert label == expected_label, f"MAPE={mape} → label='{label}' (expected '{expected_label}')"
        assert emoji == expected_emoji


# ---------------------------------------------------------------------------
# Test: compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_output_contains_both_models(self, evaluator):
        y_true   = np.array([100.0, 120.0, 110.0, 130.0])
        y_prophet = y_true * 1.10
        y_hybrid  = y_true * 1.05

        result = evaluator.compare_models(y_true, y_prophet, y_hybrid)
        models = result["model"].tolist()
        assert "Prophet (Tunggal)" in models
        assert "Hibrida Prophet + LightGBM" in models

    def test_hybrid_lower_error(self, evaluator):
        # Hibrida harus lebih akurat (lebih kecil MAPE)
        y_true   = np.array([100.0, 200.0, 150.0, 180.0])
        y_prophet = y_true + 20   # galat lebih besar
        y_hybrid  = y_true + 5    # galat lebih kecil

        result = evaluator.compare_models(y_true, y_prophet, y_hybrid)
        mape_p = result.loc[result["model"] == "Prophet (Tunggal)",        "MAPE_%"].values[0]
        mape_h = result.loc[result["model"] == "Hibrida Prophet + LightGBM", "MAPE_%"].values[0]

        # Konversi ke float jika string
        mape_p = float(mape_p) if isinstance(mape_p, (int, float)) else float(str(mape_p))
        mape_h = float(mape_h) if isinstance(mape_h, (int, float)) else float(str(mape_h))
        assert mape_h < mape_p


# ---------------------------------------------------------------------------
# Test: build_evaluation_report
# ---------------------------------------------------------------------------

class TestBuildEvaluationReport:
    def test_report_keys(self, evaluator):
        test_df = pd.DataFrame({
            "y":         [100.0, 120.0, 110.0, 130.0],
            "y_prophet": [105.0, 125.0, 115.0, 140.0],
            "y_hybrid":  [101.0, 121.0, 111.0, 132.0],
        })
        report = evaluator.build_evaluation_report(test_df)
        assert "metrics_prophet" in report
        assert "metrics_hybrid" in report
        assert "comparison_table" in report
        assert "mape_interpretation" in report
        assert "mape_emoji" in report
        assert "n_test_points" in report

    def test_n_test_points(self, evaluator):
        n = 8
        test_df = pd.DataFrame({
            "y":         np.random.rand(n) * 100,
            "y_prophet": np.random.rand(n) * 100,
            "y_hybrid":  np.random.rand(n) * 100,
        })
        report = evaluator.build_evaluation_report(test_df)
        assert report["n_test_points"] == n
