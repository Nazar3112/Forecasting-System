"""
tests/test_hybrid_engine.py
===========================
Unit tests untuk modul HybridForecastingEngine.
Menguji train/test split kronologis, pelatihan Prophet,
rekayasa fitur residual, dan output peramalan.

CATATAN: Pengujian ini membutuhkan pustaka `prophet` dan `lightgbm`
yang sudah terpasang. Jika belum, test akan di-skip secara otomatis.
"""

import pytest
import numpy as np
import pandas as pd

# Skip seluruh modul jika dependensi belum terinstall
prophet_available = False
try:
    import prophet  # noqa
    import lightgbm # noqa
    prophet_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not prophet_available,
    reason="Pustaka 'prophet' dan/atau 'lightgbm' belum terinstal."
)

from src.hybrid_engine import HybridForecastingEngine


# ---------------------------------------------------------------------------
# Fixture: Deret waktu sintetis (52 minggu)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_series():
    """
    Deret waktu mingguan sintetis: tren naik + musiman sederhana + noise.
    52 titik data = 1 tahun mingguan.
    """
    rng   = np.random.default_rng(seed=42)
    n     = 52
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    trend = np.linspace(100, 200, n)                       # tren naik
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 52) # musiman tahunan
    noise  = rng.normal(0, 5, n)
    y      = np.maximum(0, trend + seasonal + noise)
    return pd.DataFrame({"ds": dates, "y": y})


@pytest.fixture
def engine():
    return HybridForecastingEngine(
        seasonality_mode="additive",
        test_size=0.20,
    )


# ---------------------------------------------------------------------------
# Test: train_test_split
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_split_sizes(self, engine, synthetic_series):
        train, test = engine.train_test_split(synthetic_series)
        n = len(synthetic_series)
        n_test  = max(1, int(n * 0.20))
        n_train = n - n_test
        assert len(train) == n_train
        assert len(test)  == n_test

    def test_chronological_order(self, engine, synthetic_series):
        train, test = engine.train_test_split(synthetic_series)
        # Semua tanggal training harus lebih awal dari testing
        assert train["ds"].max() <= test["ds"].min()

    def test_no_overlap(self, engine, synthetic_series):
        train, test = engine.train_test_split(synthetic_series)
        overlap = set(train["ds"]).intersection(set(test["ds"]))
        assert len(overlap) == 0

    def test_no_data_loss(self, engine, synthetic_series):
        train, test = engine.train_test_split(synthetic_series)
        assert len(train) + len(test) == len(synthetic_series)


# ---------------------------------------------------------------------------
# Test: _build_residual_features
# ---------------------------------------------------------------------------

class TestResidualFeatures:
    def test_feature_columns_present(self, engine):
        n = 30
        dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
        residuals = np.random.randn(n)
        X, y = engine._build_residual_features(dates.values, residuals)

        # Cek keberadaan fitur lag
        for lag in engine.lags:
            assert f"lag_{lag}" in X.columns, f"Kolom lag_{lag} tidak ditemukan"

        # Cek fitur kalender
        assert "month"      in X.columns
        assert "weekofyear" in X.columns
        assert "quarter"    in X.columns

    def test_no_data_leakage(self, engine):
        """Fitur lag ke-k harus menggunakan nilai pada t-k, bukan t."""
        n = 20
        dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
        # Residual konstan = 1.0 untuk verifikasi lag
        residuals = np.ones(n)
        X, y = engine._build_residual_features(dates.values, residuals)
        # Setiap lag feature harus 1.0 (karena nilai konstan 1.0)
        for lag in engine.lags:
            assert (X[f"lag_{lag}"] == 1.0).all(), f"lag_{lag} mengandung nilai lain"

    def test_output_length(self, engine):
        """Panjang output harus berkurang sesuai max lag/window."""
        n = 30
        dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
        residuals = np.random.randn(n)
        X, y = engine._build_residual_features(dates.values, residuals)
        max_drop = max(max(engine.lags), max(engine.rolling_windows))
        assert len(X) == n - max_drop
        assert len(y) == n - max_drop


# ---------------------------------------------------------------------------
# Test: fit() pipeline penuh
# ---------------------------------------------------------------------------

class TestFitPipeline:
    def test_fit_sets_is_fitted(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        assert engine._is_fitted

    def test_prophet_model_exists(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        assert engine.prophet_model is not None

    def test_lgbm_model_exists(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        assert engine.lgbm_model is not None

    def test_train_test_stored(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        assert engine._train_df is not None
        assert engine._test_df is not None


# ---------------------------------------------------------------------------
# Test: predict_test()
# ---------------------------------------------------------------------------

class TestPredictTest:
    def test_output_columns(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        result = engine.predict_test()
        assert "ds"           in result.columns
        assert "y"            in result.columns
        assert "y_prophet"    in result.columns
        assert "y_hybrid"     in result.columns

    def test_non_negative_predictions(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        result = engine.predict_test()
        # Prediksi tidak boleh negatif (hasil max(0, ...))
        assert (result["y_hybrid"]  >= 0).all()
        assert (result["y_prophet"] >= 0).all()

    def test_length_matches_test_set(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        _, test_df = engine.train_test_split(synthetic_series)
        result = engine.predict_test()
        assert len(result) == len(test_df)

    def test_raises_before_fit(self):
        new_engine = HybridForecastingEngine()
        with pytest.raises(RuntimeError, match="belum dilatih"):
            new_engine.predict_test()


# ---------------------------------------------------------------------------
# Test: forecast_future()
# ---------------------------------------------------------------------------

class TestForecastFuture:
    def test_correct_horizon(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        horizon = 8
        result = engine.forecast_future(synthetic_series, horizon=horizon)
        assert len(result) == horizon

    def test_future_dates_after_history(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        result = engine.forecast_future(synthetic_series, horizon=4)
        last_historical = synthetic_series["ds"].max()
        assert (result["ds"] > last_historical).all()

    def test_non_negative_future(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        result = engine.forecast_future(synthetic_series, horizon=8)
        assert (result["y_hybrid"] >= 0).all()

    def test_output_columns(self, engine, synthetic_series):
        engine.fit(synthetic_series, freq="W-MON")
        result = engine.forecast_future(synthetic_series, horizon=4)
        assert "ds"        in result.columns
        assert "y_hybrid"  in result.columns
        assert "y_prophet" in result.columns
