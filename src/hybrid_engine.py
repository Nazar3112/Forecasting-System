"""
hybrid_engine.py
================
Kelas HybridForecastingEngine: pipeline peramalan dua tahap
  Tahap 1 → Facebook Prophet  (dekomposisi tren + musiman)
  Tahap 2 → LightGBM          (koreksi residual non-linier)
  Final   → y_hybrid = max(0, y_prophet + ê_lightgbm)

Referensi matematis dari proposal skripsi (hal. 13–14):
  y(t) = g(t) + s(t)                          [Prophet – aditif]
  residual = y_aktual − ŷ_Prophet
  ŷ_hybrid = ŷ_Prophet + ê_LightGBM
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from lightgbm import LGBMRegressor

from src.config import (
    DEFAULT_TEST_SIZE,
    DEFAULT_FREQ,
    DEFAULT_HORIZON,
    PROPHET_PARAMS,
    LIGHTGBM_PARAMS,
    RESIDUAL_LAG_STEPS,
    ROLLING_WINDOWS,
)

logger = logging.getLogger(__name__)


class HybridForecastingEngine:
    """
    Mesin peramalan hibrida Prophet + LightGBM.

    Alur komputasi:
    1. train_test_split()     → bagi data kronologis 80/20
    2. fit_prophet()          → latih Prophet pada data training
    3. extract_residuals()    → hitung galat e(t) = y − ŷ_Prophet
    4. build_features()       → buat fitur lag + rolling + kalender
    5. fit_lightgbm()         → latih LGBMRegressor pada residual
    6. evaluate_on_test()     → proyeksi hybrid di data uji
    7. forecast_future()      → proyeksi horizon masa depan
    """

    def __init__(
        self,
        seasonality_mode:  str   = "additive",
        test_size:         float = DEFAULT_TEST_SIZE,
        lags:              List[int] | None = None,
        rolling_windows:   List[int] | None = None,
        prophet_params:    Dict | None = None,
        lgbm_params:       Dict | None = None,
    ):
        """
        Parameters
        ----------
        seasonality_mode : 'additive' | 'multiplicative'
        test_size        : Proporsi data uji kronologis (0.0–1.0).
        lags             : Langkah kelambatan fitur residual.
        rolling_windows  : Ukuran jendela rolling statistics.
        prophet_params   : Override parameter Prophet.
        lgbm_params      : Override parameter LightGBM.
        """
        self.seasonality_mode = seasonality_mode
        self.test_size        = test_size
        self.lags             = lags or RESIDUAL_LAG_STEPS
        self.rolling_windows  = rolling_windows or ROLLING_WINDOWS

        # Merge parameter default dengan override
        _prophet = {**PROPHET_PARAMS, "seasonality_mode": seasonality_mode}
        if prophet_params:
            _prophet.update(prophet_params)
        self._prophet_params: Dict = _prophet

        _lgbm = {**LIGHTGBM_PARAMS}
        if lgbm_params:
            _lgbm.update(lgbm_params)
        self._lgbm_params: Dict = _lgbm

        # Model instances
        self.prophet_model: Optional[Prophet]      = None
        self.lgbm_model:    Optional[LGBMRegressor] = None

        # Data intermediat (tersimpan di-memory)
        self._train_df:    Optional[pd.DataFrame] = None
        self._test_df:     Optional[pd.DataFrame] = None
        self._residuals:   Optional[pd.Series]    = None
        self._freq:        str = DEFAULT_FREQ

        # Status
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        series_df: pd.DataFrame,
        freq:      str = DEFAULT_FREQ,
    ) -> "HybridForecastingEngine":
        """
        Jalankan pipeline pelatihan lengkap.

        Parameters
        ----------
        series_df : pd.DataFrame  Kolom ['ds', 'y'] (deret waktu kategori).
        freq      : str            Frekuensi resample ('W-MON' atau 'ME').

        Returns
        -------
        self
        """
        self._freq = freq
        series_df  = series_df[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)

        # 1. Train / Test split
        self._train_df, self._test_df = self.train_test_split(series_df)
        logger.info(
            "Split: train=%d | test=%d periode",
            len(self._train_df), len(self._test_df),
        )

        # 2. Fit Prophet
        prophet_train_pred = self._fit_prophet_internal(self._train_df)

        # 3. Hitung residual pada data training
        self._residuals = self._train_df["y"].values - prophet_train_pred

        # 4. Bangun fitur dari residual training
        X_train, y_res_train = self._build_residual_features(
            self._train_df["ds"].values, self._residuals
        )

        # 5. Fit LightGBM pada residual
        if len(X_train) > 0:
            self._fit_lgbm_internal(X_train, y_res_train)
        else:
            logger.warning("Tidak cukup data setelah feature engineering untuk melatih LightGBM.")

        self._is_fitted = True
        return self

    def predict_test(self) -> pd.DataFrame:
        """
        Hasilkan prediksi hybrid pada data uji dan kembalikan DataFrame
        dengan kolom: ['ds', 'y', 'y_prophet', 'y_residual_lgbm', 'y_hybrid'].

        Returns
        -------
        pd.DataFrame
        """
        self._require_fitted()
        return self._predict_period(self._test_df, is_future=False)

    def forecast_future(
        self,
        series_df: pd.DataFrame,
        horizon:   int = DEFAULT_HORIZON,
    ) -> pd.DataFrame:
        """
        Proyeksi kuantitas penjualan untuk n-periode ke masa depan.

        Parameters
        ----------
        series_df : pd.DataFrame  Seluruh deret waktu historis ['ds', 'y'].
        horizon   : int            Jumlah periode masa depan.

        Returns
        -------
        pd.DataFrame  Kolom: ['ds', 'y_prophet', 'y_residual_lgbm', 'y_hybrid']
        """
        self._require_fitted()

        # Buat DataFrame tanggal masa depan (mulai setelah periode terakhir)
        last_date  = series_df["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        future_df = pd.DataFrame({"ds": future_dates, "y": np.nan})
        return self._predict_period(future_df, is_future=True)

    # ------------------------------------------------------------------
    # Split & Training
    # ------------------------------------------------------------------

    def train_test_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Bagi deret waktu secara kronologis menjadi data latih dan uji.
        Tidak menggunakan random shuffle — urutan waktu dipertahankan.

        Returns
        -------
        (train_df, test_df)
        """
        n_test  = max(1, int(len(df) * self.test_size))
        n_train = len(df) - n_test
        train_df = df.iloc[:n_train].reset_index(drop=True)
        test_df  = df.iloc[n_train:].reset_index(drop=True)
        return train_df, test_df

    def _fit_prophet_internal(self, train_df: pd.DataFrame) -> np.ndarray:
        """
        Inisialisasi dan latih Prophet; kembalikan prediksi pada data latih.
        """
        self.prophet_model = Prophet(**self._prophet_params)

        # Suppressi log verbose bawaan Prophet/Stan
        import logging as _log
        _log.getLogger("prophet").setLevel(_log.WARNING)
        _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

        self.prophet_model.fit(train_df[["ds", "y"]])
        forecast = self.prophet_model.predict(train_df[["ds"]])
        return forecast["yhat"].values

    def _fit_lgbm_internal(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> None:
        """Inisialisasi dan latih LGBMRegressor."""
        self.lgbm_model = LGBMRegressor(**self._lgbm_params)
        self.lgbm_model.fit(X_train, y_train)
        logger.info(
            "LightGBM dilatih: %d sampel, %d fitur",
            len(X_train), X_train.shape[1],
        )

    # ------------------------------------------------------------------
    # Feature Engineering Residual
    # ------------------------------------------------------------------

    def _build_residual_features(
        self,
        dates:     np.ndarray,
        residuals: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Bangun matriks fitur tabular untuk melatih LightGBM pada residual.

        Fitur yang dibangun:
          - Lag residual e(t-k) untuk setiap k di self.lags
          - Rolling mean residual dengan jendela di self.rolling_windows
          - Rolling std  residual dengan jendela di self.rolling_windows
          - Fitur kalender: bulan, minggu-dalam-tahun, kuartal

        Returns
        -------
        (X : pd.DataFrame, y : np.ndarray)
        """
        series = pd.Series(residuals, index=pd.DatetimeIndex(dates))
        feat   = pd.DataFrame(index=series.index)

        # Lag features
        for lag in self.lags:
            feat[f"lag_{lag}"] = series.shift(lag)

        # Rolling statistics
        for win in self.rolling_windows:
            feat[f"roll_mean_{win}"] = series.shift(1).rolling(win).mean()
            feat[f"roll_std_{win}"]  = series.shift(1).rolling(win).std()

        # Kalender
        idx = pd.DatetimeIndex(dates)
        feat["month"]     = idx.month
        feat["weekofyear"] = idx.isocalendar().week.astype(int).values
        feat["quarter"]   = idx.quarter

        # Hapus baris dengan NaN (akibat lag/rolling)
        max_lag = max(self.lags) if self.lags else 0
        max_win = max(self.rolling_windows) if self.rolling_windows else 0
        drop_n  = max(max_lag, max_win)

        feat = feat.iloc[drop_n:]
        y    = residuals[drop_n:]

        return feat.reset_index(drop=True), y

    def _build_future_features(
        self,
        all_dates:        pd.DatetimeIndex,
        all_residuals:    np.ndarray,
        horizon:          int,
        forecast_dates:   pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Bangun fitur lag untuk periode masa depan secara rekursif.
        Menggunakan residual historis sebagai basis awal, kemudian
        mengisi residual yang diprediksi secara berulang.
        """
        # Buffer residual: historis + placeholder masa depan
        residual_buffer = list(all_residuals)
        results = []

        for i, fd in enumerate(forecast_dates):
            res_series = pd.Series(residual_buffer)
            feat_row: Dict = {}

            for lag in self.lags:
                idx_lag = len(residual_buffer) - lag
                feat_row[f"lag_{lag}"] = residual_buffer[idx_lag] if idx_lag >= 0 else 0.0

            for win in self.rolling_windows:
                window_vals = residual_buffer[-win:] if len(residual_buffer) >= win else residual_buffer
                feat_row[f"roll_mean_{win}"] = float(np.mean(window_vals)) if window_vals else 0.0
                feat_row[f"roll_std_{win}"]  = float(np.std(window_vals))  if len(window_vals) > 1 else 0.0

            feat_row["month"]      = fd.month
            feat_row["weekofyear"] = fd.isocalendar()[1]
            feat_row["quarter"]    = fd.quarter

            feat_df  = pd.DataFrame([feat_row])
            res_pred = float(self.lgbm_model.predict(feat_df)[0]) if self.lgbm_model else 0.0

            residual_buffer.append(res_pred)
            results.append(feat_row)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Prediksi
    # ------------------------------------------------------------------

    def _predict_period(
        self,
        period_df:  pd.DataFrame,
        is_future:  bool = False,
    ) -> pd.DataFrame:
        """
        Hasilkan prediksi untuk satu periode (bisa test set atau future).

        Langkah:
          1. Prophet memprediksi pada tanggal di period_df.
          2. LightGBM memprediksi koreksi residual.
          3. Jumlahkan, potong ke >= 0.
        """
        result = period_df[["ds"]].copy()
        if "y" in period_df.columns:
            result["y"] = period_df["y"].values

        # Prediksi Prophet
        prophet_forecast = self.prophet_model.predict(result[["ds"]])
        result["y_prophet"] = prophet_forecast["yhat"].values.clip(min=0)

        # Prediksi residual LightGBM
        if self.lgbm_model is not None:
            if is_future:
                # Untuk masa depan, gunakan residual training sebagai basis
                train_dates   = self._train_df["ds"].values
                train_y       = self._train_df["y"].values
                train_prophet = self.prophet_model.predict(
                    self._train_df[["ds"]]
                )["yhat"].values
                train_residuals = train_y - train_prophet

                feat_future = self._build_future_features(
                    all_dates=pd.DatetimeIndex(train_dates),
                    all_residuals=train_residuals,
                    horizon=len(result),
                    forecast_dates=pd.DatetimeIndex(result["ds"].values),
                )
                result["y_residual_lgbm"] = self.lgbm_model.predict(feat_future)
            else:
                # Untuk test set, hitung residual aktual dari training
                # lalu bangun fitur untuk periode uji
                train_prophet_pred = self.prophet_model.predict(
                    self._train_df[["ds"]]
                )["yhat"].values
                train_residuals = self._train_df["y"].values - train_prophet_pred

                X_test, _ = self._build_residual_features(
                    np.concatenate([self._train_df["ds"].values, result["ds"].values]),
                    np.concatenate([train_residuals, np.zeros(len(result))]),
                )
                # Ambil hanya baris sesuai jumlah period uji
                X_test_last = X_test.tail(len(result)).reset_index(drop=True)
                result["y_residual_lgbm"] = self.lgbm_model.predict(X_test_last)
        else:
            result["y_residual_lgbm"] = 0.0

        # Sintesis akhir: gabungkan + clip non-negatif
        result["y_hybrid"] = (result["y_prophet"] + result["y_residual_lgbm"]).clip(lower=0)

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------

    def get_prophet_components(self) -> pd.DataFrame | None:
        """
        Kembalikan DataFrame komponen Prophet (trend, weekly, yearly)
        untuk visualisasi dekomposisi di dasbor.
        """
        if self.prophet_model is None or self._train_df is None:
            return None
        full_df = pd.concat([self._train_df, self._test_df], ignore_index=True)
        forecast = self.prophet_model.predict(full_df[["ds"]])
        return forecast[["ds", "trend", "yhat", "yhat_lower", "yhat_upper"]]

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Kembalikan tabel feature importance LightGBM (gain-based).
        Berguna untuk analisis kontribusi fitur residual di laporan skripsi.
        """
        if self.lgbm_model is None:
            return None
        importances = self.lgbm_model.feature_importances_
        feature_names = self.lgbm_model.feature_name_

        fi_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return fi_df

    def _require_fitted(self) -> None:
        """Lempar RuntimeError jika model belum dilatih."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model belum dilatih. Panggil .fit() terlebih dahulu."
            )
