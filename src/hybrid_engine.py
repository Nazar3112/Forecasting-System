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
    1. train_test_split()          → bagi data kronologis 80/20
    2. fit_prophet()               → latih Prophet pada data training
    3. extract_residuals()         → hitung galat e(t) = y_aktual − ŷ_Prophet
    4. build_feature_matrix()      → buat fitur eksogen (ds) + lag QTY (y)
    5. fit_lightgbm_residual()     → latih LGBMRegressor pada residual
    6. generate_hybrid_predictions → proyeksi hibrida pada test / masa depan
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
        lags             : Langkah kelambatan fitur lag QTY.
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
        self._train_df:       Optional[pd.DataFrame] = None
        self._test_df:        Optional[pd.DataFrame] = None
        self._residuals:      Optional[np.ndarray]   = None
        self._feature_names:  List[str]              = []
        self._freq:           str = DEFAULT_FREQ

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
        Jalankan pipeline pelatihan lengkap: Prophet -> Residual -> LightGBM.

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

        # 1. Train / Test split secara kronologis
        self._train_df, self._test_df = self.train_test_split(series_df)
        logger.info(
            "Split: train=%d | test=%d periode",
            len(self._train_df), len(self._test_df),
        )

        # 2. Fit Prophet pada data latih
        prophet_train_pred = self._fit_prophet_internal(self._train_df)

        # 3. Hitung target residual: residuals = y_aktual - y_pred_prophet
        self._residuals = self._train_df["y"].values - prophet_train_pred

        # 4. Bangun matriks fitur dari data latih (fitur eksogen ds + lag QTY)
        full_train_feat = self.build_feature_matrix(self._train_df, self._train_df["y"])
        max_drop = max(max(self.lags), max(self.rolling_windows)) if self.lags and self.rolling_windows else 0

        X_train = full_train_feat.iloc[max_drop:].reset_index(drop=True)
        y_train = self._residuals[max_drop:]

        # 5. Fit LightGBM pada residual
        self.fit_lightgbm_residual(X_train, y_train)

        self._is_fitted = True
        return self

    def predict_test(self) -> pd.DataFrame:
        """
        Hasilkan prediksi hybrid pada data uji dan kembalikan DataFrame
        dengan kolom: ['ds', 'y', 'y_prophet', 'y_residual_lgbm', 'y_hybrid'].
        """
        self._require_fitted()
        return self.generate_hybrid_predictions(self._test_df, is_future=False)

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

        # Buat deret tanggal masa depan
        last_date = series_df["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        future_df = pd.DataFrame({"ds": future_dates, "y": np.nan})
        return self.generate_hybrid_predictions(future_df, is_future=True)

    # ------------------------------------------------------------------
    # Split & Training
    # ------------------------------------------------------------------

    def train_test_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Bagi deret waktu secara kronologis menjadi data latih dan uji.
        """
        n_test  = max(1, int(len(df) * self.test_size))
        n_train = len(df) - n_test
        train_df = df.iloc[:n_train].reset_index(drop=True)
        test_df  = df.iloc[n_train:].reset_index(drop=True)
        return train_df, test_df

    def _fit_prophet_internal(self, train_df: pd.DataFrame) -> np.ndarray:
        """
        Inisialisasi dan latih Prophet; kembalikan prediksi yhat pada data latih.
        """
        self.prophet_model = Prophet(**self._prophet_params)

        import logging as _log
        _log.getLogger("prophet").setLevel(_log.WARNING)
        _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

        self.prophet_model.fit(train_df[["ds", "y"]])
        forecast = self.prophet_model.predict(train_df[["ds"]])
        return forecast["yhat"].values

    def fit_lightgbm_residual(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> None:
        """
        Latih LGBMRegressor menggunakan matriks fitur X_train dan target y_train (residuals).
        """
        if len(X_train) == 0:
            logger.warning("X_train kosong. LightGBM tidak dapat dilatih.")
            return

        self._feature_names = list(X_train.columns)
        self.lgbm_model = LGBMRegressor(**self._lgbm_params)
        self.lgbm_model.fit(X_train, y_train)
        logger.info(
            "LightGBM Residual dilatih: %d sampel, %d fitur (%s)",
            len(X_train), X_train.shape[1], ", ".join(self._feature_names[:4]) + "...",
        )

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    @staticmethod
    def extract_exogenous_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
        """
        Mengekstrak fitur eksogen kalender dari kolom waktu ds:
          - month          : Bulan (1-12)
          - weekofyear     : Minggu ke-n dalam setahun (1-53)
          - quarter        : Kuartal (1-4)
          - day            : Hari dalam bulan (1-31)
          - dayofweek      : Hari dalam minggu (0=Senin, 6=Minggu)
          - is_month_end   : Indikator akhir bulan (0 / 1)
          - is_month_start : Indikator awal bulan (0 / 1)
        """
        dt = pd.to_datetime(dates)
        return pd.DataFrame({
            "month":          dt.dt.month.values,
            "weekofyear":     dt.dt.isocalendar().week.astype(int).values,
            "quarter":        dt.dt.quarter.values,
            "day":            dt.dt.day.values,
            "dayofweek":      dt.dt.dayofweek.values,
            "is_month_end":   dt.dt.is_month_end.astype(int).values,
            "is_month_start": dt.dt.is_month_start.astype(int).values,
        })

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        y_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Bangun matriks fitur independen (X) lengkap:
        Fitur eksogen waktu + fitur lag kuantitas QTY (1, 2, 3, ...) + rolling statistics.
        """
        exog = self.extract_exogenous_features(df["ds"])
        exog.index = df.index

        if y_series is not None:
            s = pd.Series(y_series.values, index=df.index)
            # Fitur lag kuantitas (QTY)
            for lag in self.lags:
                exog[f"lag_{lag}"] = s.shift(lag).values
            # Fitur rolling statistics kuantitas
            for win in self.rolling_windows:
                exog[f"roll_mean_{win}"] = s.shift(1).rolling(win).mean().values
                exog[f"roll_std_{win}"]  = s.shift(1).rolling(win).std().fillna(0).values

        return exog

    def _build_residual_features(
        self,
        dates:     np.ndarray,
        residuals: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Kompatibilitas unit-test: Bangun fitur lag dan kalender dari deret nilai.
        """
        df_temp = pd.DataFrame({"ds": dates})
        feat = self.build_feature_matrix(df_temp, pd.Series(residuals))

        max_lag = max(self.lags) if self.lags else 0
        max_win = max(self.rolling_windows) if self.rolling_windows else 0
        drop_n  = max(max_lag, max_win)

        feat_clean = feat.iloc[drop_n:].reset_index(drop=True)
        y_clean    = residuals[drop_n:]
        return feat_clean, y_clean

    # ------------------------------------------------------------------
    # Inference & Hibrida
    # ------------------------------------------------------------------

    def generate_hybrid_predictions(
        self,
        period_df:  pd.DataFrame,
        is_future:  bool = False,
    ) -> pd.DataFrame:
        """
        Menghasilkan prediksi hibrida y_hybrid = max(0, y_prophet + y_residual_lgbm).
        Memastikan struktur matriks fitur persis identik dengan data latih.
        """
        self._require_fitted()
        result = period_df[["ds"]].copy()
        if "y" in period_df.columns:
            result["y"] = period_df["y"].values

        # 1. Prediksi Prophet
        prophet_forecast = self.prophet_model.predict(result[["ds"]])
        result["y_prophet"] = prophet_forecast["yhat"].values.clip(min=0)

        # 2. Prediksi Residual LightGBM
        if self.lgbm_model is not None and self._feature_names:
            if not is_future:
                # Mode Data Uji (Test Set):
                # Gabungkan data latih + uji untuk menghitung nilai lag QTY aktual tanpa kebocoran data
                combined_df = pd.concat([self._train_df, period_df], ignore_index=True)
                full_feat = self.build_feature_matrix(combined_df, combined_df["y"])
                X_test = full_feat.tail(len(period_df))[self._feature_names].reset_index(drop=True)
                result["y_residual_lgbm"] = self.lgbm_model.predict(X_test)
            else:
                # Mode Proyeksi Masa Depan (Future Forecast):
                # Peramalan rekursif/autoregresif menggunakan buffer riwayat QTY
                full_hist = self._train_df if self._test_df is None else pd.concat([self._train_df, self._test_df], ignore_index=True)
                y_buffer = list(full_hist["y"].values)
                res_preds = []

                for i, row_ds in enumerate(result["ds"]):
                    dt = pd.to_datetime(row_ds)
                    feat_row = {
                        "month":          dt.month,
                        "weekofyear":     int(dt.isocalendar().week),
                        "quarter":        dt.quarter,
                        "day":            dt.day,
                        "dayofweek":      dt.dayofweek,
                        "is_month_end":   int(dt.is_month_end),
                        "is_month_start": int(dt.is_month_start),
                    }
                    for lag in self.lags:
                        feat_row[f"lag_{lag}"] = y_buffer[-lag] if len(y_buffer) >= lag else 0.0
                    for win in self.rolling_windows:
                        w_vals = y_buffer[-win:] if len(y_buffer) >= win else y_buffer
                        feat_row[f"roll_mean_{win}"] = float(np.mean(w_vals)) if w_vals else 0.0
                        feat_row[f"roll_std_{win}"]  = float(np.std(w_vals)) if len(w_vals) > 1 else 0.0

                    X_step = pd.DataFrame([feat_row])[self._feature_names]
                    step_res = float(self.lgbm_model.predict(X_step)[0])
                    res_preds.append(step_res)

                    # Update buffer dengan nilai prediksi hybrid untuk lag step berikutnya
                    step_hybrid = max(0.0, float(result["y_prophet"].iloc[i] + step_res))
                    y_buffer.append(step_hybrid)

                result["y_residual_lgbm"] = res_preds
        else:
            result["y_residual_lgbm"] = 0.0

        # 3. Rekonsiliasi Hibrida: y_hybrid = max(0, y_prophet + y_residual_lgbm)
        result["y_hybrid"] = (result["y_prophet"] + result["y_residual_lgbm"]).clip(lower=0)

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------

    def get_prophet_components(self) -> pd.DataFrame | None:
        """
        Kembalikan DataFrame komponen Prophet (trend, weekly, yearly).
        """
        if self.prophet_model is None or self._train_df is None:
            return None
        full_df = pd.concat([self._train_df, self._test_df], ignore_index=True)
        forecast = self.prophet_model.predict(full_df[["ds"]])
        return forecast[["ds", "trend", "yhat", "yhat_lower", "yhat_upper"]]

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Kembalikan tabel feature importance LightGBM (gain-based / split-based).
        """
        if self.lgbm_model is None or not hasattr(self.lgbm_model, "feature_importances_"):
            return None
        importances = self.lgbm_model.feature_importances_
        feature_names = self._feature_names or [f"feature_{i}" for i in range(len(importances))]

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
