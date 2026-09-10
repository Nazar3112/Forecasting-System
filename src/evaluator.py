"""
evaluator.py
============
Kelas ModelEvaluator: kalkulasi metrik akurasi peramalan
(MAE, MAPE, RMSE), analisis komparatif model tunggal vs hibrida,
dan interpretasi kualitas hasil prediksi.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import MAPE_THRESHOLDS

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Bertanggung jawab mengevaluasi galat prediksi hasil peramalan
    menggunakan tiga metrik standar ilmiah:
      - MAE  (Mean Absolute Error)
      - MAPE (Mean Absolute Percentage Error)
      - RMSE (Root Mean Squared Error)

    Juga menyediakan fungsi perbandingan performa Prophet tunggal
    vs Model Hibrida Prophet + LightGBM.
    """

    # ------------------------------------------------------------------
    # Metrik Dasar
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_mae(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> float:
        """
        Mean Absolute Error (MAE).

        Formula:
            MAE = (1/n) * Σ |y_t - ŷ_t|

        Parameters
        ----------
        y_true : array-like  Nilai penjualan aktual.
        y_pred : array-like  Nilai hasil peramalan.

        Returns
        -------
        float  Nilai MAE dalam unit barang.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_mape(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Mean Absolute Percentage Error (MAPE) dalam satuan persen (%).

        Formula:
            MAPE = (1/n) * Σ |y_t - ŷ_t| / max(|y_t|, ε) * 100

        Penanganan nilai y_true = 0: menggunakan epsilon kecil untuk
        menghindari pembagian dengan nol.

        Parameters
        ----------
        y_true   : array-like  Nilai aktual.
        y_pred   : array-like  Nilai peramalan.
        epsilon  : float       Nilai batas minimal penyebut (default 1e-6).

        Returns
        -------
        float  Nilai MAPE dalam persen (%).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0:
            return 0.0
        # Proteksi pembagian-nol: gunakan epsilon sebagai floor
        denominator = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
        mape = np.mean(np.abs(y_true - y_pred) / denominator) * 100.0
        return float(mape)

    @staticmethod
    def calculate_rmse(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> float:
        """
        Root Mean Squared Error (RMSE).

        Formula:
            RMSE = sqrt( (1/n) * Σ (y_t - ŷ_t)² )

        Parameters
        ----------
        y_true : array-like  Nilai aktual.
        y_pred : array-like  Nilai peramalan.

        Returns
        -------
        float  Nilai RMSE dalam unit barang (satuan sama dengan data).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0:
            return 0.0
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # ------------------------------------------------------------------
    # Evaluasi Komprehensif & Komparatif
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        model_name: str = "Model",
    ) -> Dict[str, float | str]:
        """
        Hitung ketiga metrik sekaligus dan kembalikan sebagai dictionary.

        Parameters
        ----------
        y_true     : array-like
        y_pred     : array-like
        model_name : str  Label model untuk identifikasi.

        Returns
        -------
        dict  {'model': ..., 'MAE': ..., 'MAPE_%': ..., 'RMSE': ...}
        """
        mae  = self.calculate_mae(y_true, y_pred)
        mape = self.calculate_mape(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)

        logger.info(
            "[%s] MAE=%.2f | MAPE=%.2f%% | RMSE=%.2f",
            model_name, mae, mape, rmse,
        )

        return {
            "model":   model_name,
            "MAE":     round(mae, 4),
            "MAPE_%":  round(mape, 4),
            "RMSE":    round(rmse, 4),
        }

    def compare_models(
        self,
        y_true:        np.ndarray | pd.Series,
        prophet_pred:  np.ndarray | pd.Series,
        hybrid_pred:   np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """
        Buat tabel perbandingan performa antara:
          - Prophet (model tunggal)
          - Prophet + LightGBM (model hibrida)

        Parameters
        ----------
        y_true       : array-like  Data aktual (data uji).
        prophet_pred : array-like  Prediksi dari Prophet saja.
        hybrid_pred  : array-like  Prediksi gabungan hibrida.

        Returns
        -------
        pd.DataFrame  Tabel perbandingan metrik kedua model.
        """
        rows = [
            self.evaluate(y_true, prophet_pred, "Prophet (Tunggal)"),
            self.evaluate(y_true, hybrid_pred,  "Hibrida Prophet + LightGBM"),
        ]
        comp_df = pd.DataFrame(rows).set_index("model")
        comp_df = comp_df.reset_index()

        # Tambahkan kolom delta sebagai string agar tidak ada type mismatch
        for metric in ["MAE", "MAPE_%", "RMSE"]:
            col_delta = f"Δ {metric} (%)"
            delta_vals = []
            for _, row in comp_df.iterrows():
                if row["model"] == "Hibrida Prophet + LightGBM":
                    # Cari nilai Prophet tunggal
                    prophet_row = comp_df[comp_df["model"] == "Prophet (Tunggal)"]
                    if not prophet_row.empty:
                        v_p = float(prophet_row[metric].values[0])
                        v_h = float(row[metric])
                        if v_p > 0:
                            delta_pct = (v_p - v_h) / v_p * 100
                            delta_vals.append(f"+{delta_pct:.2f}%" if delta_pct >= 0 else f"{delta_pct:.2f}%")
                        else:
                            delta_vals.append("N/A")
                    else:
                        delta_vals.append("N/A")
                else:
                    delta_vals.append("-")
            comp_df[col_delta] = delta_vals

        return comp_df

    def interpret_mape(self, mape_val: float) -> Tuple[str, str]:
        """
        Interpretasi kualitas model berdasarkan nilai MAPE
        sesuai kriteria ilmiah pada proposal skripsi.

        Parameters
        ----------
        mape_val : float  Nilai MAPE dalam persen.

        Returns
        -------
        (label, emoji)
            label : 'Sangat Baik' / 'Baik' / 'Layak' / 'Kurang'
            emoji : Ikon visualisasi status model
        """
        emoji_map = {
            "Sangat Baik": "🟢",
            "Baik":        "🔵",
            "Layak":       "🟡",
            "Kurang":      "🔴",
        }
        for label, (lo, hi) in MAPE_THRESHOLDS.items():
            if lo <= mape_val < hi:
                return label, emoji_map.get(label, "⚪")
        return "Kurang", "🔴"

    def build_evaluation_report(
        self,
        test_df:       pd.DataFrame,
        prophet_col:   str = "y_prophet",
        hybrid_col:    str = "y_hybrid",
        actual_col:    str = "y",
    ) -> Dict:
        """
        Buat laporan evaluasi lengkap dari DataFrame hasil pengujian.

        Parameters
        ----------
        test_df      : pd.DataFrame  Kolom [actual, prophet_pred, hybrid_pred].
        prophet_col  : str  Nama kolom prediksi Prophet.
        hybrid_col   : str  Nama kolom prediksi hibrida.
        actual_col   : str  Nama kolom nilai aktual.

        Returns
        -------
        dict  Ringkasan evaluasi lengkap.
        """
        y_true   = test_df[actual_col].values
        y_prophet = test_df[prophet_col].values
        y_hybrid  = test_df[hybrid_col].values

        metrics_prophet = self.evaluate(y_true, y_prophet, "Prophet (Tunggal)")
        metrics_hybrid  = self.evaluate(y_true, y_hybrid,  "Hibrida Prophet + LightGBM")

        mape_val = metrics_hybrid["MAPE_%"]
        label, emoji = self.interpret_mape(mape_val)

        comp_table = self.compare_models(y_true, y_prophet, y_hybrid)

        return {
            "metrics_prophet":      metrics_prophet,
            "metrics_hybrid":       metrics_hybrid,
            "comparison_table":     comp_table,
            "mape_interpretation":  label,
            "mape_emoji":           emoji,
            "n_test_points":        len(y_true),
        }
