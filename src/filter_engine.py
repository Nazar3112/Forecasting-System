"""
filter_engine.py
================
Kelas DataFilterEngine: memisahkan sub-deret waktu per kategori produk
yang dipilih pengguna, memvalidasi kecukupan historis, dan menyiapkan
DataFrame siap pakai untuk HybridForecastingEngine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import MIN_DATA_POINTS

logger = logging.getLogger(__name__)


class DataFilterEngine:
    """
    Bertanggung jawab memisahkan sub-deret waktu berdasarkan kategori
    produk yang dipilih pengguna serta memvalidasi kecukupan historis
    data sebelum diserahkan ke mesin peramalan.

    Input yang diharapkan adalah DataFrame dari DataPreprocessor dengan
    kolom standar: ['category', 'ds', 'y'].
    """

    def __init__(self, min_points: int = MIN_DATA_POINTS):
        """
        Parameters
        ----------
        min_points : int
            Ambang batas minimal observasi deret waktu agar pola musiman
            dapat terdeteksi secara statistik (default: 24 periode).
        """
        self.min_points: int = min_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_by_category(
        self,
        df: pd.DataFrame,
        selected_cat: str,
        category_col: str = "category",
    ) -> pd.DataFrame:
        """
        Ekstrak sub-deret waktu untuk satu kategori produk terpilih.

        Parameters
        ----------
        df : pd.DataFrame        DataFrame agregat ['category', 'ds', 'y'].
        selected_cat : str       Kode kategori/divisi yang dipilih pengguna.
        category_col : str       Nama kolom kategori di DataFrame.

        Returns
        -------
        pd.DataFrame  Sub-deret waktu kolom ['ds', 'y'] terurut berdasarkan
                      tanggal, dengan index direset.

        Raises
        ------
        ValueError  Jika kode kategori tidak ditemukan di DataFrame.
        """
        available = df[category_col].unique().tolist()
        if selected_cat not in available:
            raise ValueError(
                f"Kategori '{selected_cat}' tidak ditemukan. "
                f"Pilihan tersedia: {sorted(available)}"
            )

        sub_df = (
            df[df[category_col] == selected_cat][["ds", "y"]]
            .sort_values("ds")
            .drop_duplicates(subset="ds")
            .reset_index(drop=True)
        )

        logger.debug(
            "filter_by_category('%s'): %d baris diekstrak.", selected_cat, len(sub_df)
        )
        return sub_df

    def validate_sufficiency(
        self,
        series_df: pd.DataFrame,
        min_points: int | None = None,
    ) -> Tuple[bool, int, str]:
        """
        Periksa apakah jumlah observasi historis mencukupi untuk
        analisis tren dan musiman.

        Parameters
        ----------
        series_df : pd.DataFrame  Sub-deret waktu kolom ['ds', 'y'].
        min_points : int, optional  Override ambang batas minimal.

        Returns
        -------
        (bool, int, str)
            - True  jika data mencukupi
            - Jumlah baris aktual
            - Pesan status (info/peringatan)
        """
        threshold = min_points or self.min_points
        n = len(series_df)
        if n < threshold:
            msg = (
                f"⚠️ Data historis hanya {n} titik observasi "
                f"(minimal {threshold} diperlukan untuk analisis musiman). "
                "Peramalan mungkin tidak akurat — coba pilih interval Bulanan "
                "atau kategori lain dengan lebih banyak data."
            )
            logger.warning(msg)
            return False, n, msg

        msg = f"✅ Data mencukupi: {n} titik observasi (min. {threshold})."
        logger.debug(msg)
        return True, n, msg

    def get_date_range_summary(
        self, series_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Rangkuman statistik deret waktu untuk ditampilkan di sidebar.

        Parameters
        ----------
        series_df : pd.DataFrame  Sub-deret waktu ['ds', 'y'].

        Returns
        -------
        dict  Berisi tanggal awal, tanggal akhir, total periode,
              total penjualan, rata-rata, dan nilai max.
        """
        if series_df.empty:
            return {}

        return {
            "tanggal_awal":     series_df["ds"].min().strftime("%d %b %Y"),
            "tanggal_akhir":    series_df["ds"].max().strftime("%d %b %Y"),
            "total_periode":    len(series_df),
            "total_penjualan":  int(series_df["y"].sum()),
            "rata_penjualan":   round(float(series_df["y"].mean()), 2),
            "maks_penjualan":   int(series_df["y"].max()),
            "min_penjualan":    int(series_df["y"].min()),
        }

    def prepare_for_forecast(
        self,
        df: pd.DataFrame,
        selected_cat: str,
        category_col: str = "category",
    ) -> Tuple[pd.DataFrame, Dict[str, Any], bool, str]:
        """
        Pipeline tunggal: filter → validasi → rangkuman.
        Mengembalikan semua informasi yang dibutuhkan sebelum mesin
        peramalan dijalankan.

        Returns
        -------
        (series_df, summary, is_valid, message)
        """
        series_df = self.filter_by_category(df, selected_cat, category_col)
        is_valid, n_points, message = self.validate_sufficiency(series_df)
        summary = self.get_date_range_summary(series_df)
        summary["jumlah_observasi"] = n_points
        return series_df, summary, is_valid, message

    def list_low_data_categories(
        self,
        df: pd.DataFrame,
        category_col: str = "category",
    ) -> List[str]:
        """
        Daftar kode kategori yang belum memiliki data historis mencukupi.
        Berguna untuk memberi peringatan di UI saat pengguna memilih dropdown.

        Returns
        -------
        List[str]  Kode kategori dengan jumlah observasi < min_points.
        """
        counts = df.groupby(category_col).size()
        return sorted(counts[counts < self.min_points].index.tolist())
