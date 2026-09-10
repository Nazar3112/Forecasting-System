"""
preprocessor.py
===============
Kelas DataPreprocessor: Ingesti hemat memori, sanitasi,
parsing tanggal, dan agregasi deret waktu mingguan/bulanan
dari berkas log kasir POS Indomarco (.csv).
"""

from __future__ import annotations

import io
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    LOAD_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_RTYPE,
    DATE_FORMAT,
    DEFAULT_FREQ,
    MIN_DATA_POINTS,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Bertanggung jawab atas seluruh alur praproses data transaksi kasir POS
    dari berkas mentah .csv hingga menghasilkan DataFrame deret waktu yang
    siap dikonsumsi oleh modul peramalan.

    Alur proses:
        1. Baca berkas .csv (dengan seleksi kolom untuk efisiensi RAM).
        2. Sanitasi nilai string (hapus tanda kutip tunggal).
        3. Validasi integritas skema kolom wajib.
        4. Filter transaksi riil (RTYPE == 'J') & kuantitas valid (QTY > 0).
        5. Konversi kolom tanggal ke tipe datetime.
        6. Resample: agregasi kuantitas ke interval mingguan atau bulanan.
        7. Ekstraksi daftar kategori/divisi unik.
    """

    def __init__(self, freq: str = DEFAULT_FREQ):
        """
        Parameters
        ----------
        freq : str
            Frekuensi resample pandas (default: 'W-MON' untuk mingguan).
        """
        self.freq: str = freq
        self._raw_df: pd.DataFrame | None = None
        self._clean_df: pd.DataFrame | None = None
        self._timeseries_df: pd.DataFrame | None = None
        self._categories: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        source: str | io.BytesIO | io.StringIO,
        category_col: str = "DIV",
        freq: str | None = None,
    ) -> "DataPreprocessor":
        """
        Jalankan pipeline praproses lengkap dari satu panggilan.

        Parameters
        ----------
        source : str atau file-like
            Path berkas CSV atau objek BytesIO/StringIO dari st.file_uploader.
        category_col : str
            Kolom pengelompokan kategori agregasi ('DIV' atau 'CAT_COD').
        freq : str, optional
            Override frekuensi resample (misal 'ME' untuk bulanan).

        Returns
        -------
        self : DataPreprocessor
            Instance yang sama (untuk method chaining).
        """
        if freq is not None:
            self.freq = freq

        # 1. Baca CSV
        self._raw_df = self._read_csv(source)

        # 2. Validasi skema
        valid, missing = self.validate_schema(self._raw_df)
        if not valid:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {missing}. "
                "Pastikan berkas CSV memuat kolom: "
                + ", ".join(REQUIRED_COLUMNS)
            )

        # 3. Sanitasi & bersihkan
        self._clean_df = self.clean_data(self._raw_df)

        # 4. Parsing tanggal
        self._clean_df = self.parse_dates(self._clean_df)

        # 5. Resample deret waktu
        self._timeseries_df = self.resample_time_series(
            self._clean_df, category_col=category_col
        )

        # 6. Ekstrak kategori unik
        self._categories = self.extract_categories(
            self._timeseries_df, category_col="category"
        )

        logger.info(
            "Praproses selesai: %d baris deret waktu, %d kategori",
            len(self._timeseries_df),
            len(self._categories),
        )
        return self

    @property
    def timeseries_df(self) -> pd.DataFrame:
        """DataFrame agregat deret waktu: kolom [category, ds, y]."""
        if self._timeseries_df is None:
            raise RuntimeError("Panggil .fit() terlebih dahulu.")
        return self._timeseries_df

    @property
    def categories(self) -> List[str]:
        """Daftar kode kategori/divisi unik yang tersedia."""
        if self._timeseries_df is None:
            raise RuntimeError("Panggil .fit() terlebih dahulu.")
        return self._categories

    # ------------------------------------------------------------------
    # Langkah-langkah individual (dapat dipanggil langsung untuk testing)
    # ------------------------------------------------------------------

    def validate_schema(
        self, df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Periksa keberadaan kolom wajib di DataFrame.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        (bool, list) : (True jika valid, daftar kolom yang hilang)
        """
        # Nama kolom dataset mentah selalu dalam huruf besar (dengan/tanpa kutip)
        existing = [c.strip("'").strip() for c in df.columns.tolist()]
        missing = [c for c in REQUIRED_COLUMNS if c not in existing]
        return (len(missing) == 0, missing)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitasi dataset mentah POS:
        - Hapus tanda kutip tunggal dari nilai string.
        - Filter RTYPE == 'J' (transaksi jual riil).
        - Konversi QTY ke numerik; filter QTY > 0.
        - Drop baris null pada kolom wajib.

        Parameters
        ----------
        df : pd.DataFrame  Dataset mentah.

        Returns
        -------
        pd.DataFrame  Dataset yang sudah bersih.
        """
        df = df.copy()

        # Sanitasi tanda kutip dari semua kolom string
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip("'").str.strip()

        # Filter transaksi riil
        df = df[df["RTYPE"] == VALID_RTYPE].copy()

        # Konversi QTY ke numerik
        df["QTY"] = pd.to_numeric(df["QTY"], errors="coerce")

        # Buang baris kosong & kuantitas non-positif (ringkasan, retur QTY<=0)
        df = df[df["QTY"] > 0].copy()
        df = df.dropna(subset=["QTY", "DIV", "CAT_COD"]).copy()

        # Buang baris dengan kategori kosong
        df = df[df["DIV"] != ""].copy()

        logger.debug("clean_data: %d baris tersisa setelah filter", len(df))
        return df.reset_index(drop=True)

    def parse_dates(
        self,
        df: pd.DataFrame,
        date_col: str = "TANGGAL",
    ) -> pd.DataFrame:
        """
        Konversi kolom tanggal string 'DD-MM-YYYY' ke tipe datetime.

        Parameters
        ----------
        df : pd.DataFrame
        date_col : str   Nama kolom tanggal mentah.

        Returns
        -------
        pd.DataFrame  DataFrame dengan kolom 'ds' bertipe datetime64.
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df[date_col], format=DATE_FORMAT, errors="coerce")
        # Buang baris yang gagal diparse
        n_failed = df["ds"].isna().sum()
        if n_failed > 0:
            logger.warning(
                "%d baris gagal parse tanggal dan akan dibuang.", n_failed
            )
            df = df.dropna(subset=["ds"])
        return df.reset_index(drop=True)

    def resample_time_series(
        self,
        df: pd.DataFrame,
        category_col: str = "DIV",
        freq: str | None = None,
    ) -> pd.DataFrame:
        """
        Agregasi kuantitas penjualan harian ke interval waktu berkala
        per kategori produk.

        Parameters
        ----------
        df : pd.DataFrame   DataFrame bersih dengan kolom 'ds' dan 'QTY'.
        category_col : str  Kolom kategori ('DIV' atau 'CAT_COD').
        freq : str          Override frekuensi ('W-MON' atau 'ME').

        Returns
        -------
        pd.DataFrame  Kolom: ['category', 'ds', 'y']
        """
        freq = freq or self.freq
        df = df.copy()

        agg_df = (
            df.groupby(
                [category_col, pd.Grouper(key="ds", freq=freq)]
            )["QTY"]
            .sum()
            .reset_index()
        )

        # Ganti nama kolom ke standar internal
        agg_df.rename(
            columns={category_col: "category", "QTY": "y"},
            inplace=True,
        )
        agg_df = agg_df.sort_values(["category", "ds"]).reset_index(drop=True)

        # Pastikan tidak ada nilai negatif hasil resample
        agg_df["y"] = agg_df["y"].clip(lower=0)

        logger.debug(
            "resample_time_series (freq=%s): %d baris", freq, len(agg_df)
        )
        return agg_df

    def extract_categories(
        self,
        df: pd.DataFrame,
        category_col: str = "category",
        min_points: int = MIN_DATA_POINTS,
    ) -> List[str]:
        """
        Ekstrak daftar kode kategori unik yang memiliki data historis
        mencukupi (>= min_points observasi).

        Parameters
        ----------
        df : pd.DataFrame
        category_col : str
        min_points : int

        Returns
        -------
        List[str]  Daftar kode kategori terurut, disaring berdasarkan
                   kecukupan data.
        """
        counts = df.groupby(category_col).size()
        valid_cats = sorted(counts[counts >= min_points].index.tolist())
        logger.debug(
            "%d dari %d kategori memenuhi syarat minimal %d titik data.",
            len(valid_cats),
            len(counts),
            min_points,
        )
        return valid_cats

    def get_data_summary(self) -> dict:
        """
        Menghasilkan ringkasan statistik dataset untuk ditampilkan di dasbor.

        Returns
        -------
        dict  Ringkasan: total_baris, rentang_tanggal, total_kategori,
              rata_qty, nama_file.
        """
        if self._clean_df is None:
            raise RuntimeError("Panggil .fit() terlebih dahulu.")
        df = self._clean_df
        return {
            "total_baris_bersih": len(df),
            "tanggal_awal": df["ds"].min() if "ds" in df.columns else None,
            "tanggal_akhir": df["ds"].max() if "ds" in df.columns else None,
            "total_kategori": len(self._categories),
            "total_qty": int(df["QTY"].sum()),
            "rata_qty_per_transaksi": round(float(df["QTY"].mean()), 2),
            "frekuensi_resample": self.freq,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(source: str | io.BytesIO | io.StringIO) -> pd.DataFrame:
        """
        Baca berkas CSV dengan seleksi kolom untuk efisiensi RAM.
        Secara otomatis menangani path string maupun file-like object
        (BytesIO dari Streamlit file_uploader).
        """
        # Tentukan kolom yang akan dimuat (hanya yang ada di dataset)
        # —menggunakan usecols dengan callable untuk toleransi kolom hilang
        def _usecols(col: str) -> bool:
            return col.strip("'").strip() in LOAD_COLUMNS

        try:
            df = pd.read_csv(
                source,
                usecols=_usecols,
                low_memory=False,
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            # Fallback: coba encoding cp1252 untuk file Windows
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(
                source,
                usecols=_usecols,
                low_memory=False,
                encoding="cp1252",
            )

        logger.info("Berkas CSV dibaca: %d baris, %d kolom", len(df), len(df.columns))
        return df
