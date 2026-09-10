"""
visualizer.py
=============
Kelas DashboardVisualizer: konstruksi grafik interaktif Plotly,
rendering kartu metrik Streamlit, tabel analitik, dan ekspor
laporan ke CSV byte-stream UTF-8 tanpa menyentuh disk.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COLOR_ACTUAL,
    COLOR_PROPHET,
    COLOR_HYBRID,
    COLOR_FUTURE,
    COLOR_RESIDUAL,
)

# Airtable design palette overrides for charts
# (selaras dengan DESIGN-airtable.md signature card colors)
_C_ACTUAL   = "#181d26"   # ink — data aktual historis
_C_PROPHET  = "#d9a441"   # signature-mustard — Prophet tunggal
_C_HYBRID   = "#0a6b2d"   # hijau gelap selaras signature-forest — Hybrid
_C_FUTURE   = "#aa2d00"   # signature-coral — proyeksi masa depan
_C_RESIDUAL = "#254fad"   # info blue — galat/residual


logger = logging.getLogger(__name__)


class DashboardVisualizer:
    """
    Bertanggung jawab atas seluruh lapisan presentasi visual dasbor.
    Semua rendering beroperasi secara in-memory — tidak ada I/O disk.
    """

    # ------------------------------------------------------------------
    # KPI Metric Cards
    # ------------------------------------------------------------------

    @staticmethod
    def render_kpi_cards(
        total_sales:  int,
        mape_val:     float,
        rmse_val:     float,
        mae_val:      float,
        trend_pct:    float,
        mape_label:   str,
        mape_emoji:   str,
    ) -> None:
        """
        Render 4 kartu KPI utama di baris atas dasbor menggunakan
        st.metric() Streamlit.

        Parameters
        ----------
        total_sales : int    Total unit penjualan historis.
        mape_val    : float  MAPE model hibrida (%).
        rmse_val    : float  RMSE model hibrida (unit).
        mae_val     : float  MAE model hibrida (unit).
        trend_pct   : float  Persentase perubahan tren penjualan.
        mape_label  : str    Label interpretasi kualitas ('Sangat Baik', dll).
        mape_emoji  : str    Emoji warna status model.
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📦 Total Penjualan Historis",
                value=f"{total_sales:,} unit",
                delta=f"{trend_pct:+.1f}% tren",
                help="Jumlah kumulatif kuantitas terjual seluruh periode historis.",
            )

        with col2:
            st.metric(
                label=f"{mape_emoji} MAPE Model Hibrida",
                value=f"{mape_val:.2f}%",
                delta=mape_label,
                delta_color="off",
                help="Mean Absolute Percentage Error pada data uji. < 10% = Sangat Baik.",
            )

        with col3:
            st.metric(
                label="📐 RMSE",
                value=f"{rmse_val:,.2f}",
                help="Root Mean Squared Error dalam satuan unit barang.",
            )

        with col4:
            st.metric(
                label="📏 MAE",
                value=f"{mae_val:,.2f}",
                help="Mean Absolute Error rata-rata galat prediksi per periode.",
            )

    # ------------------------------------------------------------------
    # Grafik Utama
    # ------------------------------------------------------------------

    @staticmethod
    def plot_forecast(
        historical_df:    pd.DataFrame,
        test_results:     pd.DataFrame,
        future_forecast:  pd.DataFrame,
        title:            str = "Peramalan Penjualan",
        category_label:   str = "",
        freq_label:       str = "Mingguan",
        show_prophet:     bool = True,
    ) -> go.Figure:
        """
        Bangun grafik garis interaktif Plotly multi-trace.

        Traces:
          1. Data Aktual Historis (biru)
          2. Prediksi Prophet tunggal – test set (kuning, opsional)
          3. Prediksi Hibrida – test set (hijau)
          4. Proyeksi Masa Depan Hibrida (ungu, garis putus-putus)

        Parameters
        ----------
        historical_df   : ['ds', 'y']           – data aktual historis.
        test_results    : ['ds', 'y', 'y_prophet', 'y_hybrid'] – hasil uji.
        future_forecast : ['ds', 'y_hybrid']    – proyeksi masa depan.
        title           : Judul grafik.
        category_label  : Kode/nama kategori untuk subtitle.
        freq_label      : Label frekuensi ('Mingguan' / 'Bulanan').
        show_prophet    : Tampilkan kurva Prophet tunggal (mode perbandingan).

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()

        # ── Trace 1: Aktual Historis ─────────────────────────────────
        fig.add_trace(go.Scatter(
            x=historical_df["ds"],
            y=historical_df["y"],
            name="Penjualan Aktual",
            mode="lines+markers",
            line=dict(color=_C_ACTUAL, width=2),
            marker=dict(size=4, color=_C_ACTUAL),
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Penjualan Aktual: <b>%{y:,.0f}</b> unit<extra></extra>"
            ),
        ))

        # ── Trace 2: Prediksi Prophet Tunggal (data uji) ─────────────
        if show_prophet and not test_results.empty and "y_prophet" in test_results.columns:
            fig.add_trace(go.Scatter(
                x=test_results["ds"],
                y=test_results["y_prophet"],
                name="Prediksi Prophet (Data Uji)",
                mode="lines+markers",
                line=dict(color=_C_PROPHET, width=1.8, dash="dot"),
                marker=dict(size=5, symbol="diamond", color=_C_PROPHET),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Prophet: <b>%{y:,.0f}</b> unit<extra></extra>"
                ),
            ))

        # ── Trace 3: Prediksi Hibrida (data uji) ─────────────────────
        if not test_results.empty and "y_hybrid" in test_results.columns:
            fig.add_trace(go.Scatter(
                x=test_results["ds"],
                y=test_results["y_hybrid"],
                name="Prediksi Hibrida (Data Uji)",
                mode="lines+markers",
                line=dict(color=_C_HYBRID, width=2.5),
                marker=dict(size=6, symbol="circle", color=_C_HYBRID),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Hibrida: <b>%{y:,.0f}</b> unit<extra></extra>"
                ),
            ))

        # ── Trace 4: Proyeksi Masa Depan ─────────────────────────────
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(
                x=future_forecast["ds"],
                y=future_forecast["y_hybrid"],
                name="Proyeksi Masa Depan",
                mode="lines+markers",
                line=dict(color=_C_FUTURE, width=2.5, dash="dash"),
                marker=dict(size=7, symbol="star", color=_C_FUTURE),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Proyeksi: <b>%{y:,.0f}</b> unit<extra></extra>"
                ),
            ))

        # ── Area Konfidensial (test zone) ────────────────────────────
        if not test_results.empty:
            fig.add_vrect(
                x0=test_results["ds"].min(),
                x1=test_results["ds"].max(),
                fillcolor="rgba(24, 29, 38, 0.04)",
                line_width=0,
                annotation_text="Zona Uji",
                annotation_position="top left",
                annotation=dict(font=dict(size=11, color="#41454d", family="Inter, sans-serif")),
            )

        # ── Garis Pemisah: Historis / Future ─────────────────────────
        if not future_forecast.empty and not historical_df.empty:
            split_date = historical_df["ds"].max()
            fig.add_vline(
                x=split_date,
                line_dash="dot",
                line_color="#9297a0",
                annotation_text="Mulai Proyeksi",
                annotation_position="top right",
                annotation=dict(font=dict(size=11, color="#41454d", family="Inter, sans-serif")),
            )

        # ── Layout ───────────────────────────────────────────────────
        subtitle = f"Divisi/Kategori: {category_label}" if category_label else ""
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:#41454d;font-weight:400'>{subtitle}</span>" if subtitle else ""),
                font=dict(size=16, color="#181d26", family="Inter, sans-serif"),
            ),
            xaxis=dict(
                title=f"Periode ({freq_label})",
                gridcolor="#dddddd",
                linecolor="#dddddd",
                tickfont=dict(family="Inter, sans-serif", size=11, color="#41454d"),
                titlefont=dict(family="Inter, sans-serif", size=12, color="#181d26"),
            ),
            yaxis=dict(
                title="Kuantitas Penjualan (unit)",
                gridcolor="#dddddd",
                linecolor="#dddddd",
                tickfont=dict(family="Inter, sans-serif", size=11, color="#41454d"),
                titlefont=dict(family="Inter, sans-serif", size=12, color="#181d26"),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Inter, sans-serif", size=12, color="#181d26"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#dddddd",
                borderwidth=1,
            ),
            hovermode="x unified",
            height=480,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter, system-ui, sans-serif", size=12),
            margin=dict(l=40, r=20, t=80, b=40),
        )
        return fig

    @staticmethod
    def plot_residual_analysis(
        test_results:  pd.DataFrame,
        category_label: str = "",
    ) -> go.Figure:
        """
        Visualisasi analisis residual: bandingkan galat Prophet vs Hibrida
        untuk membuktikan efektivitas koreksi LightGBM.

        Returns
        -------
        go.Figure  Bar chart residual absolut per periode.
        """
        fig = go.Figure()

        if test_results.empty or "y" not in test_results.columns:
            return fig

        y_true = test_results["y"]
        err_prophet = (y_true - test_results.get("y_prophet", y_true)).abs()
        err_hybrid  = (y_true - test_results.get("y_hybrid",  y_true)).abs()

        fig.add_trace(go.Bar(
            x=test_results["ds"],
            y=err_prophet,
            name="|Galat Prophet|",
            marker_color=_C_PROPHET,
            opacity=0.85,
        ))

        fig.add_trace(go.Bar(
            x=test_results["ds"],
            y=err_hybrid,
            name="|Galat Hibrida|",
            marker_color=_C_HYBRID,
            opacity=0.9,
        ))

        subtitle = f"Kategori: {category_label}" if category_label else ""
        fig.update_layout(
            title=dict(
                text=f"<b>Analisis Galat Absolut pada Data Uji</b>" + (f"<br><span style='font-size:12px;color:#41454d;font-weight:400'>{subtitle}</span>" if subtitle else ""),
                font=dict(size=15, color="#181d26", family="Inter, sans-serif"),
            ),
            xaxis=dict(
                title="Periode",
                gridcolor="#dddddd",
                linecolor="#dddddd",
                tickfont=dict(family="Inter, sans-serif", size=11, color="#41454d"),
                titlefont=dict(family="Inter, sans-serif", size=12, color="#181d26"),
            ),
            yaxis=dict(
                title="Galat Absolut (unit)",
                gridcolor="#dddddd",
                linecolor="#dddddd",
                tickfont=dict(family="Inter, sans-serif", size=11, color="#41454d"),
                titlefont=dict(family="Inter, sans-serif", size=12, color="#181d26"),
            ),
            barmode="group",
            height=360,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter, system-ui, sans-serif", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Inter, sans-serif", size=12, color="#181d26"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#dddddd",
                borderwidth=1,
            ),
            hovermode="x unified",
            margin=dict(l=40, r=20, t=70, b=40),
        )
        return fig

    @staticmethod
    def plot_feature_importance(
        fi_df: pd.DataFrame,
        top_n: int = 15,
    ) -> go.Figure:
        """
        Horizontal bar chart LightGBM feature importance (top N fitur).

        Parameters
        ----------
        fi_df : pd.DataFrame  Kolom ['feature', 'importance'].
        top_n : int            Tampilkan N fitur teratas.

        Returns
        -------
        go.Figure
        """
        top_df = fi_df.head(top_n).sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top_df["importance"],
            y=top_df["feature"],
            orientation="h",
            marker_color=_C_RESIDUAL,
        ))
        fig.update_layout(
            title=dict(
                text="<b>Pentingnya Fitur LightGBM (Feature Importance)</b>",
                font=dict(size=15, color="#181d26", family="Inter, sans-serif"),
            ),
            xaxis=dict(
                title="Importance (Gain)",
                gridcolor="#dddddd",
                linecolor="#dddddd",
                tickfont=dict(family="Inter, sans-serif", size=11, color="#41454d"),
                titlefont=dict(family="Inter, sans-serif", size=12, color="#181d26"),
            ),
            yaxis=dict(
                title="Fitur",
                tickfont=dict(family="Inter, sans-serif", size=11, color="#41454d"),
            ),
            height=max(300, top_n * 25),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter, system-ui, sans-serif", size=12),
            margin=dict(l=150, r=20, t=60, b=40),
        )
        return fig

    # ------------------------------------------------------------------
    # Tabel Ringkasan & Ekspor
    # ------------------------------------------------------------------

    @staticmethod
    def render_forecast_table(
        future_forecast: pd.DataFrame,
        freq_label:      str = "Mingguan",
    ) -> None:
        """
        Tampilkan tabel proyeksi masa depan yang terformat di dasbor.
        """
        if future_forecast.empty:
            st.info("Belum ada data proyeksi.")
            return

        display_df = future_forecast[["ds", "y_hybrid"]].copy()
        display_df.columns = ["Periode", "Proyeksi Penjualan (unit)"]
        display_df["Periode"] = display_df["Periode"].dt.strftime("%d %b %Y")
        display_df["Proyeksi Penjualan (unit)"] = display_df[
            "Proyeksi Penjualan (unit)"
        ].round(0).astype(int)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    @staticmethod
    def render_comparison_table(comp_df: pd.DataFrame) -> None:
        """
        Tampilkan tabel komparasi performa model di dasbor.
        """
        if comp_df.empty:
            return

        # Format numerik
        styled_df = comp_df.copy()
        for col in ["MAE", "MAPE_%", "RMSE"]:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(
                    lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x
                )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    @staticmethod
    def export_csv_bytes(
        future_forecast: pd.DataFrame,
        category_label:  str = "",
    ) -> bytes:
        """
        Konversi DataFrame proyeksi masa depan ke byte stream CSV UTF-8.
        Digunakan langsung oleh st.download_button tanpa menyentuh disk.

        Parameters
        ----------
        future_forecast : pd.DataFrame  Kolom ['ds', 'y_hybrid', ...].
        category_label  : str           Label kategori untuk header CSV.

        Returns
        -------
        bytes  Konten CSV ter-encode UTF-8.
        """
        export_df = future_forecast.copy()

        # Format tanggal menjadi human-readable
        if "ds" in export_df.columns:
            export_df["Periode"] = export_df["ds"].dt.strftime("%d-%m-%Y")
            export_df.drop(columns=["ds"], inplace=True)

        # Ganti nama kolom ke Indonesia
        rename_map = {
            "y_hybrid":          "Proyeksi_Penjualan_Unit",
            "y_prophet":         "Estimasi_Prophet_Unit",
            "y_residual_lgbm":   "Koreksi_LightGBM_Unit",
        }
        export_df.rename(columns=rename_map, inplace=True)

        # Tambahkan metadata header
        buffer = io.StringIO()
        buffer.write(f"# Laporan Proyeksi Penjualan Retail\n")
        buffer.write(f"# Kategori/Divisi : {category_label}\n")
        buffer.write(f"# Model           : Hibrida Prophet + LightGBM\n")
        buffer.write(f"# Dibuat oleh     : Sistem Prediksi Tren Penjualan - Indomarco Prismatama\n")
        buffer.write("#\n")

        export_df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_trend_pct(series_df: pd.DataFrame, window: int = 4) -> float:
        """
        Hitung persentase perubahan tren penjualan:
        membandingkan rata-rata `window` periode terakhir
        terhadap rata-rata `window` periode sebelumnya.

        Returns
        -------
        float  Perubahan relatif (%) positif = naik, negatif = turun.
        """
        y = series_df["y"].values
        if len(y) < window * 2:
            return 0.0
        recent   = y[-window:]
        previous = y[-window * 2:-window]
        avg_recent   = np.mean(recent)
        avg_previous = np.mean(previous)
        if avg_previous == 0:
            return 0.0
        return round((avg_recent - avg_previous) / avg_previous * 100, 2)
