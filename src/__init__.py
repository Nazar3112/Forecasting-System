"""
__init__.py
===========
Package marker untuk direktori src/.
Mengekspor kelas-kelas utama. Impor hybrid_engine dilakukan secara
lazy agar package bisa di-load walaupun prophet/lightgbm belum terinstall.

Contoh penggunaan:
    from src import DataPreprocessor, HybridForecastingEngine
"""

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON,
    REQUIRED_COLUMNS,
    FREQ_OPTIONS,
    SEASONALITY_MODES,
)
from src.preprocessor  import DataPreprocessor
from src.filter_engine import DataFilterEngine
from src.evaluator     import ModelEvaluator

# Lazy import HybridForecastingEngine & DashboardVisualizer
# (bergantung pada prophet, lightgbm, plotly, streamlit)
def __getattr__(name):
    if name == "HybridForecastingEngine":
        from src.hybrid_engine import HybridForecastingEngine
        return HybridForecastingEngine
    if name == "DashboardVisualizer":
        from src.visualizer import DashboardVisualizer
        return DashboardVisualizer
    raise AttributeError(f"module 'src' has no attribute '{name}'")

__all__ = [
    # Konfigurasi
    "APP_TITLE",
    "APP_SUBTITLE",
    "APP_ICON",
    "REQUIRED_COLUMNS",
    "FREQ_OPTIONS",
    "SEASONALITY_MODES",
    # Kelas inti
    "DataPreprocessor",
    "DataFilterEngine",
    "HybridForecastingEngine",
    "ModelEvaluator",
    "DashboardVisualizer",
]
