"""
config.py
=========
Konfigurasi global sistem peramalan penjualan retail.
Memuat skema kolom wajib dataset POS (log kasir Indomarco),
parameter default model, dan konstanta sistem.
"""

# ---------------------------------------------------------------------------
# SKEMA KOLOM DATASET POS (merged_MTRAN_with_desc.csv)
# ---------------------------------------------------------------------------

# Kolom minimal yang wajib ada di berkas CSV input
REQUIRED_COLUMNS: list[str] = ["TANGGAL", "RTYPE", "DIV", "CAT_COD", "QTY"]

# Kolom tambahan yang diambil jika tersedia (untuk pratinjau UI)
OPTIONAL_COLUMNS: list[str] = ["DESC2", "PRDCD", "PRICE", "GROSS"]

# Semua kolom yang digunakan saat membaca CSV (menekan RAM: usecols)
LOAD_COLUMNS: list[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Mapping kolom ke nama standar internal sistem
COLUMN_MAP: dict[str, str] = {
    "TANGGAL": "tanggal_raw",  # String 'DD-MM-YYYY' sebelum di-parse
    "RTYPE":   "rtype",        # Tipe rekaman ('J' = Jual, 'D' = Draft)
    "DIV":     "div",          # Kode Divisi (2 digit, e.g. '04')
    "CAT_COD": "cat_cod",      # Kode Kategori (6 digit, e.g. '010414')
    "QTY":     "qty",          # Kuantitas barang terjual
    "DESC2":   "desc2",        # Deskripsi nama produk
    "PRDCD":   "prdcd",        # Kode PLU / SKU barang
    "PRICE":   "price",        # Harga satuan
    "GROSS":   "gross",        # Nilai transaksi kotor
}

# Filter nilai valid untuk kolom RTYPE (transaksi penjualan riil)
VALID_RTYPE: str = "J"

# Format tanggal mentah dari log kasir
DATE_FORMAT: str = "%d-%m-%Y"

# ---------------------------------------------------------------------------
# PARAMETER AGREGASI DERET WAKTU
# ---------------------------------------------------------------------------

# Pilihan frekuensi resample (alias pandas)
FREQ_OPTIONS: dict[str, str] = {
    "Mingguan (W)": "W-MON",
    "Bulanan (M)":  "ME",
}

DEFAULT_FREQ: str = "W-MON"

# Ambang batas minimal observasi agar analisis musiman valid
MIN_DATA_POINTS: int = 24

# ---------------------------------------------------------------------------
# PARAMETER TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

DEFAULT_TEST_SIZE: float = 0.20  # Porsi 20% data terakhir sebagai data uji

# ---------------------------------------------------------------------------
# PARAMETER PERAMALAN MASA DEPAN
# ---------------------------------------------------------------------------

DEFAULT_HORIZON: int = 8   # 8 periode ke depan (default)
MIN_HORIZON:     int = 1
MAX_HORIZON:     int = 52  # Maks 52 periode (setahun penuh jika mingguan)

# ---------------------------------------------------------------------------
# PARAMETER MODEL PROPHET
# ---------------------------------------------------------------------------

PROPHET_PARAMS: dict = {
    "yearly_seasonality":  True,
    "weekly_seasonality":  True,
    "daily_seasonality":   False,  # Data mingguan/bulanan — harian dimatikan
    "seasonality_mode":    "additive",  # bisa di-override: 'multiplicative'
    "interval_width":      0.80,       # Lebar interval ketidakpastian 80%
    "changepoint_prior_scale": 0.05,   # Sensitivitas deteksi tren
}

# Opsi mode dekomposisi musiman Prophet
SEASONALITY_MODES: list[str] = ["additive", "multiplicative"]

# ---------------------------------------------------------------------------
# PARAMETER MODEL LIGHTGBM
# ---------------------------------------------------------------------------

LIGHTGBM_PARAMS: dict = {
    "n_estimators":    200,
    "learning_rate":   0.05,
    "num_leaves":      15,
    "max_depth":       5,
    "min_child_samples": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "random_state":    42,
    "verbosity":       -1,   # Matikan log LightGBM
}

# Kelambatan (lag) fitur residual untuk melatih LightGBM
RESIDUAL_LAG_STEPS: list[int] = [1, 2, 3, 4]

# Jendela rolling statistics (dalam unit periode)
ROLLING_WINDOWS: list[int] = [4, 8]

# ---------------------------------------------------------------------------
# PARAMETER UI STREAMLIT
# ---------------------------------------------------------------------------

APP_TITLE:    str = "Sistem Prediksi Tren Penjualan Retail – Indomarco Prismatama"
APP_SUBTITLE: str = "Model Hibrida Prophet + LightGBM | In-Memory Analytics Dashboard"
APP_ICON:     str = "📦"

# Warna brand untuk grafik Plotly
COLOR_ACTUAL:   str = "#3B82F6"   # Biru  – data aktual historis
COLOR_PROPHET:  str = "#F59E0B"   # Kuning – prediksi Prophet tunggal
COLOR_HYBRID:   str = "#10B981"   # Hijau  – prediksi hibrida
COLOR_FUTURE:   str = "#6366F1"   # Ungu  – proyeksi masa depan
COLOR_RESIDUAL: str = "#EF4444"   # Merah  – galat residual

# Interpretasi nilai MAPE (%)
MAPE_THRESHOLDS: dict[str, tuple] = {
    "Sangat Baik": (0.0,  10.0),
    "Baik":        (10.0, 20.0),
    "Layak":       (20.0, 50.0),
    "Kurang":      (50.0, float("inf")),
}
