# Sistem Prediksi Tren Penjualan Retail – Indomarco Prismatama
### Model Hibrida Prophet + LightGBM | Streamlit In-Memory Dashboard

> **Penelitian Skripsi S1 Teknik Informatika**  
> **Alfiyan Nazar (220511053)** | Universitas Muhammadiyah Cirebon, 2026

---

## 📋 Deskripsi Sistem

Sistem analitik web berbasis **Streamlit** yang memproses data log transaksi kasir POS (`.csv`) untuk menghasilkan proyeksi penjualan per kategori produk menggunakan **Model Hibrida Facebook Prophet + LightGBM**.

**Arsitektur**: Stateless In-Memory — tanpa database relasional, tanpa login.  
**Desain UI**: Binance Design System (`DESIGN-binance.md`) — dark canvas finansial dengan aksen kuning Binance.

📖 **Panduan Pengguna & Bahan Presentasi**: Baca [PANDUAN_PENGGUNAAN.md](PANDUAN_PENGGUNAAN.md) untuk petunjuk langkah demi langkah, penjelasan fitur awam, dan naskah demo sidang.

### Alur Model Hibrida:
```
y(t) = ŷ_Prophet(t) + ê_LightGBM(t)
```
1. **Prophet** → dekomposisi tren + musiman (aditif / multiplikatif)
2. **Residual** → `e(t) = y_aktual − ŷ_Prophet`
3. **LightGBM** → mempelajari pola non-linier dari galat residual
4. **Sintesis** → `ŷ_hybrid = max(0, ŷ_Prophet + ê_LightGBM)`

---

## 🗂️ Struktur Folder

```
Forecasting System/
├── app.py                    # Streamlit Dashboard (Controller)
├── requirements.txt          # Dependensi pustaka Python
├── README.md                 # Dokumentasi teknis sistem
├── PANDUAN_PENGGUNAAN.md     # Panduan lengkap pengguna & materi presentasi
├── DESIGN-binance.md         # Spesifikasi Binance Design System
├── src/
│   ├── __init__.py
│   ├── config.py             # Parameter default & skema kolom
│   ├── preprocessor.py       # DataPreprocessor (ingesti, sanitasi, resample)
│   ├── filter_engine.py      # DataFilterEngine (filter kategori, validasi)
│   ├── hybrid_engine.py      # HybridForecastingEngine (Prophet + LightGBM)
│   ├── evaluator.py          # ModelEvaluator (MAE, MAPE, RMSE)
│   └── visualizer.py         # DashboardVisualizer (Plotly, CSV export)
└── tests/
    ├── test_preprocessor.py  # Unit tests DataPreprocessor
    ├── test_evaluator.py     # Unit tests ModelEvaluator
    └── test_hybrid_engine.py # Unit tests HybridForecastingEngine
```

---

## 🛠️ Instalasi & Setup

### Prasyarat
- Python >= 3.10 (diuji pada Python 3.12)
- pip

### Langkah Instalasi

```bash
# 1. Clone atau buka folder proyek
cd "Forecasting System"

# 2. (Opsional) Buat virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install semua dependensi
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📋 Format Berkas CSV Input

Berkas log kasir POS harus memuat kolom berikut (nama kolom **sensitif huruf besar**):

| Kolom | Format | Keterangan |
| :--- | :--- | :--- |
| `TANGGAL` | `'DD-MM-YYYY'` | Tanggal transaksi |
| `RTYPE` | `'J'` / `'D'` | Tipe rekaman (`J` = transaksi jual riil) |
| `DIV` | String 2 digit | Kode Divisi produk |
| `CAT_COD` | String 6 digit | Kode Kategori produk |
| `QTY` | Numerik ≥ 0 | Kuantitas barang terjual |

> Nilai string dibungkus tanda kutip tunggal (`'...'`) — sistem secara otomatis membersihkannya.

---

## 🧪 Menjalankan Unit Tests

```bash
# Install pytest jika belum ada
pip install pytest

# Jalankan semua test
pytest tests/ -v

# Atau per modul
pytest tests/test_preprocessor.py -v
pytest tests/test_evaluator.py -v
pytest tests/test_hybrid_engine.py -v   # Perlu prophet + lightgbm
```

---

## 📊 Metrik Evaluasi Akurasi

| Metrik | Formula | Interpretasi |
| :--- | :--- | :--- |
| **MAE** | `(1/n) Σ\|y - ŷ\|` | Rata-rata galat absolut (unit barang) |
| **MAPE** | `(1/n) Σ\|y-ŷ\|/y × 100%` | Persentase galat relatif |
| **RMSE** | `√((1/n) Σ(y-ŷ)²)` | Galat kuadrat — sensitif terhadap outlier |

**Interpretasi MAPE:**
- 🟢 < 10% → Sangat Baik
- 🔵 10–20% → Baik
- 🟡 20–50% → Layak
- 🔴 > 50% → Kurang

---

## 📚 Dependensi Utama

| Pustaka | Versi | Fungsi |
| :--- | :--- | :--- |
| `streamlit` | ≥1.35 | Framework aplikasi web interaktif |
| `prophet` | ≥1.1.5 | Dekomposisi deret waktu (tren + musiman) |
| `lightgbm` | ≥4.3 | Gradient boosting untuk koreksi residual |
| `pandas` | ≥2.2 | Manipulasi data tabular & resampling |
| `plotly` | ≥5.20 | Visualisasi grafik interaktif |
| `scikit-learn` | ≥1.4 | Utilitas metrik evaluasi |
| `numpy` | ≥1.26 | Komputasi numerik |

---

## 👤 Tentang

**Alfiyan Nazar** – 220511053  
Program Studi Teknik Informatika, Fakultas Teknik  
Universitas Muhammadiyah Cirebon, 2026  

Penelitian: *Rancang Bangun Sistem Prediksi Tren Penjualan Produk Retail Berdasarkan Kategori Menggunakan Model Hibrida Prophet dan LightGBM (Studi Kasus: PT. Indomarco Prismatama)*
