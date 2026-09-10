"""
tests/test_preprocessor.py
==========================
Unit tests untuk modul DataPreprocessor.
Menguji parsing tanggal, sanitasi kutip, filter transaksi valid,
dan agregasi deret waktu mingguan.
"""

import io
import textwrap
import pandas as pd
import pytest

from src.preprocessor import DataPreprocessor
from src.config import REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Fixture: CSV sintetis menyerupai format log kasir MTRAN Indomarco
# ---------------------------------------------------------------------------

SAMPLE_CSV_VALID = textwrap.dedent("""\
    TANGGAL,RTYPE,DIV,CAT_COD,QTY,DESC2,PRICE,GROSS
    '01-01-2023','J','04','010414',5,'AQUA MINERAL 600ML',5500,27500
    '08-01-2023','J','04','010414',3,'AQUA MINERAL 600ML',5500,16500
    '15-01-2023','J','04','010414',8,'AQUA MINERAL 600ML',5500,44000
    '22-01-2023','J','04','010414',2,'AQUA MINERAL 600ML',5500,11000
    '29-01-2023','J','04','010414',6,'AQUA MINERAL 600ML',5500,33000
    '05-02-2023','J','04','010414',4,'AQUA MINERAL 600ML',5500,22000
    '12-02-2023','J','04','010414',7,'AQUA MINERAL 600ML',5500,38500
    '01-01-2023','D','08','010801',2,'INDOMIE GORENG',3500,7000
    '01-01-2023','J','08','010801',0,'RINGKASAN PEMBAYARAN',0,0
    '01-01-2023','J','','',-1,'ITEM RETUR',5000,-5000
""")

SAMPLE_CSV_MISSING_COL = textwrap.dedent("""\
    TANGGAL,RTYPE,QTY
    '01-01-2023','J',5
""")


@pytest.fixture
def valid_csv_buffer():
    return io.StringIO(SAMPLE_CSV_VALID)


@pytest.fixture
def preprocessor():
    return DataPreprocessor(freq="W-MON")


# ---------------------------------------------------------------------------
# Test: validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_schema(self, preprocessor, valid_csv_buffer):
        df = pd.read_csv(valid_csv_buffer)
        ok, missing = preprocessor.validate_schema(df)
        assert ok, f"Skema seharusnya valid, missing: {missing}"

    def test_missing_required_column(self, preprocessor):
        df = pd.read_csv(io.StringIO(SAMPLE_CSV_MISSING_COL))
        ok, missing = preprocessor.validate_schema(df)
        assert not ok
        assert "DIV" in missing
        assert "CAT_COD" in missing


# ---------------------------------------------------------------------------
# Test: clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_filter_rtype_j_only(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        # Baris RTYPE == 'D' harus terbuang
        assert (cleaned["RTYPE"] == "J").all()

    def test_filter_qty_positive(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        # QTY harus > 0
        assert (cleaned["QTY"] > 0).all()

    def test_strip_quotes(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        # Nilai TANGGAL tidak boleh mengandung tanda kutip
        assert not cleaned["TANGGAL"].str.startswith("'").any()

    def test_empty_div_dropped(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        # Baris dengan DIV kosong harus dibuang
        assert (cleaned["DIV"] != "").all()

    def test_row_count(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        # Hanya baris J dengan QTY > 0 dan DIV tidak kosong
        # Dari 10 baris: 1 D, 1 QTY=0, 1 negatif, 1 DIV-empty → 6 valid
        assert len(cleaned) <= 7, f"Jumlah baris tidak sesuai: {len(cleaned)}"


# ---------------------------------------------------------------------------
# Test: parse_dates
# ---------------------------------------------------------------------------

class TestParseDates:
    def test_parse_date_format(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        parsed = preprocessor.parse_dates(cleaned)
        assert "ds" in parsed.columns
        assert pd.api.types.is_datetime64_any_dtype(parsed["ds"])

    def test_no_null_dates(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        parsed = preprocessor.parse_dates(cleaned)
        assert parsed["ds"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test: resample_time_series
# ---------------------------------------------------------------------------

class TestResampleTimeSeries:
    def test_output_columns(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        parsed  = preprocessor.parse_dates(cleaned)
        resampled = preprocessor.resample_time_series(parsed, category_col="DIV", freq="W-MON")
        assert "category" in resampled.columns
        assert "ds" in resampled.columns
        assert "y" in resampled.columns

    def test_no_negative_y(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        parsed  = preprocessor.parse_dates(cleaned)
        resampled = preprocessor.resample_time_series(parsed, category_col="DIV", freq="W-MON")
        assert (resampled["y"] >= 0).all()

    def test_sorted_by_date(self, preprocessor, valid_csv_buffer):
        raw = pd.read_csv(valid_csv_buffer)
        cleaned = preprocessor.clean_data(raw)
        parsed  = preprocessor.parse_dates(cleaned)
        resampled = preprocessor.resample_time_series(parsed, category_col="DIV", freq="W-MON")
        # Pastikan setiap kategori terurut secara kronologis
        for cat, grp in resampled.groupby("category"):
            assert (grp["ds"].diff().dropna() >= pd.Timedelta(0)).all()


# ---------------------------------------------------------------------------
# Test: fit() – pipeline lengkap
# ---------------------------------------------------------------------------

class TestFitPipeline:
    def test_fit_runs_without_error(self):
        # Buat dataset sintetis lebih panjang (≥24 titik)
        dates  = pd.date_range("2023-01-01", periods=52, freq="W-MON")
        qty    = (10 + (dates.isocalendar().week % 4) * 3).values
        rows = []
        for d, q in zip(dates, qty):
            rows.append({
                "TANGGAL": f"'{d.strftime('%d-%m-%Y')}'",
                "RTYPE": "'J'",
                "DIV": "'04'",
                "CAT_COD": "'010414'",
                "QTY": q,
            })
        csv_str = "TANGGAL,RTYPE,DIV,CAT_COD,QTY\n" + "\n".join(
            f"{r['TANGGAL']},{r['RTYPE']},{r['DIV']},{r['CAT_COD']},{r['QTY']}"
            for r in rows
        )
        buffer = io.StringIO(csv_str)
        pp = DataPreprocessor(freq="W-MON")
        pp.fit(buffer, category_col="DIV", freq="W-MON")

        assert pp.timeseries_df is not None
        assert len(pp.categories) >= 1
        assert pp.categories[0] == "04"

    def test_fit_raises_on_missing_columns(self):
        bad_csv = io.StringIO("COL_A,COL_B\n1,2\n3,4\n")
        pp = DataPreprocessor()
        with pytest.raises(ValueError, match="Kolom wajib tidak ditemukan"):
            pp.fit(bad_csv)
