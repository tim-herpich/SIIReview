# tests/test_marketdata.py

import os
import pytest
import pandas as pd
from marketdata import MarketData

# Get the directory where this test file is located.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Since the inputs folder is inside tests/, set its path accordingly.
INPUTS_DIR = os.path.join(CURRENT_DIR, "inputs")

# Define the absolute paths to the input Excel files.
RATES_FILE = os.path.join(INPUTS_DIR, "tests_rates.xlsx")
SPREADS_FILE = os.path.join(INPUTS_DIR, "tests_spreads.xlsx")

# --------------------------
# Tests for tests_rates.xlsx
# --------------------------

def test_open_rates_workbook():
    """Test that the rates workbook can be opened."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    assert md.workbook is not None, "Workbook should not be None after opening."
    md.close_workbook()

def test_zero_rates_alt_columns():
    """Test that the 'zero_rates_alt' sheet has the expected columns."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_alt")
    md.close_workbook()
    expected_columns = ["DLT", "Tenor", "LLFR Weights", "Input Rates"]
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' not found in 'zero_rates_alt'."

def test_zero_rates_alt_first_row():
    """Test that the first row of 'zero_rates_alt' has the expected values."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_alt")
    md.close_workbook()
    # Expected first row: DLT=1, Tenor=1, LLFR Weights=0, Input Rates=-0.00445
    first_row = df.iloc[0]
    assert first_row["DLT"] == 1, "First row DLT value should be 1."
    assert first_row["Tenor"] == 1, "First row Tenor value should be 1."
    assert first_row["LLFR Weights"] == 0, "First row LLFR Weights should be 0."
    assert abs(first_row["Input Rates"] - (-0.00445)) < 1e-6, "First row Input Rates does not match expected value."

def test_zero_rates_alt_row_ten():
    """Test the values for the row where Tenor equals 10 in 'zero_rates_alt'."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_alt")
    md.close_workbook()
    # From the provided data, for Tenor 10:
    # Expected: DLT=1, LLFR Weights=0, Input Rates=-0.002665102
    row = df[df["Tenor"] == 10].iloc[0]
    assert row["DLT"] == 1, "For Tenor=10, DLT should be 1."
    assert row["LLFR Weights"] == 0, "For Tenor=10, LLFR Weights should be 0."
    assert abs(row["Input Rates"] - (-0.002665102)) < 1e-6, "For Tenor=10, Input Rates does not match expected value."

def test_zero_rates_alt_last_row():
    """Test the values for the last row (Tenor=150) in 'zero_rates_alt'."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_alt")
    md.close_workbook()
    # According to the provided data, for Tenor 150:
    # Expected: DLT=0, LLFR Weights=0, Input Rates=0.029825892
    row = df[df["Tenor"] == 150].iloc[0]
    assert row["DLT"] == 0, "For Tenor=150, DLT should be 0."
    assert row["LLFR Weights"] == 0, "For Tenor=150, LLFR Weights should be 0."
    assert abs(row["Input Rates"] - 0.029825892) < 1e-6, "For Tenor=150, Input Rates does not match expected value."

def test_zero_rates_sw_columns():
    """Test that the 'zero_rates_sw' sheet has the expected columns."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_sw")
    md.close_workbook()
    expected_columns = ["DLT", "Tenor", "Input Rates"]
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' not found in 'zero_rates_sw'."

def test_zero_rates_sw_first_row():
    """Test that the first row of 'zero_rates_sw' has the expected values."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_sw")
    md.close_workbook()
    # Expected first row: DLT=1, Tenor=1, Input Rates=0.009249672
    first_row = df.iloc[0]
    assert first_row["DLT"] == 1, "First row DLT in 'zero_rates_sw' should be 1."
    assert first_row["Tenor"] == 1, "First row Tenor in 'zero_rates_sw' should be 1."
    assert abs(first_row["Input Rates"] - 0.009249672) < 1e-6, "First row Input Rates in 'zero_rates_sw' does not match expected value."

def test_zero_rates_sw_row_ten():
    """Test the values for the row where Tenor equals 10 in 'zero_rates_sw'."""
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_sw")
    md.close_workbook()
    # Expected for Tenor=10: DLT=1, Input Rates=0.021872289
    row = df[df["Tenor"] == 10].iloc[0]
    assert row["DLT"] == 1, "For Tenor=10 in 'zero_rates_sw', DLT should be 1."
    assert row["Tenor"] == 10, "Row should have Tenor value 10."
    assert abs(row["Input Rates"] - 0.021872289) < 1e-6, "For Tenor=10 in 'zero_rates_sw', Input Rates does not match expected value."

# ------------------------------
# Tests for tests_spreads.xlsx
# ------------------------------

def test_open_spreads_workbook():
    """Test that the spreads workbook can be opened."""
    md = MarketData(SPREADS_FILE)
    md.open_workbook()
    assert md.workbook is not None, "Workbook for spreads should not be None after opening."
    md.close_workbook()

def test_spreads_va_columns():
    """Test that the 'spreads_va' sheet has the expected columns."""
    md = MarketData(SPREADS_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("spreads_va")
    md.close_workbook()
    # Expected: an 'Issuer' column and columns 1 to 20.
    assert "Issuer" in df.columns, "'Issuer' column is missing in 'spreads_va'."
    for col in range(1, 21):
        assert col in df.columns, f"Column '{col}' is missing in 'spreads_va'."

def test_spreads_va_row_count():
    """Test that the 'spreads_va' sheet contains the expected number of rows."""
    md = MarketData(SPREADS_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("spreads_va")
    md.close_workbook()
    # Based on the provided sample data, there should be 52 rows.
    assert len(df) == 52, f"Expected 52 rows in 'spreads_va', got {len(df)}."

def test_spreads_va_first_row():
    """Test that the first row of 'spreads_va' has the expected values (for issuer AT)."""
    md = MarketData(SPREADS_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("spreads_va")
    md.close_workbook()
    first_row = df.iloc[0]
    assert first_row["Issuer"] == "AT", "The first row Issuer should be 'AT'."
    value = float(first_row[1]) if not isinstance(first_row[1], float) else first_row[1]
    assert abs(value - 0.0022) < 1e-6, "The value in column 1 for issuer AT does not match expected value."

def test_spreads_va_fr_values():
    """Test that the row for issuer 'FR' in 'spreads_va' has the expected values."""
    md = MarketData(SPREADS_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("spreads_va")
    md.close_workbook()
    row = df[df["Issuer"] == "FR"].iloc[0]
    # For issuer FR, expect: column 1 equals 0.0120625 and column 2 equals 0.0124375.
    assert abs(float(row[1]) - 0.0120625) < 1e-6, "Issuer FR, column '1' does not match expected value."
    assert abs(float(row[2]) - 0.0124375) < 1e-6, "Issuer FR, column '2' does not match expected value."

def test_spreads_va_jp_values():
    """Test that the row for issuer 'JP' in 'spreads_va' has the expected values."""
    md = MarketData(SPREADS_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("spreads_va")
    md.close_workbook()
    row = df[df["Issuer"] == "JP"].iloc[0]
    # For issuer JP, expect: column 1 equals -0.0134.
    assert abs(float(row[1]) - (-0.0134)) < 1e-6, "Issuer JP, column '1' does not match expected value."

def test_spreads_va_us_values():
    """Test that the row for issuer 'US' in 'spreads_va' has the expected values."""
    md = MarketData(SPREADS_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("spreads_va")
    md.close_workbook()
    row = df[df["Issuer"] == "US"].iloc[0]
    # For issuer US, expect: column 1 equals 0.021.
    assert abs(float(row[1]) - 0.021) < 1e-6, "Issuer US, column '1' does not match expected value."
