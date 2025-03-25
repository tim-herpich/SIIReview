"""
tests/test_marketdata.py
Tests for the MarketData class.
"""

import os
import pytest
from marketdata import MarketData

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(CURRENT_DIR, "testinputs")
RATES_FILE = os.path.join(INPUTS_DIR, "rates.xlsx")
SPREADS_FILE = os.path.join(INPUTS_DIR, "spreads.xlsx")


@pytest.fixture
def rates_workbook():
    """
    Fixture to open the rates workbook.
    """
    md = MarketData(RATES_FILE)
    md.open_workbook()
    yield md
    md.close_workbook()


def test_open_rates_workbook(rates_workbook):
    """
    Test that the rates workbook can be opened.
    """
    assert rates_workbook.workbook is not None, "Workbook should not be None after opening."


def test_zero_rates_alt_columns(rates_workbook):
    """
    Test that the 'zero_rates_alt' sheet has the expected columns.
    """
    df = rates_workbook.parse_sheet_to_df("zero_rates_alt")
    expected_columns = ["DLT", "Tenor", "LLFR Weights", "Input Rates"]
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' not found in 'zero_rates_alt'."


def test_zero_rates_alt_first_row():
    """
    Test that the first row of 'zero_rates_alt' has the expected values.
    """
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_alt")
    md.close_workbook()
    first_row = df.iloc[0]
    assert first_row["DLT"] == 1, "First row DLT value should be 1."
    assert first_row["Tenor"] == 1, "First row Tenor value should be 1."
    assert first_row["LLFR Weights"] == 0, "First row LLFR Weights should be 0."
    assert abs(first_row["Input Rates"] - (0.03514)) < 1e-6, "First row Input Rates does not match expected value."


def test_zero_rates_alt_row_ten():
    """
    Test the values for the row where Tenor equals 10 in 'zero_rates_alt'.
    """
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_alt")
    md.close_workbook()
    row = df[df["Tenor"] == 10].iloc[0]
    assert row["DLT"] == 1, "For Tenor=10, DLT should be 1."
    assert row["LLFR Weights"] == 0, "For Tenor=10, LLFR Weights should be 0."
    assert abs(row["Input Rates"] - (0.0247)) < 1e-6, "For Tenor=10, Input Rates does not match expected value."


def test_zero_rates_alt_last_row():
    """
    Test the values for the last row (Tenor=150) in 'zero_rates_alt'.
    """
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_alt")
    md.close_workbook()
    row = df[df["Tenor"] == 150].iloc[0]
    assert row["DLT"] == 0, "For Tenor=150, DLT should be 0."
    assert row["LLFR Weights"] == 0, "For Tenor=150, LLFR Weights should be 0."
    assert abs(row["Input Rates"] - 0.03121) < 1e-6, "For Tenor=150, Input Rates does not match expected value."


def test_zero_rates_sw_columns():
    """
    Test that the 'zero_rates_sw' sheet has the expected columns.
    """
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_sw")
    md.close_workbook()
    expected_columns = ["DLT", "Tenor", "Input Rates"]
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' not found in 'zero_rates_sw'."


def test_zero_rates_sw_first_row():
    """
    Test that the first row of 'zero_rates_sw' has the expected values.
    """
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_sw")
    md.close_workbook()
    first_row = df.iloc[0]
    assert first_row["DLT"] == 1, "First row DLT in 'zero_rates_sw' should be 1."
    assert first_row["Tenor"] == 1, "First row Tenor in 'zero_rates_sw' should be 1."
    assert abs(first_row["Input Rates"] - 0.03514) < 1e-6, "First row Input Rates in 'zero_rates_sw' does not match expected value."


def test_zero_rates_sw_row_ten():
    """
    Test the values for the row where Tenor equals 10 in 'zero_rates_sw'.
    """
    md = MarketData(RATES_FILE)
    md.open_workbook()
    df = md.parse_sheet_to_df("zero_rates_sw")
    md.close_workbook()
    row = df[df["Tenor"] == 10].iloc[0]
    assert row["DLT"] == 1, "For Tenor=10 in 'zero_rates_sw', DLT should be 1."
    assert row["Tenor"] == 10, "Row should have Tenor value 10."
    assert abs(row["Input Rates"] - 0.0247) < 1e-6, "For Tenor=10 in 'zero_rates_sw', Input Rates does not match expected value."
