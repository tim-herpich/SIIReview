#!/usr/bin/env python3
"""
tests/test_extrapolation_e2e.py

End-to-end tests for extrapolation methods:
  - Alternative Extrapolation (ExtrapolationAlt)
  - Smith–Wilson Extrapolation (ExtrapolationSW)

This module loads market data from Excel inputs and validates the extrapolation results.
"""

import numpy as np
import pandas as pd
import pytest
from marketdata import MarketData
from tests.Parameters.testparameters import TestParameters
from bootstrapping import Bootstrapping
from extrapolation.alternative import ExtrapolationAlt
from extrapolation.smithwilson import ExtrapolationSW


@pytest.fixture
def market_data_alt():
    """
    Fixture to load alternative market data from Excel for extrapolation.
    """
    md_rates = MarketData(filepath="tests/testinputs/rates.xlsx")
    md_rates.open_workbook()
    df_alt = md_rates.parse_sheet_to_df("zero_rates_alt")
    md_rates.close_workbook()
    return df_alt


@pytest.fixture
def market_data_sw():
    """
    Fixture to load Smith–Wilson market data from Excel.
    """
    md_rates = MarketData(filepath="tests/testinputs/rates.xlsx")
    md_rates.open_workbook()
    df_sw = md_rates.parse_sheet_to_df("zero_rates_sw")
    md_rates.close_workbook()
    return df_sw


@pytest.fixture
def parameters():
    """
    Fixture to load test parameters.
    """
    return TestParameters()


@pytest.fixture
def bootstrap_data(market_data_alt, parameters):
    """
    Fixture to perform bootstrapping on the alternative market data.
    """
    max_tenor = parameters.max_tenorofAlt
    dlt_array = np.zeros(max_tenor)
    rate_array = np.zeros(max_tenor)
    weight_array = np.zeros(max_tenor)
    for dlt_val, tenor, wt, rate in zip(
            market_data_alt["DLT"], market_data_alt["Tenor"],
            market_data_alt["LLFR Weights"], market_data_alt["Input Rates"]):
        t = int(round(tenor))
        if 1 <= t <= max_tenor:
            idx = t - 1
            dlt_array[idx] = dlt_val
            rate_array[idx] = rate
            weight_array[idx] = wt
    bootstrapper = Bootstrapping(
        instrument=parameters.instrument,
        rates=rate_array * 100.0,
        dlt=dlt_array,
        coupon_freq=parameters.coupon_freq,
        compounding_in=parameters.compounding_in,
        cra=parameters.CRA,
        max_tenor=max_tenor
    )
    boot_df = bootstrapper.bootstrap()
    return boot_df, dlt_array, weight_array


@pytest.fixture
def extrapolation_alt(bootstrap_data, parameters):
    """
    Fixture to perform alternative extrapolation using bootstrapped data.
    """
    boot_df, dlt_array, weight_array = bootstrap_data
    extrap_alt = ExtrapolationAlt(
        zero_rates=boot_df["Zero_CC"].values,
        FSP=parameters.FSP,
        UFR=parameters.UFR,
        LLFR=0,  # placeholder; will be computed next
        alpha=parameters.alpha
    )
    llfr = extrap_alt.get_llfr(dlt_array, weight_array)
    extrap_alt.LLFR = llfr
    results_alt = extrap_alt.extrapolate()
    return results_alt


def test_extrapolation_alt_zero_rate_tenor_1(extrapolation_alt):
    """
    Test the continuous zero rate for tenor 1 using alternative extrapolation.
    """
    expected = 0.03357  # Placeholder expected value from real market data.
    actual = extrapolation_alt.loc[extrapolation_alt['Tenors']
                                   == 1, 'Zero_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_extrapolation_alt_zero_rate_tenor_10(extrapolation_alt):
    """
    Test the continuous zero rate for tenor 10 using alternative extrapolation.
    """
    expected = 0.02342  # Placeholder expected value.
    actual = extrapolation_alt.loc[extrapolation_alt['Tenors']
                                   == 10, 'Zero_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_extrapolation_alt_zero_rate_tenor_30(extrapolation_alt):
    """
    Test the continuous zero rate for tenor 30 using alternative extrapolation.
    """
    expected = 0.02432  # Placeholder expected value.
    actual = extrapolation_alt.loc[extrapolation_alt['Tenors']
                                   == 30, 'Zero_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_extrapolation_alt_zero_rate_tenor_60(extrapolation_alt):
    """
    Test the continuous zero rate for tenor 60 using alternative extrapolation.
    """
    expected = 0.02797  # Placeholder expected value.
    actual = extrapolation_alt.loc[extrapolation_alt['Tenors']
                                   == 60, 'Zero_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


@pytest.fixture
def sw_extrapolation():
    """
    Fixture to set up the Smith–Wilson extrapolation instance.

    Loads SW rates from "inputs/rates.xlsx" (sheet "zero_rates_sw")
    and alternative rates from "inputs/rates.xlsx" (sheet "zero_rates_alt") for bootstrapping.
    The DLT values in the bootstrapped DataFrame are then overridden using the SW sheet.
    """
    # Load SW rates data.
    md_rates = MarketData(filepath="inputs/rates.xlsx")
    md_rates.open_workbook()
    df_sw = md_rates.parse_sheet_to_df("zero_rates_sw")
    md_rates.close_workbook()

    # Load alternative rates data for bootstrapping.
    md_alt = MarketData(filepath="inputs/rates.xlsx")
    md_alt.open_workbook()
    df_alt = md_alt.parse_sheet_to_df("zero_rates_alt")
    md_alt.close_workbook()

    # Load parameters.
    params = TestParameters()
    max_tenor = params.max_tenorofAlt

    # Prepare bootstrapping arrays.
    dlt_array = np.zeros(max_tenor)
    rate_array = np.zeros(max_tenor)
    weight_array = np.zeros(max_tenor)
    for dlt_val, tenor, wt, rate in zip(
            df_alt["DLT"], df_alt["Tenor"],
            df_alt["LLFR Weights"], df_alt["Input Rates"]):
        t = int(round(tenor))
        if 1 <= t <= max_tenor:
            idx = t - 1
            dlt_array[idx] = dlt_val
            rate_array[idx] = rate
            weight_array[idx] = wt

    # Bootstrap the curve using the alternative rates.
    # Bootstrapping expects rates in percentage, so multiply by 100.
    bootstrapper = Bootstrapping(
        instrument=params.instrument,
        rates=rate_array * 100.0,
        dlt=dlt_array,
        coupon_freq=params.coupon_freq,
        compounding_in=params.compounding_in,
        cra=params.CRA,
        max_tenor=max_tenor
    )
    boot_df = bootstrapper.bootstrap()

    # For SW extrapolation, override the DLT column using the SW sheet.
    boot_df['DLT'] = df_sw['DLT'].values

    # Instantiate and run the Smith–Wilson extrapolator.
    extrap_sw = ExtrapolationSW(
        curve_data=boot_df,
        UFR=params.UFR,
        alpha_min=params.alpha_min_SW,
        CR=params.CR_SW,
        CP=params.CP_SW
    )
    results_sw = extrap_sw.extrapolate()
    return results_sw


def test_zero_rate_tenor_10(sw_extrapolation):
    """
    Test the continuous zero rate for tenor 10 using Smith–Wilson extrapolation.
    """
    expected = 0.02342  # placeholder expected value
    actual = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                  == 10, 'Zero_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_zero_rate_tenor_60(sw_extrapolation):
    """
    Test the continuous zero rate for tenor 60 using Smith–Wilson extrapolation.
    """
    expected = 0.02770  # placeholder expected value
    actual = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                  == 60, 'Zero_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_forward_rate_tenor_10(sw_extrapolation):
    """
    Test the forward rate (continuous compounding) for tenor 10 using Smith–Wilson extrapolation.
    """
    expected = 0.02395  # placeholder expected value
    actual = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                  == 10, 'Forward_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_forward_rate_tenor_60(sw_extrapolation):
    """
    Test the forward rate (continuous compounding) for tenor 60 using Smith–Wilson extrapolation.
    """
    expected = 0.03236  # placeholder expected value
    actual = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                  == 60, 'Forward_CC'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_discount_factor_tenor_10(sw_extrapolation):
    """
    Test the discount factor for tenor 10 using Smith–Wilson extrapolation.
    """
    expected = 0.79117  # placeholder expected value
    actual = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                  == 10, 'Discount'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_discount_factor_tenor_60(sw_extrapolation):
    """
    Test the discount factor for tenor 60 using Smith–Wilson extrapolation.
    """
    expected = 0.18967  # placeholder expected value
    actual = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                  == 60, 'Discount'].values[0]
    np.testing.assert_almost_equal(actual, expected, decimal=4)


def test_forward_rate_calculation_consistency(sw_extrapolation):
    """
    Ensure that the forward rate computed from discount factors is consistent.
    For a given tenor (e.g., tenor 5), forward rate = log(discount[t-1] / discount[t]).
    """
    tenor = 5
    discount_prev = sw_extrapolation.loc[sw_extrapolation['Tenors'] == (
        tenor - 1), 'Discount'].values[0]
    discount_curr = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                         == tenor, 'Discount'].values[0]
    computed_forward = np.log(discount_prev / discount_curr)
    reported_forward = sw_extrapolation.loc[sw_extrapolation['Tenors']
                                            == tenor, 'Forward_CC'].values[0]
    np.testing.assert_almost_equal(
        computed_forward, reported_forward, decimal=4)
