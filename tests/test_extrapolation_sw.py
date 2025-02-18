import numpy as np
import pandas as pd
import pytest
from extrapolation.smithwilson_excel import ExtrapolationSWExcel

@pytest.fixture
def extrap_sw():
    return ExtrapolationSWExcel()

def test_hh(extrap_sw):
    """
    Test that the helper function hh returns (z + exp(-z))/2.
    """
    z = 0.5
    result = extrap_sw.hh(z)
    expected = (z + np.exp(-z)) / 2
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_Hmat(extrap_sw):
    """
    Test that Hmat(u, v) equals hh(u+v) - hh(|u-v|).
    """
    u, v = 0.5, 0.7
    result = extrap_sw.Hmat(u, v)
    expected = extrap_sw.hh(u+v) - extrap_sw.hh(abs(u-v))
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_smith_wilson_extrapolation_structure(extrap_sw):
    """
    Using a dummy curve_data DataFrame, test that smith_wilson_extrapolation returns a DataFrame
    with the expected columns and number of rows.
    """
    # Create a simple dummy curve_data DataFrame with required columns.
    data = {
        'DLT': [1, 1, 1],
        'Tenor': [1, 2, 3],
        'Input Rates': [0.02, 0.025, 0.03]
    }
    df = pd.DataFrame(data)
    result = extrap_sw.smith_wilson_extrapolation(
        instrument='Zero',
        curve_data=df,
        coupon_freq=1,
        CRA=10.0,
        UFR=0.033,
        alpha_min=0.05,
        CR=1.0,
        CP=60
    )
    # Expected columns.
    expected_columns = ['Tenors', 'Zero_CC', 'Forward_CC', 'Discount', 'Zero_AC', 'Forward_AC']
    for col in expected_columns:
        assert col in result.columns, f"Missing column {col} in smith_wilson_extrapolation result."
    # The implementation sets max_v = 121, so we expect 122 rows.
    assert len(result) == 122

def test_smith_wilson_extrapolation_flat_curve(extrap_sw):
    """
    Test that if the input curve_data represents a flat curve,
    the extrapolated zero curve remains nearly flat in the liquid part.
    """
    # Create a dummy flat input curve_data with a flat rate of 3% for liquid points.
    n = 5
    data = {
        "DLT": [1]*n,
        "Tenor": list(range(1, n+1)),
        "Input Rates": [0.03]*n
    }
    curve_data = pd.DataFrame(data)
    coupon_freq = 1
    CRA = 0.0  # For simplicity, no CRA adjustment.
    UFR = 0.033
    alpha_min = 0.05
    CR = 1.0
    CP = 60
    result = extrap_sw.smith_wilson_extrapolation(
        instrument="Zero",
        curve_data=curve_data,
        coupon_freq=coupon_freq,
        CRA=CRA,
        UFR=UFR,
        alpha_min=alpha_min,
        CR=CR,
        CP=CP
    )
    # Check that for the first n liquid points, the extrapolated zero rates are nearly constant.
    liquid_zero = result['Zero_CC'].iloc[:n].values
    assert np.max(liquid_zero) - np.min(liquid_zero) < 1e-4, \
        "Extrapolated zero rates in the liquid part should be nearly constant for a flat input curve."

def test_smith_wilson_extrapolation_discount_monotonicity(extrap_sw):
    """
    Test that the discount factors from smith_wilson_extrapolation are positive and strictly decreasing.
    """
    n = 5
    data = {
        "DLT": [1]*n,
        "Tenor": list(range(1, n+1)),
        "Input Rates": [0.03]*n
    }
    curve_data = pd.DataFrame(data)
    coupon_freq = 1
    CRA = 0.0
    UFR = 0.033
    alpha_min = 0.05
    CR = 1.0
    CP = 60
    result = extrap_sw.smith_wilson_extrapolation(
        instrument="Zero",
        curve_data=curve_data,
        coupon_freq=coupon_freq,
        CRA=CRA,
        UFR=UFR,
        alpha_min=alpha_min,
        CR=CR,
        CP=CP
    )
    discount = result['Discount'].values
    # All discount factors should be positive.
    assert np.all(discount > 0), "All discount factors should be positive."
    # They should be strictly decreasing.
    assert np.all(np.diff(discount[:-1]) < 0), "Discount factors should be strictly decreasing."


def test_smith_wilson_forward_rate_consistency(extrap_sw):
    """
    Test that the forward rates computed in the SW extrapolation are consistent with the discount factors.
    Specifically, for each index i from 1 to len(Discount)-2:
    
        Forward_AC[i] ≈ Discount[i-1]/Discount[i] - 1,
        Forward_CC[i] ≈ log(Discount[i-1]/Discount[i]).
    
    This test checks both the annually compounded (AC) and continuously compounded (CC) forward rates.
    """
    # Create a dummy input curve_data DataFrame with 5 liquid points.
    n = 5
    data = {
        "DLT": [1] * n,
        "Tenor": list(range(1, n + 1)),
        "Input Rates": [0.03] * n
    }
    curve_data = pd.DataFrame(data)
    
    coupon_freq = 1
    CRA = 0.0
    UFR = 0.033
    alpha_min = 0.05
    CR = 1.0
    CP = 60

    result = extrap_sw.smith_wilson_extrapolation(
        instrument="Zero",
        curve_data=curve_data,
        coupon_freq=coupon_freq,
        CRA=CRA,
        UFR=UFR,
        alpha_min=alpha_min,
        CR=CR,
        CP=CP
    )

    # Extract the discount curve and forward rate arrays.
    discount = result['Discount'].values
    forward_ac = result['Forward_AC'].values
    forward_cc = result['Forward_CC'].values

    # We test indices 1 to len(discount)-2 because the last row may be set to a special value.
    for i in range(1, len(discount) - 1):
        if discount[i] != 0:
            expected_ac = discount[i - 1] / discount[i] - 1
            expected_cc = np.log(discount[i - 1] / discount[i])
            np.testing.assert_allclose(
                forward_ac[i],
                expected_ac,
                rtol=1e-6,
                err_msg=f"Forward_AC rate at index {i} inconsistent: expected {expected_ac}, got {forward_ac[i]}"
            )
            np.testing.assert_allclose(
                forward_cc[i],
                expected_cc,
                rtol=1e-6,
                err_msg=f"Forward_CC rate at index {i} inconsistent: expected {expected_cc}, got {forward_cc[i]}"
            )

def test_getInputwithVA(extrap_sw):
    """
    Test that getInputwithVA produces a DataFrame with the same columns as the provided curve_data,
    and that the 'Input Rates' column is adjusted by adding VA_value/10000.
    """
    # Create a dummy zero_rates_extrapolated_ac array (length >= LLP+1)
    zero_rates_extrapolated_ac = np.linspace(0.01, 0.05, 10)
    LLP = 5
    VA_value = 50  # basis points
    # Create a dummy curve_data DataFrame with the expected columns.
    dummy_curve_data = pd.DataFrame(columns=["DLT", "Tenor", "Input Rates"])
    result = extrap_sw.getInputwithVA(zero_rates_extrapolated_ac, LLP, VA_value, dummy_curve_data)
    # Check that the result has LLP rows.
    assert len(result) == LLP
    # The 'DLT' column should be all ones.
    assert (result["DLT"] == 1).all()
    # The 'Tenor' column should be [1, 2, ..., LLP].
    np.testing.assert_array_equal(result["Tenor"].values, np.arange(1, LLP+1))
    # The 'Input Rates' should equal zero_rates_extrapolated_ac[1:LLP+1] + VA_value/10000.
    expected_input = zero_rates_extrapolated_ac[1:LLP+1] + VA_value/10000
    np.testing.assert_allclose(result["Input Rates"].astype(float).values, expected_input, rtol=1e-6)
