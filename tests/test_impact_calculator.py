import numpy as np
import pandas as pd
import pytest
from impact import ImpactCalculator

# -----------------------------
# Helper function to create a discount curve DataFrame.
# -----------------------------
def make_discount_curve(tenors, rate):
    """
    Create a discount curve DataFrame with a constant zero rate.
    Columns: "Tenors" and "Zero_CC".
    """
    return pd.DataFrame({
        "Tenors": tenors,
        "Zero_CC": np.full(len(tenors), rate)
    })

@pytest.fixture
def impact_calc():
    return ImpactCalculator()

# -----------------------------
# Tests for _interpolate_rate
# -----------------------------
def test_interpolate_rate_exact(impact_calc):
    tenors = np.array([0, 1, 2, 3, 4, 5])
    rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    rate = impact_calc._interpolate_rate(df, 3)
    np.testing.assert_allclose(rate, 0.04, rtol=1e-6)

def test_interpolate_rate_in_between(impact_calc):
    tenors = np.array([0, 1, 2, 3, 4, 5])
    rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    rate = impact_calc._interpolate_rate(df, 1.5)
    np.testing.assert_allclose(rate, 0.025, rtol=1e-6)

def test_interpolate_rate_extrapolation_lower(impact_calc):
    tenors = np.array([1, 2, 3, 4, 5])
    rates = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    rate = impact_calc._interpolate_rate(df, 0)
    np.testing.assert_allclose(rate, 0.01, rtol=1e-6)

def test_interpolate_rate_extrapolation_upper(impact_calc):
    tenors = np.array([0, 1, 2, 3])
    rates = np.array([0.01, 0.02, 0.03, 0.04])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    rate = impact_calc._interpolate_rate(df, 5)
    np.testing.assert_allclose(rate, 0.06, rtol=1e-6)

# -----------------------------
# Tests for compute_zcb_pv
# -----------------------------
def test_compute_zcb_pv_exact_maturity(impact_calc):
    """
    Test PV computation for a maturity exactly in the discount curve.
    """
    discount_curve = make_discount_curve(tenors=[1, 5, 10, 20, 30], rate=0.03)
    maturity = 10
    llp = 0  # No adjustment for LLP
    expected_pv = np.exp(-0.03 * (maturity - llp))  # PV using the exact rate from the curve

    computed_pv = impact_calc.compute_zcb_pv(discount_curve, maturity, llp)
    np.testing.assert_allclose(computed_pv, expected_pv, rtol=1e-6)

def test_compute_zcb_pv_interpolated_maturity(impact_calc):
    """
    Test PV computation for a maturity between available tenors.
    """
    discount_curve = pd.DataFrame({
        "Tenors": [1, 5, 10, 20, 30],
        "Zero_CC": [0.02, 0.025, 0.03, 0.035, 0.04]
    })
    maturity = 7  # Between 5 and 10 years
    llp = 0
    interpolated_rate = 0.025 + ((0.03 - 0.025) / (10 - 5)) * (maturity - 5)  # Linear interpolation
    expected_pv = np.exp(-interpolated_rate * (maturity - llp))

    computed_pv = impact_calc.compute_zcb_pv(discount_curve, maturity, llp)
    np.testing.assert_allclose(computed_pv, expected_pv, rtol=1e-6)

def test_compute_zcb_pv_extrapolated_maturity(impact_calc):
    """
    Test PV computation for a maturity beyond the available tenors (extrapolation).
    """
    discount_curve = make_discount_curve(tenors=[1, 5, 10, 20, 30], rate=0.04)
    maturity = 40  # Beyond max tenor
    llp = 0
    expected_pv = np.exp(-0.04 * (maturity - llp))  # Extrapolated rate assumed constant

    computed_pv = impact_calc.compute_zcb_pv(discount_curve, maturity, llp)
    np.testing.assert_allclose(computed_pv, expected_pv, rtol=1e-6)

def test_compute_zcb_pv_maturity_equal_to_llp(impact_calc):
    """
    Test PV computation when maturity is exactly equal to LLP.
    """
    discount_curve = make_discount_curve(tenors=[1, 5, 10, 20, 30], rate=0.03)
    maturity = 10
    llp = 10  # Maturity and LLP are the same
    expected_pv = 1.0  # Discounting period is zero, so PV should be 1

    computed_pv = impact_calc.compute_zcb_pv(discount_curve, maturity, llp)
    np.testing.assert_allclose(computed_pv, expected_pv, rtol=1e-6)

def test_compute_zcb_pv_zero_maturity(impact_calc):
    """
    Test behavior when maturity is zero (should return PV=1).
    """
    discount_curve = make_discount_curve(tenors=[1, 5, 10, 20, 30], rate=0.03)
    maturity = 0
    llp = 0
    expected_pv = 1.0

    computed_pv = impact_calc.compute_zcb_pv(discount_curve, maturity, llp)
    np.testing.assert_allclose(computed_pv, expected_pv, rtol=1e-6)
