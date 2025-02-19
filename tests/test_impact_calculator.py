"""
tests/test_impact_calculator.py
Tests for the ImpactCalculator class.
"""

import numpy as np
import pandas as pd
import pytest
from impact import ImpactCalculator

def make_discount_curve(tenors, rate):
    """
    Helper: Create a discount curve DataFrame with a constant zero rate.
    """
    return pd.DataFrame({
        "Tenors": tenors,
        "Zero_CC": np.full(len(tenors), rate)
    })

@pytest.fixture
def impact_calc():
    return ImpactCalculator()

def test_interpolate_rate_exact(impact_calc):
    tenors = np.array([0, 1, 2, 3, 4, 5])
    rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    rate = impact_calc._interpolate_rate(df, 3)
    np.testing.assert_allclose(rate, 0.04, rtol=1e-6)

def test_compute_zcb_pv_exact(impact_calc):
    discount_curve = make_discount_curve([1,5,10,20,30], 0.03)
    maturity = 10
    llp = 0
    expected = np.exp(-0.03 * (maturity - llp))
    computed = impact_calc.compute_zcb_pv(discount_curve, maturity, llp)
    np.testing.assert_allclose(computed, expected, rtol=1e-6)
