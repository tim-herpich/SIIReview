"""
tests/test_bootstrapping.py
Tests for the Bootstrapping class (refactored version).
"""

import math
import numpy as np
import pytest
from bootstrapping import Bootstrapping


@pytest.fixture
def bootstrap_instance():
    # Use dummy data for a Swap instrument
    # Rates are in percentage (as required by the Bootstrapping class)
    rates = [100, 105, 110, 115, 120]
    dlt = np.array([1, 1, 1, 0, 0])
    coupon_freq = 1.0
    compounding_in = 'A'
    cra = 10.0
    max_tenor = 5
    return Bootstrapping(
        instrument='Swap',
        rates=rates,
        dlt=dlt,
        coupon_freq=coupon_freq,
        compounding_in=compounding_in,
        cra=cra,
        max_tenor=max_tenor
    )


def test_newton_raphson_forward_swap_convergence(bootstrap_instance):
    """
    Test that newton_raphson_forward_swap converges and returns a finite float.
    """
    fw_guess = 0.02
    swap_rate = 0.05
    m = 2
    c = 1.0
    result = bootstrap_instance.newton_raphson_forward_swap(
        fw_guess, swap_rate, m, c)
    assert isinstance(result, float)
    assert not math.isnan(result)
    assert -0.5 < result < 1.0


def test_bootstrap_returns_dataframe(bootstrap_instance):
    """
    Test that bootstrap() returns a DataFrame with the expected structure.
    """
    df = bootstrap_instance.bootstrap()
    expected_columns = ['Tenors', 'Zero_AC',
                        'Forward_AC', 'Zero_CC', 'Forward_CC', 'Discount']
    for col in expected_columns:
        assert col in df.columns, f"Missing column {col} in bootstrapped DataFrame."
    assert len(df) == bootstrap_instance.max_tenor


def test_bootstrap_discount_monotonicity(bootstrap_instance):
    """
    Test that discount factors are strictly decreasing.
    """
    df = bootstrap_instance.bootstrap()
    discounts = df['Discount'].values
    for i in range(1, len(discounts)):
        assert discounts[i] < discounts[i -
                                        1], f"Discount at index {i} is not lower than at {i-1}."
