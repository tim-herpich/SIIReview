import math
import numpy as np
import pandas as pd
import pytest
from bootstrapping import Bootstrapping

@pytest.fixture
def bootstrap_instance():
    return Bootstrapping()

### Tests for newton_raphson_forward_swap

def test_newton_raphson_forward_swap_convergence(bootstrap_instance):
    """
    Test that newton_raphson_forward_swap converges to a value and returns a float.
    We use arbitrary parameters and check that the returned forward rate is finite.
    """
    fw_guess = 0.02
    swap_rate = 0.05
    m = 2
    c = 1.0
    result = bootstrap_instance.newton_raphson_forward_swap(fw_guess, swap_rate, m, c)
    assert isinstance(result, float)
    assert not math.isnan(result)
    # Expect result to be within a reasonable range.
    assert -0.5 < result < 1.0

### Tests for bootstrap_zero_to_zero_full

def test_bootstrap_zero_to_zero_full_structure(bootstrap_instance):
    """
    Test that bootstrap_zero_to_zero_full returns a DataFrame with the expected columns.
    """
    # Dummy input: zero rates in percentage, all valid (DLT==1)
    zero_rates_init = np.array([100, 105, 110, 115, 120], dtype=float)
    dlt = np.ones(5, dtype=int)
    compounding_in = 'A'
    cra = 10.0  # credit risk adjustment in basis points
    max_tenor = 5
    compounding_out = 'A'
    
    df = bootstrap_instance.bootstrap_zero_to_zero_full(
        zero_rates_init, dlt, compounding_in, cra, max_tenor, compounding_out
    )
    for col in ['Tenors', 'Zero_CC', 'Forward_CC', 'Discount']:
        assert col in df.columns
    assert len(df) == max_tenor

def test_bootstrap_zero_to_zero_full_discount_monotonicity(bootstrap_instance):
    """
    Test that the discount factors from bootstrap_zero_to_zero_full are strictly decreasing.
    """
    zero_rates_init = np.array([100, 105, 110, 115, 120], dtype=float)
    dlt = np.ones(5, dtype=int)
    compounding_in = 'A'
    cra = 10.0
    max_tenor = 5
    compounding_out = 'A'
    
    df = bootstrap_instance.bootstrap_zero_to_zero_full(
        zero_rates_init, dlt, compounding_in, cra, max_tenor, compounding_out
    )
    discounts = df['Discount'].values
    for i in range(1, len(discounts)):
        assert discounts[i] < discounts[i - 1], f"Discount at index {i} is not lower than index {i-1}."

### Tests for bootstrap_swap_to_zero_full

def test_bootstrap_swap_to_zero_full_structure(bootstrap_instance):
    """
    Test that bootstrap_swap_to_zero_full returns a DataFrame with the expected columns.
    """
    # Dummy input: swap rates (in percentage) and corresponding DLT flags.
    swap_rates = np.array([100, 105, 110, 115, 120], dtype=float)
    dlt = np.array([1, 1, 1, 0, 0], dtype=int)
    coupon_freq = 1.0
    cra = 10.0
    max_tenor = 5
    compounding_out = 'A'
    
    df = bootstrap_instance.bootstrap_swap_to_zero_full(
        swap_rates, dlt, coupon_freq, cra, max_tenor, compounding_out
    )
    for col in ['Tenors', 'Zero_CC', 'Forward_CC', 'Discount']:
        assert col in df.columns
    assert len(df) == max_tenor

def test_bootstrap_swap_to_zero_full_discount_monotonicity(bootstrap_instance):
    """
    Test that the discount factors from bootstrap_swap_to_zero_full are strictly decreasing.
    """
    swap_rates = np.array([100, 105, 110, 115, 120], dtype=float)
    dlt = np.array([1, 1, 1, 0, 0], dtype=int)
    coupon_freq = 1.0
    cra = 10.0
    max_tenor = 5
    compounding_out = 'A'
    
    df = bootstrap_instance.bootstrap_swap_to_zero_full(
        swap_rates, dlt, coupon_freq, cra, max_tenor, compounding_out
    )
    discounts = df['Discount'].values
    for i in range(1, len(discounts)):
        assert discounts[i] < discounts[i - 1], f"Discount at index {i} is not lower than index {i-1}."

