import math
import numpy as np
import pandas as pd
import pytest
from extrapolation.alternative import ExtrapolationAlt

@pytest.fixture
def extrap_alt():
    return ExtrapolationAlt()

def test_alternative_extrapolation_structure(extrap_alt):
    """
    Test that alternative_extrapolation returns a DataFrame with the correct structure.
    """
    zero_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    FSP = 3
    UFR = 0.03
    LLFR = 0.03  # For simplicity, use the same value as UFR.
    alpha = 0.1
    df = extrap_alt.alternative_extrapolation(zero_rates, FSP, UFR, LLFR, alpha)
    # Check that the return is a DataFrame with the expected columns.
    assert isinstance(df, pd.DataFrame)
    for col in ['Tenors', 'Zero_CC', 'Forward_CC', 'Discount']:
        assert col in df.columns, f"Missing expected column: {col}"
    # The number of rows should equal the length of zero_rates.
    assert len(df) == len(zero_rates)

def test_alternative_extrapolation_preserves_initial_values(extrap_alt):
    """
    For indices less than FSP, the extrapolated zero curve should equal the input values.
    """
    zero_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    FSP = 3
    UFR = 0.03
    LLFR = 0.03
    alpha = 0.1
    df = extrap_alt.alternative_extrapolation(zero_rates, FSP, UFR, LLFR, alpha)
    # For indices 0 to FSP-1, Zero_CC should equal the input zero_rates.
    np.testing.assert_allclose(df['Zero_CC'].iloc[:FSP].values, zero_rates[:FSP], rtol=1e-6)

def test_alternative_extrapolation_discount_initial(extrap_alt):
    """
    Test that the discount factor at index 0 equals exp(-zero_rates[0]).
    """
    zero_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    FSP = 3
    UFR = 0.03
    LLFR = 0.03
    alpha = 0.1
    df = extrap_alt.alternative_extrapolation(zero_rates, FSP, UFR, LLFR, alpha)
    expected_discount0 = math.exp(-zero_rates[0])
    np.testing.assert_allclose(df['Discount'].iloc[0], expected_discount0, rtol=1e-6)

def test_get_llfr_valid(extrap_alt):
    """
    Test that get_llfr returns the expected long‐term forward rate for a simple example.
    Given:
      - zero_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
      - dlt = [1, 1, 1, 0, 0]
      - weights = [0.25, 0.25, 0.25, 0.25, 0]
    The expected calculation (ignoring the first segment if no preceding liquid point)
    yields an LLFR of approximately 0.03.
    """
    zero_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    dlt = np.array([1, 1, 1, 0, 0])
    weights = np.array([0.25, 0.25, 0.25, 0.25, 0])
    llfr = extrap_alt.get_llfr(zero_rates, dlt, weights)
    np.testing.assert_allclose(llfr, 0.03, rtol=1e-6)

def test_zero_boot_withVA(extrap_alt):
    """
    Test that zero_boot_withVA correctly adjusts the forward curve.
    For indices < FSP, it adds log(1 + VA_value/10000) to the forward values,
    then returns the cumulative mean.
    """
    max_tenor = 10
    FSP = 5
    VA_value = 50  # basis points
    # Create a dummy forward curve array (all values 0.02)
    fwd_boot = np.full(max_tenor, 0.02)
    # Compute the increment:
    increment = math.log(1 + VA_value/10000)
    # Expected update for indices < FSP:
    expected_fwd = fwd_boot.copy()
    expected_fwd[:FSP] += increment
    # The zero curve is the cumulative mean of the updated forward curve.
    expected_zero = np.array([np.mean(expected_fwd[:i+1]) for i in range(max_tenor)])
    result = extrap_alt.zero_boot_withVA(fwd_boot.copy(), max_tenor, FSP, VA_value)
    np.testing.assert_allclose(result, expected_zero, rtol=1e-6)

def test_alternative_extrapolation_quantitative(extrap_alt):
    """
    Quantitative test for alternative_extrapolation.
    
    With:
      zero_rates = [0.01, 0.02, 0.03, 0.04, 0.05],
      FSP = 3, UFR = 0.03, LLFR = 0.03, alpha = 0.1,
    the extrapolated zero rates for indices beyond FSP are computed by:
      For i < FSP, Zero_CC[i] = zero_rates[i].
      For i = FSP (i.e., index 3, year 4):
          fwtemp = ln(1+UFR) + (LLFR - ln(1+UFR)) * (1 - exp(-alpha*(4-3)))/(alpha*(4-3))
          Then, Zero_CC[3] = (3*zero_rates[2] + fwtemp) / 4.
      For i = 4 (year 5):
          fwtemp = ln(1+UFR) + (LLFR - ln(1+UFR)) * (1 - exp(-alpha*(5-3)))/(alpha*(5-3))
          Then, Zero_CC[4] = (3*zero_rates[2] + 2*fwtemp) / 5.
    
    We pre-calculate these values and compare.
    """
    zero_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    FSP = 3
    UFR = 0.03
    LLFR = 0.03
    alpha = 0.1
    ln_ufr = math.log(1+UFR)  # ln(1.03)
    # For indices 0 to 2, use input values.
    expected = np.empty_like(zero_rates)
    expected[0:3] = zero_rates[0:3]
    # For index 3 (year = 4)
    fwtemp_3 = ln_ufr + (LLFR - ln_ufr) * (1 - math.exp(-alpha*(4-3)))/(alpha*(4-3))
    expected[3] = (3*zero_rates[2] + fwtemp_3) / 4
    # For index 4 (year = 5)
    fwtemp_4 = ln_ufr + (LLFR - ln_ufr) * (1 - math.exp(-alpha*(5-3)))/(alpha*(5-3))
    expected[4] = (3*zero_rates[2] + 2*fwtemp_4) / 5

    df = extrap_alt.alternative_extrapolation(zero_rates, FSP, UFR, LLFR, alpha)
    # Compare the computed Zero_CC column with our expected values.
    np.testing.assert_allclose(df['Zero_CC'].values, expected, rtol=1e-6)
