"""
tests/test_extrapolation_alt.py
Tests for the Alternative Extrapolation (ExtrapolationAlt) class.
"""

import math
import numpy as np
import pandas as pd
import pytest
from extrapolation.alternative import ExtrapolationAlt


@pytest.fixture
def extrap_alt_instance():
    """
    Fixture for an ExtrapolationAlt instance with a dummy zero rate curve.
    """
    zero_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    FSP = 3
    UFR = 0.03
    LLFR = 0.03  # Placeholder LLFR
    alpha = 0.1
    return ExtrapolationAlt(zero_rates, FSP, UFR, LLFR, alpha)


def test_alternative_extrapolation_structure(extrap_alt_instance):
    """
    Test that extrapolate() returns a DataFrame with the expected structure.
    """
    df = extrap_alt_instance.extrapolate()
    expected_cols = ['Tenors', 'Zero_CC', 'Forward_CC', 'Discount', 'Zero_AC', 'Forward_AC']
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == len(extrap_alt_instance.zero_rates)


def test_alternative_extrapolation_preserves_initial(extrap_alt_instance):
    """
    Test that for indices below FSP, the extrapolated zero curve equals the input zero rates.
    """
    df = extrap_alt_instance.extrapolate()
    np.testing.assert_allclose(
        df['Zero_CC'].iloc[:extrap_alt_instance.FSP].values,
        extrap_alt_instance.zero_rates[:extrap_alt_instance.FSP],
        rtol=1e-6
    )


def test_extrapolation_discount_initial(extrap_alt_instance):
    """
    Test that the discount factor at index 0 equals exp(-zero_rate[0]).
    """
    df = extrap_alt_instance.extrapolate()
    expected = math.exp(-extrap_alt_instance.zero_rates[0])
    np.testing.assert_allclose(df['Discount'].iloc[0], expected, rtol=1e-6)
