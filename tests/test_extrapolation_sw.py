"""
tests/test_extrapolation_sw.py
Tests for the Smith–Wilson extrapolation (ExtrapolationSW) class.
"""

import math
import numpy as np
import pandas as pd
import pytest
from extrapolation.smithwilson import ExtrapolationSW


@pytest.fixture
def sw_instance():
    # Create a dummy curve_data DataFrame with required columns.
    data = {
        'DLT': [1, 1, 1],
        'Tenors': [1, 2, 3],
        'Zero_CC': [0.02, 0.025, 0.03],
        'Discount': [math.exp(-0.02*1), math.exp(-0.025*2), math.exp(-0.03*3)],
        'Forward_CC': [0.02, 0.025, 0.03]
    }
    df = pd.DataFrame(data)
    # Parameters for SW
    UFR = 0.033
    alpha_min = 0.05
    CR = 1.0
    CP = 60
    return ExtrapolationSW(curve_data=df, UFR=UFR, alpha_min=alpha_min, CR=CR, CP=CP)


def test_sw_kernel_functions(sw_instance):
    """
    Test the helper functions _w and _w_vector.
    """
    t1, t2 = 0.5, 0.7
    omega = math.log(1 + sw_instance.UFR)
    alpha = 0.1
    w_val = sw_instance._w(t1, t2, alpha, omega)
    expected = np.exp(-omega*(t1+t2)) * (alpha*min(t1, t2) - np.exp(-alpha *
                                                                    max(t1, t2))*0.5*(np.exp(alpha*min(t1, t2))-np.exp(-alpha*min(t1, t2))))
    np.testing.assert_allclose(w_val, expected, rtol=1e-6)
    # Test vector function:
    t_obs = np.array([0.5, 1.0, 1.5])
    w_vec = sw_instance._w_vector(1.0, t_obs, alpha, omega)
    expected_vec = np.array(
        [sw_instance._w(1.0, ti, alpha, omega) for ti in t_obs])
    np.testing.assert_allclose(w_vec, expected_vec, rtol=1e-6)


def test_sw_extrapolation_structure(sw_instance):
    """
    Test that extrapolate() returns a DataFrame with the expected structure.
    """
    df = sw_instance.extrapolate()
    expected_columns = ['Tenors', 'Zero_CC',
                        'Forward_CC', 'Discount', 'Zero_AC', 'Forward_AC']
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
    # For our dummy input, expect number of rows equal to length of curve_data
    assert len(df) == len(sw_instance.curve_data)


def test_sw_add_va(sw_instance):
    """
    Test that add_va() returns a DataFrame with adjusted VA values.
    """
    LLP = 2
    VA_value = 50  # basis points
    df_va = sw_instance.add_va(LLP, VA_value)
    # Check that the first LLP rows have been adjusted.
    # For example, compare the original Zero_AC and the VA-adjusted Zero_AC.
    original_df = sw_instance.extrapolate()
    adjusted = df_va.loc[:LLP-1, 'Zero_AC'].values
    expected = original_df.loc[:LLP-1, 'Zero_AC'].values + VA_value/10000.0
    np.testing.assert_allclose(adjusted, expected, rtol=1e-6)
