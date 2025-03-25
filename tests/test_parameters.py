"""
tests/test_parameters.py
Tests for the Parameters class.
"""

import pytest
from tests.Parameters.testparameters import TestParameters


@pytest.fixture
def params():
    """
    Fixture for a TestParameters instance.
    """
    return TestParameters()


def test_compounding_in(params):
    """Test the compounding_in parameter."""
    assert params.compounding_in == 'A'


def test_compounding_out(params):
    """Test the compounding_out parameter."""
    assert params.compounding_out == 'C'


def test_instrument(params):
    """Test the instrument parameter."""
    assert params.instrument == 'Zero'


def test_VA_value_is_float_class(params):
    """Test that the default VA_value is the float class."""
    assert params.VA_value is float


def test_coupon_freq(params):
    """Test the coupon frequency."""
    assert params.coupon_freq == 1.0


def test_FSP(params):
    """Test the FSP parameter."""
    assert params.FSP == 20


def test_UFR(params):
    """Test the UFR parameter."""
    assert params.UFR == 0.033


def test_CRA(params):
    """Test the CRA parameter."""
    assert params.CRA == 10.0


def test_alpha_min_SW(params):
    """Test the alpha_min_SW parameter."""
    assert params.alpha_min_SW == 0.05


def test_CR_SW(params):
    """Test the CR_SW parameter."""
    assert params.CR_SW == 1.0


def test_CP_SW(params):
    """Test the CP_SW parameter."""
    assert params.CP_SW == 60


def test_LLP_SW(params):
    """Test the LLP_SW parameter."""
    assert params.LLP_SW == 20


def test_max_tenorofAlt(params):
    """Test the max_tenorofAlt parameter."""
    assert params.max_tenorofAlt == 150


def test_asset_size(params):
    """Test the asset_size parameter."""
    assert params.asset_size == 1e6


def test_asset_duration(params):
    """Test the asset_duration parameter."""
    assert params.asset_duration == 6.8


def test_liability_size(params):
    """Test the liability_size parameter."""
    assert params.liability_size == 0.8e6


def test_liability_duration(params):
    """Test the liability_duration parameter."""
    assert params.liability_duration == 10.0


def test_fi_asset_size(params):
    """Test the fi_asset_size parameter."""
    assert params.fi_asset_size == 0.62 * params.asset_size


def test_pvbp_fi_assets(params):
    """Test the pvbp_fi_assets parameter."""
    assert params.pvbp_fi_assets == 0.1 * params.asset_duration


def test_pvbp_liabs(params):
    """Test the pvbp_liabs parameter."""
    assert params.pvbp_liabs == 0.1 * params.liability_duration


def test_scenarios_is_list(params):
    """Test that scenarios is a list."""
    assert isinstance(params.scenarios, list)


def test_scenarios_length(params):
    """Test that scenarios contains at least one element."""
    assert len(params.scenarios) > 0


def test_scenarios_content(params):
    """Test that scenarios match the expected content."""
    expected = [
        {'name': 'low_interest_base_spreads', 'irshift': -200, 'csshift': 0, 'vaspread': 25},
        {'name': 'low_interest_high_spreads', 'irshift': -200, 'csshift': 100, 'vaspread': 53},
        {'name': 'base_interest_base_spreads', 'irshift': 0, 'csshift': 0, 'vaspread': 25},
        {'name': 'base_interest_high_spreads', 'irshift': 200, 'csshift': 100, 'vaspread': 53},
        {'name': 'high_interest_base_spreads', 'irshift': 200, 'csshift': 0, 'vaspread': 25},
        {'name': 'high_interest_high_spreads', 'irshift': 200, 'csshift': 100, 'vaspread': 53}
    ]
    assert params.scenarios == expected
