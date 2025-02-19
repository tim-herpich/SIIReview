# tests/test_parameters.py

import pytest
from parameters import Parameters

@pytest.fixture
def params():
    return Parameters()

def test_compounding_in(params):
    assert params.compounding_in == 'A'

def test_compounding_out(params):
    assert params.compounding_out == 'C'

def test_instrument(params):
    assert params.instrument == 'Zero'

def test_VA_value_is_float_class(params):
    # The default value is the built-in float (i.e. the type)
    assert params.VA_value is float

def test_coupon_freq(params):
    assert params.coupon_freq == 1.0

def test_FSP(params):
    assert params.FSP == 20

def test_UFR(params):
    assert params.UFR == 0.033

def test_CRA(params):
    assert params.CRA == 10.0

def test_alpha_min_SW(params):
    assert params.alpha_min_SW == 0.05

def test_CR_SW(params):
    assert params.CR_SW == 1.0

def test_CP_SW(params):
    assert params.CP_SW == 60

def test_LLP_SW(params):
    assert params.LLP_SW == 20

def test_max_tenorofAlt(params):
    assert params.max_tenorofAlt == 150

def test_asset_size(params):
    assert params.asset_size == 1e6

def test_asset_duration(params):
    assert params.asset_duration == 6.8

def test_liability_size(params):
    assert params.liability_size == 0.8e6

def test_liability_duration(params):
    assert params.liability_duration == 10.0

def test_fi_asset_size(params):
    assert params.fi_asset_size == 0.62 * params.asset_size

def test_pvbp_fi_assets(params):
    assert params.pvbp_fi_assets == 0.1 * params.asset_duration

def test_pvbp_liabs(params):
    assert params.pvbp_liabs == 0.1 * params.liability_duration

def test_scenarios_is_list(params):
    assert isinstance(params.scenarios, list)

def test_scenarios_length(params):
    assert len(params.scenarios) > 0

def test_scenarios_content(params):
    expected = [
        {'name': 'low_interest_base_spreads', 'irshift': -200, 'csshift': 0, 'vaspread': 27},
        {'name': 'low_interest_high_spreads', 'irshift': -200, 'csshift': 100, 'vaspread': 45},
        {'name': 'base_interest_base_spreads', 'irshift': 0, 'csshift': 0, 'vaspread': 27},
        {'name': 'base_interest_high_spreads', 'irshift': 200, 'csshift': 100, 'vaspread': 45},
        {'name': 'high_interest_base_spreads', 'irshift': 200, 'csshift': 0, 'vaspread': 27},
        {'name': 'high_interest_high_spreads', 'irshift': 200, 'csshift': 100, 'vaspread': 45}
    ]
    assert params.scenarios == expected

