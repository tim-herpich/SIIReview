"""
tests/test_scenariorunner.py
Unit tests for the ScenarioRunner class.
"""

import pandas as pd
import pytest
from scenariorunner import ScenarioRunner
from parameters import Parameters


@pytest.fixture
def dummy_alt_data():
    """
    Fixture for dummy alternative bootstrapping input DataFrame.
    Columns: DLT, Tenor, LLFR Weights, Input Rates.
    """
    data = {
        "DLT": [1, 1, 1, 1, 1],
        "Tenor": [1, 2, 3, 4, 5],
        "LLFR Weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "Input Rates": [0.02, 0.025, 0.03, 0.035, 0.04]
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_sw_data():
    """
    Fixture for dummy Smith–Wilson input DataFrame.
    Columns: DLT, Tenor, Input Rates.
    """
    data = {
        "DLT": [1, 1, 1, 1, 1],
        "Tenor": [1, 2, 3, 4, 5],
        "Input Rates": [0.015, 0.02, 0.025, 0.03, 0.035]
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_va_spreads():
    """
    Fixture for dummy VA spread DataFrame with expected issuer labels.
    """
    issuers = [
        'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR',
        'HU', 'IS', 'IE', 'IT', 'LV', 'LI', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL',
        'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'CH', 'UK', 'AU', 'CA', 'CN', 'HK',
        'JP', 'US', 'Finan_0', 'Finan_1', 'Finan_2', 'Finan_3', 'Finan_4', 'Finan_5', 'Finan_6',
        'Nonfinan_0', 'Nonfinan_1', 'Nonfinan_2', 'Nonfinan_3', 'Nonfinan_4', 'Nonfinan_5', 'Nonfinan_6'
    ]
    data = {
        '1': [0.002] * len(issuers),
        '2': [0.0025] * len(issuers)
    }
    df = pd.DataFrame(data, index=issuers)
    return df


@pytest.fixture
def dummy_params():
    """
    Fixture for a dummy Parameters object.
    """
    return Parameters()


@pytest.fixture
def test_scenario_runner_outputs(dummy_alt_data, dummy_sw_data, dummy_va_spreads, dummy_params):
    """
    Fixture to run ScenarioRunner and return the curves dictionary and impact DataFrame.
    """
    scenario = {
        "name": "base_interest_base_spreads",
        "irshift": 0,
        "csshift": 0,
        "vaspread": 30
    }
    runner = ScenarioRunner(scenario, dummy_alt_data, dummy_sw_data, dummy_va_spreads, dummy_params)
    curves, impact_df = runner.run()
    return curves, impact_df


def test_scenario_runner_curves_structure(test_scenario_runner_outputs):
    """
    Test that the curves dictionary from ScenarioRunner.run() contains expected keys and structure.
    """
    curves, impact_df = test_scenario_runner_outputs
    expected_keys = [
        'Alternative Extrapolation with VA',
        'Alternative Extrapolation',
        'Smith-Wilson Extrapolation with VA',
        'Smith-Wilson Extrapolation'
    ]
    for key in expected_keys:
        assert key in curves, f"Missing curve key: {key}"
        assert "Tenors" in curves[key].columns, f"'Tenors' column missing in {key}"


def test_scenario_runner_impact_structure(test_scenario_runner_outputs):
    """
    Test that the impact DataFrame from ScenarioRunner.run() has the expected structure.
    """
    curves, impact_df = test_scenario_runner_outputs
    assert "Maturity" in impact_df.columns, "Impact DataFrame is missing 'Maturity' column"
