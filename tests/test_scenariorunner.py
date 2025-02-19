# test/test_scenariorunner.py

"""
This module contains unit tests for the ScenarioRunner class.
"""

import pandas as pd
import pytest
from scenariorunner import ScenarioRunner
from parameters import Parameters


@pytest.fixture
def dummy_alt_data():
    """
    Creates a dummy DataFrame for alternative bootstrapping inputs.

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
    Creates a dummy DataFrame for Smith–Wilson inputs.

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
    Creates a dummy DataFrame for VA spread data with expected issuer labels.
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
    Returns a dummy Parameters object.
    """
    return Parameters()


def test_scenario_runner_outputs(dummy_alt_data, dummy_sw_data, dummy_va_spreads, dummy_params):
    """
    Test that ScenarioRunner.run() returns a curves dictionary and an impact DataFrame
    with the expected structure.
    """
    scenario = {
        "name": "base_interest_base_spreads",
        "irshift": 0,
        "csshift": 0,
        "vaspread": 30
    }
    runner = ScenarioRunner(scenario, dummy_alt_data,
                            dummy_sw_data, dummy_va_spreads, dummy_params)
    curves, impact_df = runner.run()

    # Expected keys in the curves dictionary.
    expected_keys = [
        'Alternative Extrapolation with VA',
        'Alternative Extrapolation',
        'Smith-Wilson Extrapolation with VA',
        'Smith-Wilson Extrapolation'
    ]
    for key in expected_keys:
        assert key in curves, f"Missing curve key: {key}"
        # Each curve should be a DataFrame with a 'Tenors' column.
        assert "Tenors" in curves[key].columns, f"'Tenors' column missing in {key}"

    # Impact DataFrame should contain a 'Maturity' column.
    assert "Maturity" in impact_df.columns, "Impact DataFrame is missing 'Maturity' column"
