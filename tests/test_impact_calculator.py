import numpy as np
import pandas as pd
import pytest
from impact import ImpactCalculator

# Helper function to create a discount curve DataFrame.
def make_discount_curve(tenors, rate):
    """
    Create a discount curve DataFrame with a constant zero rate.
    Columns: "Tenors" and "Zero_CC".
    """
    return pd.DataFrame({
        "Tenors": tenors,
        "Zero_CC": np.full(len(tenors), rate)
    })

@pytest.fixture
def impact_calc():
    return ImpactCalculator()

# -----------------------------
# Tests for _interpolate_rate
# -----------------------------

def test_interpolate_rate_exact(impact_calc):
    """
    Test _interpolate_rate at a duration that exactly matches a tenor.
    """
    tenors = np.array([0, 1, 2, 3, 4, 5])
    rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    # For duration exactly 3, expect 0.04.
    rate = impact_calc._interpolate_rate(df, 3)
    np.testing.assert_allclose(rate, 0.04, rtol=1e-6)

def test_interpolate_rate_in_between(impact_calc):
    """
    Test _interpolate_rate for a duration between tenors.
    For a linear curve, the interpolated value should follow the linear interpolation.
    """
    tenors = np.array([0, 1, 2, 3, 4, 5])
    rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    # For duration 1.5, expected = 0.02 + 0.5*(0.03 - 0.02) = 0.025.
    rate = impact_calc._interpolate_rate(df, 1.5)
    np.testing.assert_allclose(rate, 0.025, rtol=1e-6)

def test_interpolate_rate_extrapolation_lower(impact_calc):
    """
    Test _interpolate_rate for a duration below the smallest tenor.
    """
    tenors = np.array([1, 2, 3, 4, 5])
    rates = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    # For duration 0 (extrapolation), slope = 0.03 - 0.02 = 0.01.
    # Extrapolated value = 0.02 + (0 - 1)*0.01 = 0.01.
    rate = impact_calc._interpolate_rate(df, 0)
    np.testing.assert_allclose(rate, 0.01, rtol=1e-6)

def test_interpolate_rate_extrapolation_upper(impact_calc):
    """
    Test _interpolate_rate for a duration above the largest tenor.
    """
    tenors = np.array([0, 1, 2, 3])
    rates = np.array([0.01, 0.02, 0.03, 0.04])
    df = pd.DataFrame({"Tenors": tenors, "Zero_CC": rates})
    # For duration 5 (extrapolation), assume slope from the last interval: 0.04-0.03 = 0.01.
    # Extrapolated value = 0.04 + (5-3)*0.01 = 0.04 + 0.02 = 0.06.
    rate = impact_calc._interpolate_rate(df, 5)
    np.testing.assert_allclose(rate, 0.06, rtol=1e-6)

# ----------------------------------
# Tests for _reevaluate_portfolio
# ----------------------------------

def test_reevaluate_portfolio_no_change(impact_calc):
    """
    Test that if zero_rate_alt equals zero_rate_new, the portfolio remains unchanged.
    """
    portfolio_size = 1000.0
    duration = 5.0
    rate = 0.03
    result = impact_calc._reevaluate_portfolio(portfolio_size, duration, rate, rate)
    np.testing.assert_allclose(result, portfolio_size, rtol=1e-6)

def test_reevaluate_portfolio_decrease(impact_calc):
    """
    Test that if zero_rate_new > zero_rate_alt, the portfolio value decreases.
    """
    portfolio_size = 1000.0
    duration = 5.0
    zero_rate_alt = 0.03
    zero_rate_new = 0.04
    expected = portfolio_size * np.exp(-(zero_rate_new - zero_rate_alt) * duration)
    result = impact_calc._reevaluate_portfolio(portfolio_size, duration, zero_rate_alt, zero_rate_new)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_reevaluate_portfolio_increase(impact_calc):
    """
    Test that if zero_rate_new < zero_rate_alt, the portfolio value increases.
    """
    portfolio_size = 1000.0
    duration = 5.0
    zero_rate_alt = 0.04
    zero_rate_new = 0.03
    expected = portfolio_size * np.exp(-(zero_rate_new - zero_rate_alt) * duration)  # exp(0.05)
    result = impact_calc._reevaluate_portfolio(portfolio_size, duration, zero_rate_alt, zero_rate_new)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

# ---------------------------------------
# Tests for reevaluate_portfolios method
# ---------------------------------------

def test_reevaluate_portfolios_no_change(impact_calc):
    """
    Test reevaluate_portfolios when discount curves yield no change.
    Use constant curves for assets and identical curves for liabilities.
    """
    asset_size = 1e6
    asset_duration = 5.0
    liability_size = 0.8e6
    liability_duration = 7.0
    tenors = np.arange(0, 6)  # 0 through 5
    rate_assets = 0.03
    rate_liabs = 0.04
    discount_assets = make_discount_curve(tenors, rate_assets)
    discount_liabs = make_discount_curve(tenors, rate_liabs)
    
    df = impact_calc.reevaluate_portfolios(
        asset_size, asset_duration, liability_size, liability_duration,
        discount_liabs, discount_liabs, discount_assets
    )
    # With identical liability curves for SW and Alt, revaluation leaves values unchanged.
    np.testing.assert_allclose(df.loc['Value', 'Assets Reevaluated'], asset_size, rtol=1e-6)
    np.testing.assert_allclose(df.loc['Value', 'Liabilities Reevaluated'], liability_size, rtol=1e-6)
    np.testing.assert_allclose(df.loc['Value', 'Own Funds Reevaluated'], asset_size - liability_size, rtol=1e-6)

def test_reevaluate_portfolios_with_liability_change(impact_calc):
    """
    Test reevaluate_portfolios when the liability discount curves differ.
    Let the SW curve have a lower rate (0.03) than the Alt curve (0.04), causing revaluation.
    """
    asset_size = 1e6
    asset_duration = 5.0
    liability_size = 0.8e6
    liability_duration = 7.0
    tenors = np.arange(0, 6)
    rate_assets = 0.03
    rate_liabs_SW = 0.03  # lower rate → higher liability value
    rate_liabs_Alt = 0.04
    discount_assets = make_discount_curve(tenors, rate_assets)
    discount_liabs_SW = make_discount_curve(tenors, rate_liabs_SW)
    discount_liabs_Alt = make_discount_curve(tenors, rate_liabs_Alt)
    
    df = impact_calc.reevaluate_portfolios(
        asset_size, asset_duration, liability_size, liability_duration,
        discount_liabs_SW, discount_liabs_Alt, discount_assets
    )
    # Liability revaluation: liability_size * exp(- (rate_SW - rate_Alt) * liability_duration)
    expected_liab_reeval = liability_size * np.exp(- (rate_liabs_Alt - rate_liabs_SW) * liability_duration)
    np.testing.assert_allclose(df.loc['Value', 'Liabilities Reevaluated'], expected_liab_reeval, rtol=1e-6)
    expected_equity = asset_size - expected_liab_reeval
    np.testing.assert_allclose(df.loc['Value', 'Own Funds Reevaluated'], expected_equity, rtol=1e-6)

# ----------------------------
# Tests for assess_impact method
# ----------------------------

def test_assess_impact_zero(impact_calc):
    """
    Test that assess_impact returns zero impact when liability discount curves are identical.
    """
    asset_size = 1e6
    asset_duration = 5.0
    liability_size = 0.8e6
    liability_duration = 7.0
    tenors = np.arange(0, 6)
    rate_assets = 0.03
    rate_liabs = 0.04
    discount_assets = make_discount_curve(tenors, rate_assets)
    discount_liabs = make_discount_curve(tenors, rate_liabs)
    
    df = impact_calc.assess_impact(
        asset_size, asset_duration, liability_size, liability_duration,
        discount_liabs, discount_liabs, discount_assets
    )
    # When both liability curves are identical, there is no change.
    np.testing.assert_allclose(df.loc['Value', 'Own Funds Impact'], 0, rtol=1e-6)
    np.testing.assert_allclose(df.loc['Value', 'Own Funds Impact rel.'], 0, rtol=1e-6)

def test_assess_impact_nonzero(impact_calc):
    """
    Test that assess_impact returns the expected impact when liability discount curves differ.
    Use SW curve with rate 0.03 and Alt curve with rate 0.04.
    """
    asset_size = 1e6
    asset_duration = 5.0
    liability_size = 0.8e6
    liability_duration = 7.0
    tenors = np.arange(0, 6)
    rate_assets = 0.03
    rate_liabs_SW = 0.03
    rate_liabs_Alt = 0.04
    discount_assets = make_discount_curve(tenors, rate_assets)
    discount_liabs_SW = make_discount_curve(tenors, rate_liabs_SW)
    discount_liabs_Alt = make_discount_curve(tenors, rate_liabs_Alt)
    
    df = impact_calc.assess_impact(
        asset_size, asset_duration, liability_size, liability_duration,
        discount_liabs_SW, discount_liabs_Alt, discount_assets
    )
    # Calculate expected liability revaluation:
    expected_liab_reeval = liability_size * np.exp(- (rate_liabs_Alt - rate_liabs_SW) * liability_duration)
    expected_own_funds_reeval = asset_size - expected_liab_reeval
    expected_impact = expected_own_funds_reeval - (asset_size - liability_size)
    expected_rel = expected_impact / (asset_size - liability_size)
    
    np.testing.assert_allclose(df.loc['Value', 'Own Funds Impact'], expected_impact, rtol=1e-6)
    np.testing.assert_allclose(df.loc['Value', 'Own Funds Impact rel.'], expected_rel, rtol=1e-6)

def test_assess_impact_varying_durations(impact_calc):
    """
    Test assess_impact with different asset and liability durations.
    The impact should vary consistently with the duration.
    """
    asset_size = 1e6
    liability_size = 0.8e6
    tenors = np.arange(0, 6)
    rate_assets = 0.03
    rate_liabs_SW = 0.03
    rate_liabs_Alt = 0.04
    discount_assets = make_discount_curve(tenors, rate_assets)
    discount_liabs_SW = make_discount_curve(tenors, rate_liabs_SW)
    discount_liabs_Alt = make_discount_curve(tenors, rate_liabs_Alt)
    
    durations = [3.0, 5.0, 7.0, 10.0]
    impacts = []
    for duration in durations:
        df = impact_calc.assess_impact(
            asset_size, duration, liability_size, duration,
            discount_liabs_SW, discount_liabs_Alt, discount_assets
        )
        impact_abs = df.loc['Value', 'Own Funds Impact']
        impacts.append(impact_abs)
    # With increasing duration, the impact magnitude should increase.
    assert impacts[0] < impacts[1] < impacts[2] < impacts[3]

def test_assess_impact_with_extreme_rates(impact_calc):
    """
    Test assess_impact when the difference between the SW and Alt discount rates is large.
    This simulates a stress scenario.
    """
    asset_size = 1e6
    asset_duration = 5.0
    liability_size = 0.8e6
    liability_duration = 7.0
    tenors = np.arange(0, 6)
    rate_assets = 0.03
    # Use extreme differences for liability curves.
    rate_liabs_SW = 0.02
    rate_liabs_Alt = 0.05
    discount_assets = make_discount_curve(tenors, rate_assets)
    discount_liabs_SW = make_discount_curve(tenors, rate_liabs_SW)
    discount_liabs_Alt = make_discount_curve(tenors, rate_liabs_Alt)
    
    df = impact_calc.assess_impact(
        asset_size, asset_duration, liability_size, liability_duration,
        discount_liabs_SW, discount_liabs_Alt, discount_assets
    )
    impact_abs = df.loc['Value', 'Own Funds Impact']
    impact_rel = df.loc['Value', 'Own Funds Impact rel.']
    # Expect a significant impact.
    assert abs(impact_abs) > 0.01 * (asset_size - liability_size)
    # The relative impact should be a reasonable fraction.
    assert abs(impact_rel) > 0.01

