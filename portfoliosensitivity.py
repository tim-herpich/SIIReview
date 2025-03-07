"""
Module for performing sensitivity analysis on liability valuation.
Loops over asset-size and liability-duration ratios, updates dependent parameters,
runs the scenario processing and computes the present value (PV) of a unit cash flow 
at selected maturities.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from scenariorunner import ScenarioRunner
from impact import ImpactCalculator
import warnings


class PortfolioSensitivity:
    """
    Class to perform sensitivity analysis on portfolio characteristics.
    Varies the asset size and liability duration based on provided ranges and iteration step size.
    """

    def __init__(self, df_alt: pd.DataFrame, df_sw: pd.DataFrame, va_spreads_df: pd.DataFrame, base_params):
        """
        Initialize the SensitivityAnalyzer.

        Args:
            df_alt (pd.DataFrame): DataFrame for alternative curve input.
            df_sw (pd.DataFrame): DataFrame for SW curve input.
            va_spreads_df (pd.DataFrame): DataFrame containing VA spreads.
            base_params: Base parameters object containing sensitivity configuration.
        """
        self.df_alt = df_alt
        self.df_sw = df_sw
        self.va_spreads_df = va_spreads_df
        self.base_params = base_params
        # Base portfolio characteristics
        self.base_liability_size = base_params.liability_size
        self.base_liability_duration = base_params.liability_duration
        # Sensitivity analysis configuration from parameters
        # e.g., [10, 40]
        self.sensitivity_tenors = base_params.sensitivity_tenors
        # tuple (min, max)
        self.asset_ratio_range = base_params.sensitivity_asset_ratio_range
        # tuple (min, max)
        self.asset_duration_ratio_range = base_params.sensitivity_asset_duration_ratio_range
        self.num_steps = base_params.sensitivity_num_steps

    def run_analysis(self) -> pd.DataFrame:
        """
        Run sensitivity analysis over asset size and asset duration.

        Returns:
            pd.DataFrame: DataFrame with columns:
              'Asset/Liability Ratio', 'Asset Duration/Liability Duration Ratio', and PV delta for each tenor.
        """
        results = []
        asset_ratios = np.linspace(
            self.asset_ratio_range[0], self.asset_ratio_range[1], self.num_steps)
        duration_ratios = np.linspace(
            self.asset_duration_ratio_range[0], self.asset_duration_ratio_range[1], self.num_steps)

        # Always use the base market scenario: base_interest_base_spreads
        base_scenario = next(
            (s for s in self.base_params.scenarios if s['name'] == 'base_interest_base_spreads'), None)
        if base_scenario is None:
            base_scenario = {'name': 'base_interest_base_spreads',
                             'irshift': 0, 'csshift': 0, 'vaspread': 25}

        for ar in asset_ratios:
            for dr in duration_ratios:
                # Create a copy of the base parameters and update the portfolio characteristics.
                params = deepcopy(self.base_params)
                params.asset_size = self.base_liability_size * ar
                params.liability_size = self.base_liability_size  # fixed
                # Here we vary the asset duration while keeping liability duration fixed.
                params.asset_duration = self.base_liability_duration * dr
                params.liability_duration = self.base_liability_duration  # fixed

                # Recalculate dependent variables.
                params.fi_asset_size = 0.62 * params.asset_size
                params.pvbp_fi_assets = 0.1 * params.asset_duration
                params.pvbp_liabs = 0.1 * params.liability_duration

                try:
                    # Run the scenario with the base market scenario.
                    runner = ScenarioRunner(
                        base_scenario, self.df_alt, self.df_sw, self.va_spreads_df, params)
                    curves, impact_df = runner.run()
                    # Expect curves to be a dictionary with two keys: one for Alternative with VA and one for SW with VA.
                    if isinstance(curves, dict):
                        discount_curve_alt = curves.get('Alternative', None)
                        discount_curve_sw = curves.get('Smith-Wilson', None)
                        if discount_curve_alt is None or discount_curve_sw is None:
                            # Fallback: if keys are not present, assume first two curves are alt and sw.
                            values = list(curves.values())
                            discount_curve_alt = values[0]
                            discount_curve_sw = values[1] if len(
                                values) > 1 else values[0]
                    else:
                        # If curves is not a dictionary, use it for both.
                        discount_curve_alt = discount_curve_sw = curves

                    impact_calc = ImpactCalculator()
                    pv_delta_values = {}
                    for m in self.sensitivity_tenors:
                        pv_alt = impact_calc.compute_zcb_pv(
                            discount_curve_alt, m)
                        pv_sw = impact_calc.compute_zcb_pv(
                            discount_curve_sw, m)
                        pv_delta_values[f'PV (Alt - SW) at {int(m)} Years'] = pv_alt - pv_sw
                    # Also record the asset duration ratio (asset_duration / liability_duration)
                    ratio_duration = params.asset_duration / params.liability_duration
                    result = {
                        'Asset/Liability Ratio': ar,
                        'Asset Duration/Liability Duration Ratio': ratio_duration
                    }
                    result.update(pv_delta_values)
                    results.append(result)
                except Exception as e:
                    warnings.warn(
                        f"Sensitivity analysis failed for AR={ar:.2f}, AssetDurRatio={dr:.2f}: {e}")
        df_results = pd.DataFrame(results)
        return df_results
