"""
This module defines the ScenarioRunner class that encapsulates all scenario‐specific 
processing for bootstrapping, extrapolation (both alternative and Smith–Wilson), VA 
adjustments, and impact calculations.
"""

import numpy as np
import pandas as pd
from bootstrapping import Bootstrapping
from extrapolation.smithwilson import ExtrapolationSW
from extrapolation.alternative import ExtrapolationAlt
from va import VaSpreadCalculator
from impact import ImpactCalculator


class ScenarioRunner:
    """
    Processes a single market scenario.

    This class performs the following steps:
      1. Adjusts raw market data based on scenario-specific shifts.
      2. Prepares input arrays for bootstrapping.
      3. Bootstraps the zero curve.
      4. Performs alternative extrapolation:
           - First, computes LLFR without VA adjustment.
           - Then, applies a VA shift to the bootstrapped forward rates,
             recomputes a new LLFR (LLFRwithVA), and re-extrapolates.
      5. Performs Smith–Wilson extrapolation:
           - First, runs a standard SW extrapolation.
           - Then, adjusts the liquid portion (up to LLP) with the VA spread and 
             re-runs the SW extrapolation.
      6. Computes impact measures (e.g. present value of a unit ZCB) using the two methods.

    Attributes:
        scenario (dict): Scenario parameters (e.g. name, interest rate shift, spread shift, VA spread).
        df_alt (pd.DataFrame): Alternative bootstrapping input data.
        df_sw (pd.DataFrame): Smith–Wilson input data.
        va_spreads_df (pd.DataFrame): VA spread data.
        params (object): Global parameters (including market scenarios).
    """

    def __init__(self, scenario, df_alt, df_sw, va_spreads_df, params):
        """
        Initialize a ScenarioRunner instance.

        Args:
            scenario (dict): Scenario parameters.
            df_alt (pd.DataFrame): DataFrame with alternative (bootstrapped) curve input data.
            df_sw (pd.DataFrame): DataFrame with Smith–Wilson specific curve input data.
            va_spreads_df (pd.DataFrame): DataFrame with VA spread data.
            params (object): Global parameters and settings.
        """
        self.scenario = scenario
        self.df_alt = df_alt.copy()
        self.df_sw = df_sw.copy()
        self.va_spreads_df = va_spreads_df.copy()
        self.params = params

    def adjust_data(self):
        """
        Adjust the market data per scenario.

        - Shifts interest rates (converting basis points to decimal) for scenarios 
          indicating high or low interest.
        - Adjusts VA spreads for scenarios with high spreads.
        - Sets the legacy VA spread value in the global parameters.
        """
        if 'high_interest' in self.scenario['name'] or 'low_interest' in self.scenario['name']:
            shift = self.scenario['irshift'] / 10000.0
            self.df_alt['Input Rates'] += shift
            self.df_sw['Input Rates'] += shift
        if 'high_spreads' in self.scenario['name']:
            self.va_spreads_df += self.scenario['csshift'] / 10000.0
        # Update the legacy VA value in the parameters.
        self.params.VA_value = self.scenario['vaspread']

    def run(self):
        """
        Run all processing steps for the scenario.

        Returns:
            tuple: (curves, impact_df) where curves is a dictionary of extrapolated curves,
                   and impact_df is a DataFrame containing impact analysis results.
        """
        self.adjust_data()

        # --- Prepare Bootstrapping Inputs ---
        max_tenor = self.params.max_tenorofAlt
        dlt_array = np.zeros(max_tenor)
        rate_array = np.zeros(max_tenor)
        weight_array = np.zeros(max_tenor)
        for dlt_val, tenor, wt, rate in zip(self.df_alt['DLT'],
                                            self.df_alt['Tenor'],
                                            self.df_alt['LLFR Weights'],
                                            self.df_alt['Input Rates']):
            t = int(round(tenor))
            if 1 <= t <= max_tenor:
                idx = t - 1
                dlt_array[idx] = dlt_val
                rate_array[idx] = rate
                weight_array[idx] = wt

        # --- Bootstrapping ---
        bootstrapper = Bootstrapping(
            instrument=self.params.instrument,
            rates=rate_array * 100.0,      # convert to percentage
            dlt=dlt_array,
            coupon_freq=self.params.coupon_freq,
            compounding_in=self.params.compounding_in,
            cra=self.params.CRA,
            max_tenor=self.params.max_tenorofAlt
        )
        boot_df = bootstrapper.bootstrap()

        # --- Alternative Extrapolation without VA ---
        # Create an instance using the bootstrapped zero rates.
        alt_extrap = ExtrapolationAlt(
            zero_rates=boot_df['Zero_CC'].values,
            FSP=self.params.FSP,
            UFR=self.params.UFR,
            LLFR=0,  # placeholder; will be computed below
            alpha=self.params.alpha
        )
        # Compute LLFR from the original bootstrapped zero curve.
        llfr_noVA = alt_extrap.get_llfr(dlt_array, weight_array)
        alt_extrap.LLFR = llfr_noVA  # store as the non-VA LLFR
        results_alt = alt_extrap.extrapolate()

        # --- Alternative Extrapolation with VA ---
        # Compute the new VA spread.
        va_calc = VaSpreadCalculator(
            va_spreads_df=self.va_spreads_df,
            fi_asset_size=self.params.fi_asset_size,
            liability_size=self.params.liability_size,
            pvbp_fi_assets=self.params.pvbp_fi_assets,
            pvbp_liabs=self.params.pvbp_liabs
        )
        va_new = va_calc.compute_total_va()
        # Apply the VA shift to the bootstrapped forward rates.
        zero_boot_with_new_VA = alt_extrap.zero_boot_withVA(
            boot_df['Forward_CC'].values, va_new)
        # Create a new instance of ExtrapolationAlt using the VA-adjusted zero rates.
        alt_extrap_va = ExtrapolationAlt(
            zero_rates=zero_boot_with_new_VA,
            FSP=self.params.FSP,
            UFR=self.params.UFR,
            LLFR=0,  # placeholder for LLFR with VA
            alpha=self.params.alpha
        )
        # Compute LLFR based on the VA-adjusted zero curve.
        llfr_withVA = alt_extrap_va.get_llfr(dlt_array, weight_array)
        alt_extrap_va.LLFR = llfr_withVA  # store as the VA-adjusted LLFR
        results_alt_with_VA = alt_extrap_va.extrapolate()

        # --- Smith–Wilson Extrapolation with VA ---
        # First, run SW extrapolation on the original SW data.
        sw_rates_df = boot_df.copy()
        sw_rates_df['DLT'] = self.df_sw['DLT']
        sw_extrap = ExtrapolationSW(
            curve_data=sw_rates_df,
            UFR=self.params.UFR,
            alpha_min=self.params.alpha_min_SW,
            CR=self.params.CR_SW,
            CP=self.params.CP_SW
        )
        results_sw = sw_extrap.extrapolate()
        # Now, create a VA-adjusted input for SW:
        df_sw_withVA = sw_extrap.add_va(
            self.params.LLP_SW, self.params.VA_value)
        # Re-run SW extrapolation using the VA-adjusted data.
        sw_extrap_va = ExtrapolationSW(
            curve_data=df_sw_withVA,
            UFR=self.params.UFR,
            alpha_min=self.params.alpha_min_SW,
            CR=self.params.CR_SW,
            CP=self.params.CP_SW
        )
        results_sw_with_VA = sw_extrap_va.extrapolate()

        # --- Impact Analysis ---
        # Compute present values of a unit zero-coupon bond (ZCB) for a range of maturities.
        impact_calc = ImpactCalculator()
        impact_results = {
            "Maturity": [],
            "PV Alternative Extrapolation": [],
            "PV Smith-Wilson Extrapolation": [],
            "PV (Alt - SW)": []
        }
        # For maturities from LLP to CP_SW in steps of 5 years:
        for maturity in range(self.params.LLP_SW, self.params.CP_SW + 1, 5):
            # Note: Use the appropriate zero curves with VA.
            pv_alt = impact_calc.compute_zcb_pv(
                results_alt_with_VA, maturity, self.params.LLP_SW)
            pv_sw = impact_calc.compute_zcb_pv(
                results_sw_with_VA, maturity, self.params.LLP_SW)
            impact_results["Maturity"].append(maturity)
            impact_results["PV Alternative Extrapolation"].append(pv_alt)
            impact_results["PV Smith-Wilson Extrapolation"].append(pv_sw)
            impact_results["PV (Alt - SW)"].append(pv_alt - pv_sw)
        impact_df = pd.DataFrame(impact_results)

        # Prepare a dictionary of all resulting curves.
        curves = {
            'Alternative Extrapolation with VA': results_alt_with_VA,
            'Alternative Extrapolation': results_alt,
            'Smith-Wilson Extrapolation with VA': results_sw_with_VA,
            'Smith-Wilson Extrapolation': results_sw
        }
        return curves, impact_df
