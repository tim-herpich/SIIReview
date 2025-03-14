"""
Module implementing the ExtrapolationAdditionalAnalyzer class.

This class performs an additional analysis:
  1. For a given market scenario (selected via parameters.additional_analysis_scenario),
     it applies the scenario shifts to the market data (as in regular analysis).
  2. It then performs the regular analysis (bootstrapping and extrapolation without VA)
     using the input spot rates.
  3. Next, it applies a series of IR shocks (specified in parameters.ir_shocks) to the illiquid
     part of the input spot rate curve (t >= LLP or FSP) and recomputes the extrapolated curves
     (both Smith–Wilson and Alternative) without VA.
  4. All resulting curves are stored in a dictionary and returned.

Each method includes detailed docstrings.
"""

import numpy as np
import pandas as pd
from bootstrapping import Bootstrapping
from extrapolation.alternative import ExtrapolationAlt
from extrapolation.smithwilson import ExtrapolationSW


class ExtrapolationAdditionalAnalyzer:
    def __init__(self, cp, df_alt: pd.DataFrame, df_sw: pd.DataFrame):
        """
        Initialize the ExtrapolationAdditionalAnalyzer.

        Args:
            cp: Parameters instance.
            additional_scenario (dict): Market scenario to use for additional analysis.
                If empty, the analysis is skipped.
            df_alt (pd.DataFrame): DataFrame from the 'zero_rates_alt' sheet.
            df_sw (pd.DataFrame): DataFrame from the 'zero_rates_sw' sheet.
        """
        self.cp = cp
        self.additional_scenario = cp.additional_analysis_scenario
        self.df_alt = df_alt.copy()
        self.df_sw = df_sw.copy()
        self.curve_dict = {}

    def adjust_data_for_additional_scenario(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust the market data based on the additional_analysis_scenario.
        This mimics the adjustments performed in ScenarioRunner.

        Args:
            df (pd.DataFrame): Input DataFrame with market data.

        Returns:
            pd.DataFrame: Adjusted DataFrame.
        """
        scenario = self.additional_scenario
        df_adjusted = df.copy()
        # If the scenario name contains 'high_interest' or 'low_interest', shift the rates.
        if 'high_interest' in scenario['name'] or 'low_interest' in scenario['name']:
            shift = scenario['irshift'] / 10000.0
            df_adjusted['Input Rates'] = df_adjusted['Input Rates'] + shift
        return df_adjusted

    def prepare_bootstrap_arrays(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray): # type: ignore
        """
        Prepare arrays for bootstrapping from the adjusted market data.

        Args:
            df (pd.DataFrame): Adjusted DataFrame with columns 'DLT', 'Tenor', 'LLFR Weights', 'Input Rates'.

        Returns:
            Tuple of three numpy arrays: dlt_array, rate_array, weight_array.
        """
        max_tenor = self.cp.max_tenorofAlt
        dlt_array = np.zeros(max_tenor)
        rate_array = np.zeros(max_tenor)
        weight_array = np.zeros(max_tenor)
        for dlt_val, ten, wt, rate in zip(df['DLT'], df['Tenor'], df['LLFR Weights'], df['Input Rates']):
            t = int(round(ten))
            if 1 <= t <= max_tenor:
                idx = t - 1
                dlt_array[idx] = dlt_val
                rate_array[idx] = rate
                weight_array[idx] = wt
        return dlt_array, rate_array, weight_array

    def bootstrap_spot_curve(self, rate_array, dlt_array) -> pd.DataFrame:
        """
        Bootstraps the zero rate curve using the given rate and DLT arrays.

        Args:
            rate_array (np.ndarray): Array of input rates (in percentage).
            dlt_array (np.ndarray): Array of DLT flags.

        Returns:
            pd.DataFrame: Bootstrapped zero curve.
        """
        bootstr = Bootstrapping(self.cp.instrument, rate_array * 100.0, dlt_array,
                                self.cp.coupon_freq, self.cp.compounding_in,
                                self.cp.CRA, self.cp.max_tenorofAlt)
        return bootstr.bootstrap()

    def compute_base_extrapolations(self, df_boot: pd.DataFrame, dlt_array: np.ndarray, weight_array: np.ndarray) -> dict:
        """
        Compute the base extrapolated curves (without VA) from the bootstrapped spot rates.

        Args:
            df_boot (pd.DataFrame): Bootstrapped zero curve.
            dlt_array (np.ndarray): Array of DLT flags.
            weight_array (np.ndarray): Array of weights.

        Returns:
            dict: Dictionary with keys 'SW' and 'Alt' for the extrapolated curves.
        """
        # Alternative extrapolation without VA: LLFR set to zero.
        alt_extrap = ExtrapolationAlt(
            df_boot['Zero_CC'].values, self.cp.FSP, self.cp.UFR, 0, self.cp.alpha)
        llfr_noVA = alt_extrap.get_llfr(dlt_array, weight_array)
        alt_extrap.LLFR = llfr_noVA
        results_alt = alt_extrap.extrapolate()

        # Smith-Wilson extrapolation without VA.
        sw_rates_df = df_boot.copy()
        sw_rates_df['DLT'] = self.df_sw['DLT']
        sw_extrap = ExtrapolationSW(
            sw_rates_df, self.cp.UFR, self.cp.alpha_min_SW, self.cp.CR_SW, self.cp.CP_SW)
        results_sw = sw_extrap.extrapolate()

        return {'SW': results_sw, 'Alt': results_alt}

    def compute_shocked_extrapolations(self, df_boot: pd.DataFrame, dlt_array: np.ndarray, weight_array: np.ndarray) -> dict:
        """
        For each IR shock defined in cp.ir_shocks, apply the shock to the illiquid part of the spot rate curve,
        and recompute the extrapolated curves.

        Args:
            df_boot (pd.DataFrame): Bootstrapped zero curve.
            dlt_array (np.ndarray): Array of DLT flags.
            weight_array (np.ndarray): Array of weights.

        Returns:
            dict: Dictionary with keys like "Shock_50bp" mapping to a dict with keys 'SW' and 'Alt'.
        """
        shock_dict = {}
        LLP = self.cp.LLP_SW  # Define illiquid regime index (using LLP)
        for shock in self.cp.ir_shocks:
            df_shocked = df_boot.copy()
            df_shocked['DLT'] = self.df_sw['DLT']
            shock_decimal = shock / 10000.0
            df_shocked.loc[df_shocked.index >= LLP,
                           'Zero_CC'] = df_shocked.loc[df_shocked.index >= LLP, 'Zero_CC'] + shock_decimal

            # Recompute Alternative extrapolation.
            alt_extrap = ExtrapolationAlt(
                df_shocked['Zero_CC'].values, self.cp.FSP, self.cp.UFR, 0, self.cp.alpha)
            llfr_shock = alt_extrap.get_llfr(dlt_array, weight_array)
            alt_extrap.LLFR = llfr_shock
            results_alt_shock = alt_extrap.extrapolate()

            # Recompute Smith-Wilson extrapolation.
            sw_extrap = ExtrapolationSW(df_shocked.copy(
            ), self.cp.UFR, self.cp.alpha_min_SW, self.cp.CR_SW, self.cp.CP_SW)
            results_sw_shock = sw_extrap.extrapolate()

            key = f"Shock_{shock}bp"
            shock_dict[key] = {'SW': results_sw_shock,
                               'Alt': results_alt_shock}
        return shock_dict

    def run_additional_analysis(self) -> dict:
        """
        Run the additional extrapolation analysis.

        Steps:
          1. Adjust the market data using the selected additional_analysis_scenario.
          2. Prepare bootstrapping arrays and bootstrap the spot rate curve.
          3. Compute the base extrapolated curves (without VA).
          4. For each IR shock, apply the shock to the illiquid regime and recompute extrapolated curves.

        Returns:
            dict: A dictionary containing the base and shocked extrapolated curves.
        """
        # Adjust market data based on additional_analysis_scenario.
        df_alt_adjusted = self.adjust_data_for_additional_scenario(self.df_alt)
        # Prepare bootstrapping arrays.
        dlt_array, rate_array, weight_array = self.prepare_bootstrap_arrays(
            df_alt_adjusted)
        # Bootstrap the spot rate curve.
        df_boot = self.bootstrap_spot_curve(rate_array, dlt_array)
        # Compute base extrapolations without VA.
        base_curves = self.compute_base_extrapolations(
            df_boot, dlt_array, weight_array)
        self.curve_dict['Base'] = base_curves
        # Compute shocked extrapolations.
        shocked_curves = self.compute_shocked_extrapolations(
            df_boot, dlt_array, weight_array)
        self.curve_dict.update(shocked_curves)
        return self.curve_dict
