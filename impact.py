import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class OnwFundsImpactAssessor:
    """
    A class to assess the impact on a portfolio's equity using simplified parameters:
    total sizes and single durations for assets and liabilities.
    """

    def __init__(self):
        pass

    def _interpolate_rate(self, discount_curve, duration):
        """
        Interpolate the discount rate for a given duration based on the discount curve.

        Args:
            discount_curve: Discount curve with 'Tenors' and 'Rates'.
            duration: Duration in years for which to interpolate the rate.

        Returns:
            float: Interpolated discount rate.
        """
        tenors = discount_curve['Tenors'].values
        rates = discount_curve['Zero_CC'].values

        # Create interpolation function (linear interpolation)
        interp_func = interp1d(
            tenors, rates, kind='linear', fill_value='extrapolate')

        # Estimate discount rate
        discount_rate = interp_func(duration)
        return float(discount_rate)

    def _calculate_pv(self, cash_flow, discount_rate, duration):
        """
        Calculate the present value of a single cash flow using continuous compounding.

        Args:
            cash_flow: The cash flow amount.
            discount_rate: The discount rate (in decimal).
            duration: Time in years until the cash flow occurs.

        Returns:
            float: Present value of the cash flow.
        """
        pv = cash_flow * np.exp(-discount_rate * duration)
        return pv

    def calculate_pvs(self, asset_size, asset_duration, liability_size, liability_duration, discount_curve_SWWithVA, discount_curve_AltWithVA, discount_curve_assets):
        """
        Calculate the present values of assets and liabilities under the old (SW + VA) and the new (Alt + VA) discount curves.

        Returns:
            Dataframe that contains PVs under the old (SW + VA) and the new (Alt + VA) discount curves.
        """
        # Old Curve (SW + VA) calculations
        ra_SWWithVA = self._interpolate_rate(
            discount_curve_assets, asset_duration)
        rl_SWWithVA = self._interpolate_rate(
            discount_curve_SWWithVA, liability_duration)

        pv_assets_SWWithVA = self._calculate_pv(
            asset_size, ra_SWWithVA, asset_duration)
        pv_liabilities_SWWithVA = self._calculate_pv(
            liability_size, rl_SWWithVA, liability_duration)
        equity_SWWithVA = pv_assets_SWWithVA - pv_liabilities_SWWithVA

        # New Curve (Alt + New VA) calculations
        ra_AltWithVA = self._interpolate_rate(
            discount_curve_assets, asset_duration)
        rl_AltWithVA = self._interpolate_rate(
            discount_curve_AltWithVA, liability_duration)

        pv_assets_AltWithVA = self._calculate_pv(
            asset_size, ra_AltWithVA, asset_duration)
        pv_liabilities_AltWithVA = self._calculate_pv(
            liability_size, rl_AltWithVA, liability_duration)
        equity_AltWithVA = pv_assets_AltWithVA - pv_liabilities_AltWithVA

        results_dict = {
            'PV_Assets_SWWithVA': pv_assets_SWWithVA,
            'PV_Liabilities_SWWithVA': pv_liabilities_SWWithVA,
            'Equity_SWWithVA': equity_SWWithVA,
            'PV_Assets_AltWithNewVA': pv_assets_AltWithVA,
            'PV_Liabilities_AltWithNewVA': pv_liabilities_AltWithVA,
            'Equity_AltWithNewVA': equity_AltWithVA
        }
        return pd.DataFrame(data=results_dict, index=['Value'])

    def assess_impact(self, asset_size, asset_duration, liability_size, liability_duration, discount_curve_SWWithVA, discount_curve_AltWithVA, discount_curve_assets):
        """
        Assess the impact on equity by comparing two discount curves.

        Returns:
            Dataframe that contains PVs, equities under both curves, and the impact on equity.
        """
        pvs = self.calculate_pvs(asset_size, asset_duration, liability_size,
                                 liability_duration, discount_curve_SWWithVA, discount_curve_AltWithVA, discount_curve_assets)
        impact = pvs['Equity_AltWithNewVA'].values[0] - pvs['Equity_SWWithVA'].values[0]
        results = pvs.copy()
        results['Impact'] = impact

        return results
