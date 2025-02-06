"""
Module for assessing the impact on portfolio equity.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class ImpactCalculator:
    """
    Class to assess the impact on own funds (equity) from changes in discount curves.
    """

    def __init__(self):
        pass

    def _interpolate_rate(self, discount_curve, duration):
        """
        Interpolate the discount rate for a given duration.

        Args:
            discount_curve (pd.DataFrame): DataFrame with 'Tenors' and 'Zero_CC' columns.
            duration (float): Duration in years.

        Returns:
            float: Interpolated discount rate.
        """
        tenors = discount_curve['Tenors'].values
        rates = discount_curve['Zero_CC'].values
        interp_func = interp1d(tenors, rates, kind='linear', fill_value='extrapolate')
        discount_rate = interp_func(duration)
        return float(discount_rate)

    def _reevaluate_portfolio(self, portfolio_size, duration, zero_rate_alt, zero_rate_new):
        """
        Reevaluate the portfolio under a changed discount rate.

        Args:
            portfolio_size (float): The original portfolio size.
            duration (float): Duration in years.
            zero_rate_alt (float): The original (alternative) zero rate.
            zero_rate_new (float): The new zero rate.

        Returns:
            float: The reevaluated portfolio size.
        """
        portfolio_size_reeval = portfolio_size * np.exp(-(zero_rate_new - zero_rate_alt) * duration)
        return portfolio_size_reeval

    def reevaluate_portfolios(self, asset_size, asset_duration, liability_size, liability_duration,
                              discount_curve_SWWithVA, discount_curve_AltWithVA, discount_curve_assets):
        """
        Reevaluates asset and liability portfolios under different discount curves.

        Returns:
            pd.DataFrame: DataFrame with reevaluated portfolio values.
        """
        ra_SWWithVA = self._interpolate_rate(discount_curve_assets, asset_duration)
        ra_AltWithVA = self._interpolate_rate(discount_curve_assets, asset_duration)
        asset_size_reeval = self._reevaluate_portfolio(asset_size, asset_duration, ra_SWWithVA, ra_AltWithVA)

        rl_SWWithVA = self._interpolate_rate(discount_curve_SWWithVA, liability_duration)
        rl_AltWithVA = self._interpolate_rate(discount_curve_AltWithVA, liability_duration)
        liability_size_reeval = self._reevaluate_portfolio(liability_size, liability_duration, rl_SWWithVA, rl_AltWithVA)

        equity_size_reeval = asset_size_reeval - liability_size_reeval

        results_dict = {
            'Assets': asset_size,
            'Zero Rate Assets SW': ra_SWWithVA,
            'Zero Rate Assets Alternative': ra_AltWithVA,
            'Assets Reevaluated': asset_size_reeval,
            'Liabilities': liability_size,
            'Zero Rate Liabilities SW': rl_SWWithVA,
            'Zero Rate Liabilities Alternative': rl_AltWithVA,
            'Liabilities Reevaluated': liability_size_reeval,
            'Own Funds': asset_size - liability_size,
            'Own Funds Reevaluated': equity_size_reeval,
        }
        return pd.DataFrame(data=results_dict, index=['Value'])

    def assess_impact(self, asset_size, asset_duration, liability_size, liability_duration,
                      zero_curve_SWWithVA, zero_curve_AltWithVA, zero_curve_assets):
        """
        Assess the impact on own funds by comparing discount curves.

        Returns:
            pd.DataFrame: DataFrame with impact metrics.
        """
        results = self.reevaluate_portfolios(asset_size, asset_duration, liability_size,
                                             liability_duration, zero_curve_SWWithVA,
                                             zero_curve_AltWithVA, zero_curve_assets)
        impacts_own_funds = results['Own Funds Reevaluated'].values[0] - results['Own Funds'].values[0]
        impacts_own_funds_rel = impacts_own_funds / (asset_size - liability_size)
        results['Own Funds Impact'] = impacts_own_funds
        results['Own Funds Impact rel.'] = impacts_own_funds_rel
        return results
