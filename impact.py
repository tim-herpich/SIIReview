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

    def _reevaluate_portfolio(self, portfolio_size, duration, zero_rate_alt, zero_rate_new):
        """
        Reevaluate the portfolio under a changing discount rate.

        Args:
            portfolio_size: The portfolio volume.
            discount_rate: The discount rate (in decimal).
            duration: duration of portfolio.

        Returns:
            float: Reevaluated portfolio.
        """
        portfolio_size_reeval = portfolio_size * \
            np.exp(-(zero_rate_new-zero_rate_alt) * duration)
        return portfolio_size_reeval

    def reevaluate_portfolios(self, asset_size, asset_duration, liability_size, liability_duration, discount_curve_SWWithVA, discount_curve_AltWithVA, discount_curve_assets):
        """
        Calculates the imapcts on assets and liabilities under the old (SW + VA) and the new (Alt + VA) discount curves.

        Returns:
            Dataframe that contains imapcts on assets under the old (SW + VA) and the new (Alt + VA) discount curves.
        """
        # Asset impact calculations
        ra_SWWithVA = self._interpolate_rate(
            discount_curve_assets, asset_duration)
        ra_AltWithVA = self._interpolate_rate(
            discount_curve_assets, asset_duration)
        asset_size_reeval = self._reevaluate_portfolio(
            asset_size, asset_duration, ra_SWWithVA, ra_AltWithVA)

        # Liability impact calculations
        rl_SWWithVA = self._interpolate_rate(
            discount_curve_SWWithVA, liability_duration)
        rl_AltWithVA = self._interpolate_rate(
            discount_curve_AltWithVA, liability_duration)
        liability_size_reeval = self._reevaluate_portfolio(
            liability_size, liability_duration, rl_SWWithVA, rl_AltWithVA)

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

    def assess_impact(self, asset_size, asset_duration, liability_size, liability_duration, zero_curve_SWWithVA, zero_curve_AltWithVA, zero_curve_assets):
        """
        Assess the impact on equity by comparing two discount curves.

        Returns:
            Dataframe that contains (reevaluated) asset, liabilities, equities and the respective impacts due to differing discount curves.
        """
        results = self.reevaluate_portfolios(asset_size, asset_duration, liability_size,
                                             liability_duration, zero_curve_SWWithVA, zero_curve_AltWithVA, zero_curve_assets)
        impacts_own_funds = results['Own Funds Reevaluated'].values[0] - \
            results['Own Funds'].values[0]
        impacts_own_funds_rel = impacts_own_funds / \
            (asset_size - liability_size)
        results['Own Funds Impact'] = impacts_own_funds
        results['Own Funds Impact rel.'] = impacts_own_funds_rel
        return results
