"""
Module for computing the present value of a unit face value zero-coupon bond (ZCB)
using a given discount curve.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class ImpactCalculator:
    """
    Class to compute the present value (PV) of a unit face value ZCB using a discount curve.
    """

    def __init__(self):
        pass

    def _interpolate_rate(self, discount_curve: pd.DataFrame, maturity: float) -> float:
        """
        Interpolate the zero rate for a given maturity from the discount curve.

        Args:
            discount_curve (pd.DataFrame): DataFrame with columns 'Tenors' and 'Zero_CC'.
            maturity (float): Maturity in years.

        Returns:
            float: Interpolated zero rate.
        """
        tenors = discount_curve['Tenors'].values
        rates = discount_curve['Zero_CC'].values
        interp_func = interp1d(tenors, rates, kind='linear', fill_value='extrapolate')
        discount_rate = interp_func(maturity)
        return float(discount_rate)

    def compute_zcb_pv(self, discount_curve: pd.DataFrame, maturity: float, llp: float) -> float:
        """
        Compute the present value of a unit face value zero-coupon bond with the given maturity,
        using the zero rate obtained from the discount curve.

        Args:
            discount_curve (pd.DataFrame): DataFrame with columns 'Tenors' and 'Zero_CC'.
            maturity (float): Maturity in years.

        Returns:
            float: Present value of the ZCB.
        """
        zero_rate = self._interpolate_rate(discount_curve, maturity)
        pv = np.exp(-zero_rate * (maturity-llp))
        return pv
