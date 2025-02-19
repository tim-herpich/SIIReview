"""
This module implements the ExtrapolationAlt class for performing an alternative 
(Dutch) extrapolation of bootstrapped zero rate curves.
"""

import math
import numpy as np
import pandas as pd


class ExtrapolationAlt:
    """
    Performs alternative extrapolation on bootstrapped zero rate curves.

    The class is initialized with the bootstrapped zero rates and several extrapolation 
    parameters including the First Smoothing Point (FSP), Ultimate Forward Rate (UFR), 
    Long-Term Forward Rate (LLFR) and a convergence speed parameter (alpha). It provides 
    the method `extrapolate()` to generate a full zero curve.
    """

    def __init__(self, zero_rates, FSP: int, UFR: float, LLFR: float, alpha: float):
        """
        Initialize the ExtrapolationAlt object.

        Args:
            zero_rates: Input bootstrapped zero rates (annual compounding, in decimal).
            FSP (int): First Smoothing Point.
            UFR (float): Ultimate Forward Rate.
            LLFR (float): Long-Term Forward Rate.
            alpha (float): Convergence speed parameter.
        """
        self.zero_rates = np.array(zero_rates)
        self.FSP = FSP
        self.UFR = UFR
        self.LLFR = LLFR
        self.alpha = alpha

    def extrapolate(self) -> pd.DataFrame:
        """
        Perform the alternative extrapolation of the zero rate curve.

        The method uses the bootstrapped zero rates up to FSP and then applies a convergence 
        formula to generate rates for tenors beyond FSP.

        Returns:
            pd.DataFrame: DataFrame containing the extrapolated curve with columns:
                          'Tenors', 'Zero_CC', 'Forward_CC', 'Discount', 'Zero_AC', 'Forward_AC'.
        """
        max_range = len(self.zero_rates)
        discount = np.zeros(max_range)
        forwardcc = np.zeros(max_range)
        zerocc = np.zeros(max_range)

        # Initialization for the first tenor.
        discount[0] = math.exp(-self.zero_rates[0])
        zerocc[0] = self.zero_rates[0]
        forwardcc[0] = zerocc[0]

        # For tenors up to FSP, preserve the bootstrapped rates.
        for i in range(1, self.FSP):
            year = i + 1
            zerocc[i] = self.zero_rates[i]
            forwardcc[i] = year * zerocc[i] - (year - 1) * zerocc[i - 1]
            discount[i] = discount[i - 1] * math.exp(-forwardcc[i])

        # Compute the logarithm of (1 + UFR) for use in the convergence formula.
        ln_ufr = math.log(1 + self.UFR)
        # For tenors beyond FSP, use the convergence formula.
        for i in range(self.FSP, max_range):
            year = i + 1
            fwtemp = ln_ufr + (self.LLFR - ln_ufr) * (1 - math.exp(-self.alpha *
                                                                   (year - self.FSP))) / (self.alpha * (year - self.FSP))
            # Blend the rate at FSP and the computed forward rate to obtain the extrapolated zero.
            zerocc[i] = (self.FSP * zerocc[self.FSP - 1] +
                         (year - self.FSP) * fwtemp) / year
            discount[i] = math.exp(-year * zerocc[i])
            forwardcc[i] = math.log(
                discount[i - 1] / discount[i]) if i > 0 else zerocc[i]

        # Convert continuous compounding rates to annual compounding rates.
        zeroac = np.exp(zerocc) - 1
        forwardac = np.exp(forwardcc) - 1

        output_dict = {
            'Tenors': np.arange(1, max_range + 1, dtype=int),
            'Zero_CC': zerocc,
            'Forward_CC': forwardcc,
            'Discount': discount,
            'Zero_AC': zeroac,
            'Forward_AC': forwardac,
        }
        return pd.DataFrame(data=output_dict)

    def get_llfr(self, dlt, weights) -> float:
        """
        Calculate the Long-Term Forward Rate (LLFR) using the extrapolated zero rates.

        This method computes a weighted average of forward segments between the first liquid 
        point (FSP) and subsequent tenors.

        Args:
            dlt: Array of DLT flags indicating liquidity.
            weights: Array of weights used in the LLFR computation (should sum to 1).

        Returns:
            float: The computed LLFR.

        Raises:
            ValueError: If the weights do not sum to 1.
        """
        pos_idx = []
        pos_wts = []
        sum_w = 0.0
        n = len(weights)
        for i in range(n):
            if weights[i] > 0:
                pos_idx.append(i)
                pos_wts.append(weights[i])
                sum_w += weights[i]
        if abs(sum_w - 1.0) > 1e-12:
            raise ValueError("LLFR weights do not sum to 1.")
        if not pos_idx:
            return 0.0
        fsp_idx = pos_idx[0]
        LLPbefore = -1
        # Identify the last liquid point before FSP.
        for j in range(fsp_idx):
            if dlt[j] == 1:
                LLPbefore = j
        llfr_val = 0.0
        if LLPbefore >= 0 and fsp_idx > LLPbefore:
            B = fsp_idx + 1
            A = LLPbefore + 1
            seg_fw = ((B * self.zero_rates[fsp_idx]) -
                      (A * self.zero_rates[LLPbefore])) / (B - A)
            llfr_val += pos_wts[0] * seg_fw
        # Compute weighted contributions from subsequent segments.
        for idx in range(1, len(pos_idx)):
            i2 = pos_idx[idx]
            B = i2 + 1
            A = fsp_idx + 1
            seg_fw = ((B * self.zero_rates[i2]) -
                      (A * self.zero_rates[fsp_idx])) / (B - A)
            llfr_val += pos_wts[idx] * seg_fw
        return llfr_val

    def zero_boot_withVA(self, fwd_boot_withVA, VA_value: float) -> np.array:
        """
        Adjust the bootstrapped forward rates by a parallel VA shift and compute the 
        corresponding zero rate curve.

        Args:
            fwd_boot_withVA: Array of bootstrapped forward rates (continuous compounding).
            VA_value (float): VA spread in basis points.

        Returns:
            np.array: Zero rate curve with the VA adjustment applied.
        """
        zero_boot_withVA = np.zeros(len(fwd_boot_withVA))
        # Apply VA shift for tenors up to FSP.
        for i in range(self.FSP):
            fwd_boot_withVA[i] += math.log(1 + VA_value / 10000)
        # Compute the zero rate as the average of the forward rates up to each tenor.
        for i in range(len(fwd_boot_withVA)):
            zero_boot_withVA[i] = np.mean(fwd_boot_withVA[:i+1])
        return zero_boot_withVA
