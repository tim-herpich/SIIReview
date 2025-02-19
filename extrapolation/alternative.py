"""
Module for alternative (Dutch) extrapolation of zero rate curves.
"""

import math
import numpy as np
import pandas as pd


class ExtrapolationAlt:
    """
    Class implementing alternative extrapolation methods for zero rate curves.
    """

    def __init__(self):
        pass

    def alternative_extrapolation(self, zero_rates, FSP, UFR, LLFR, alpha):
        """
        Perform alternative extrapolation on bootstrapped zero rates.

        Args:
            zero_rates (array-like): Bootstrapped zero rates.
            FSP (int): First Smoothing Point.
            UFR (float): Ultimate Forward Rate.
            LLFR (float): Long-term forward rate.
            alpha (float): Speed of convergence parameter.

        Returns:
            pd.DataFrame: DataFrame with extrapolated zero curve, forward curve, and discount factors.
        """
        max_range = len(zero_rates)
        discount = np.zeros(max_range)
        forwardcc = np.zeros(max_range)
        zerocc = np.zeros(max_range)

        discount[0] = math.exp(-zero_rates[0])
        zerocc[0] = zero_rates[0]
        forwardcc[0] = zerocc[0]

        for i in range(1, FSP):
            year = i + 1
            zerocc[i] = zero_rates[i]
            forwardcc[i] = year * zerocc[i] - (year - 1) * zerocc[i - 1]
            discount[i] = discount[i - 1] * math.exp(-forwardcc[i])

        ln_ufr = math.log(1 + UFR)
        for i in range(FSP, max_range):
            year = i + 1
            fwtemp = ln_ufr + (LLFR - ln_ufr) * (1 - math.exp(-alpha * (year - FSP))) / (alpha * (year - FSP))
            zerocc[i] = (FSP * zerocc[FSP - 1] + (year - FSP) * fwtemp) / year
            discount[i] = math.exp(-year * zerocc[i])
            forwardcc[i] = math.log(discount[i - 1] / discount[i]) if i > 0 else zerocc[i]

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

    def get_llfr(self, zero_rates, dlt, weights):
        """
        Calculate the Long-Term Forward Rate (LLFR).

        Args:
            zero_rates (array-like): Zero rates.
            dlt (array-like): DLT flags.
            weights (array-like): Weights for the LLFR calculation.

        Returns:
            float: Calculated LLFR.
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
        for j in range(fsp_idx):
            if dlt[j] == 1:
                LLPbefore = j

        llfr_val = 0.0
        if LLPbefore >= 0 and fsp_idx > LLPbefore:
            B = fsp_idx + 1
            A = LLPbefore + 1
            seg_fw = ((B * zero_rates[fsp_idx]) - (A * zero_rates[LLPbefore])) / (B - A)
            llfr_val += pos_wts[0] * seg_fw

        for idx in range(1, len(pos_idx)):
            i2 = pos_idx[idx]
            B = i2 + 1
            A = fsp_idx + 1
            seg_fw = ((B * zero_rates[i2]) - (A * zero_rates[fsp_idx])) / (B - A)
            llfr_val += pos_wts[idx] * seg_fw

        return llfr_val

    def get_first_fw_llfr(self, zero_rates, dlt, weights):
        """
        Get the first forward rate segment used for LLFR calculation.

        Args:
            zero_rates (array-like): Zero rates.
            dlt (array-like): DLT flags.
            weights (array-like): Weights.

        Returns:
            float: First forward rate segment.
        """
        pos_idx = [i for i in range(len(weights)) if weights[i] > 0]
        if not pos_idx:
            return 0.0
        fsp_idx = pos_idx[0]

        LLPbefore = -1
        for j in range(fsp_idx):
            if dlt[j] == 1:
                LLPbefore = j

        if LLPbefore < 0 or fsp_idx <= LLPbefore:
            return 0.0

        B = fsp_idx + 1
        A = LLPbefore + 1
        seg_fw = ((B * zero_rates[fsp_idx]) - (A * zero_rates[LLPbefore])) / (B - A)
        return seg_fw

    def zero_boot_withVA(self, fwd_boot_withVA, max_tenor, FSP, VA_value):
        """
        Incorporate VA into the bootstrapped forward curve and compute the zero curve with VA.

        Args:
            fwd_boot_withVA (array-like): Bootstrapped forward rates.
            max_tenor (int): Maximum tenor.
            FSP (int): First Smoothing Point.
            VA_value (float): VA spread in basis points.

        Returns:
            np.array: Zero curve with VA adjustment.
        """
        zero_boot_withVA = np.zeros(max_tenor)
        for i in range(FSP):
            fwd_boot_withVA[i] += math.log(1 + VA_value / 10000)
        for i in range(max_tenor):
            zero_boot_withVA[i] = np.mean(fwd_boot_withVA[:i + 1])
        return zero_boot_withVA
