"""
Module for bootstrapping zero rate curves.
"""

import numpy as np
import pandas as pd
from math import log, exp, isnan


class Bootstrapping:
    """
    A class to bootstrap zero rate curves from input market data.
    """

    def __init__(self):
        pass

    def newton_raphson_forward_swap(self, fw_guess, swap_rate, m, c, max_iter=500, tol=1e-15):
        """
        Compute the forward swap rate using the Newton-Raphson method.

        Args:
            fw_guess (float): Initial guess for the forward rate.
            swap_rate (float): The swap rate.
            m (int): Number of coupon payments.
            c (float): Constant parameter.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.

        Returns:
            float: The computed forward swap rate.
        """
        fw = fw_guess
        for _ in range(max_iter):
            temp = (1 + fw) ** (-m)
            fx = swap_rate * (1 - temp) / fw + temp - c
            if abs(fx) < tol:
                break
            temp2 = temp / (1 + fw)
            dfx = swap_rate * ((1 + (m + 1) * fw) * temp2 - 1) / (fw ** 2) - m * temp2
            fw -= fx / dfx
        return fw

    def bootstrap_to_zero_full(self, instrument, rates, dlt, coupon_freq, compounding_in, cra, max_tenor, compounding_out):
        """
        Bootstrap the full zero rate curve.

        Args:
            instrument (str): Instrument type ("Zero", "Bond", or "Swap").
            rates (array-like): Input rates.
            dlt (array-like): DLT flags.
            coupon_freq (float): Coupon frequency.
            compounding_in (str): Input compounding.
            cra (float): Credit risk adjustment (in basis points).
            max_tenor (int): Maximum tenor.
            compounding_out (str): Output compounding.

        Returns:
            pd.DataFrame: DataFrame containing the zero curve, forward curve, and discount factors.
        """
        if instrument in ['Swap', 'Bond']:
            return self.bootstrap_swap_to_zero_full(
                swap_rates=rates,
                dlt=dlt,
                coupon_freq=coupon_freq,
                cra=cra,
                max_tenor=max_tenor,
                compounding_out=compounding_out
            )
        elif instrument == 'Zero':
            return self.bootstrap_zero_to_zero_full(
                zero_rates_init=rates,
                dlt=dlt,
                compounding_in=compounding_in,
                cra=cra,
                max_tenor=max_tenor,
                compounding_out=compounding_out
            )
        else:
            raise Exception(f'Instrument {instrument} is not defined.')

    def bootstrap_swap_to_zero_full(self, swap_rates, dlt, coupon_freq, cra, max_tenor, compounding_out):
        """
        Bootstrap the zero curve from swap rates.

        Args:
            swap_rates (array-like): Swap rates (in percentage).
            dlt (array-like): DLT flags.
            coupon_freq (float): Coupon frequency.
            cra (float): Credit risk adjustment in basis points.
            max_tenor (int): Maximum tenor.
            compounding_out (str): Output compounding.

        Returns:
            pd.DataFrame: DataFrame with zero, forward, and discount curves.
        """
        forward = np.zeros(max_tenor)
        discount = np.ones(max_tenor)
        zero = np.zeros(max_tenor)

        valid_tenors = []
        valid_swaps = []
        for i in range(max_tenor):
            if dlt[i] == 1 and not isnan(swap_rates[i]):
                val_dec = swap_rates[i] / 100.0 - cra / 10000.0
                valid_tenors.append(i)
                valid_swaps.append(val_dec)

        if len(valid_tenors) == 0:
            results_dict = {
                'Tenors': np.arange(max_tenor, dtype=int),
                'Zero_CC': zero,
                'Forward_CC': forward,
                'Discount': discount
            }
            return pd.DataFrame(data=results_dict)

        first_idx = valid_tenors[0]
        first_tenor = first_idx + 1
        guess_fw = valid_swaps[0] / coupon_freq

        fwtemp = self.newton_raphson_forward_swap(
            fw_guess=guess_fw,
            swap_rate=(valid_swaps[0] / coupon_freq),
            m=coupon_freq * first_tenor,
            c=1.0
        )

        for i in range(first_idx + 1):
            year = i + 1
            forward[i] = (1 + fwtemp) ** coupon_freq - 1
            discount[i] = 1 / (1 + forward[i]) if i == 0 else discount[i - 1] / (1 + forward[i])
            zero[i] = (1.0 / discount[i]) ** (1.0 / year) - 1

        sumdiscount = (1 - (1 + fwtemp) ** (-coupon_freq * first_tenor)) / fwtemp
        last_idx = first_idx

        for idx in range(1, len(valid_tenors)):
            curr_idx = valid_tenors[idx]
            curr_tenor = curr_idx + 1
            swap_val = valid_swaps[idx] / coupon_freq
            guess_fw = forward[last_idx] / coupon_freq
            m2 = (curr_tenor - (last_idx + 1)) * coupon_freq

            fwtemp = self.newton_raphson_forward_swap(
                fw_guess=guess_fw,
                swap_rate=swap_val,
                m=m2,
                c=(1 - valid_swaps[idx] * sumdiscount) / discount[last_idx]
            )
            dtemp = (1 + fwtemp) ** -1
            sumdiscount += discount[last_idx] * (1 - dtemp ** m2) / fwtemp

            for i in range(last_idx + 1, curr_idx + 1):
                year = i + 1
                forward[i] = (1 + fwtemp) ** coupon_freq - 1
                discount[i] = discount[i - 1] / (1 + forward[i])
                zero[i] = (1.0 / discount[i]) ** (1.0 / year) - 1

            last_idx = curr_idx

        if last_idx < (max_tenor - 1):
            for i in range(last_idx + 1, max_tenor):
                year = i + 1
                forward[i] = forward[last_idx]
                discount[i] = discount[i - 1] / (1 + forward[i])
                zero[i] = (1.0 / discount[i]) ** (1.0 / year) - 1

        if compounding_out == 'C':
            for i in range(max_tenor):
                fval = forward[i]
                zval = zero[i]
                forward[i] = log(1 + fval) if fval > -1 else 0.0
                zero[i] = log(1 + zval) if zval > -1 else 0.0

        results_dict = {
            'Tenors': np.arange(max_tenor, dtype=int),
            'Zero_CC': zero,
            'Forward_CC': forward,
            'Discount': discount
        }
        return pd.DataFrame(data=results_dict)

    def bootstrap_zero_to_zero_full(self, zero_rates_init, dlt, compounding_in, cra, max_tenor, compounding_out):
        """
        Bootstrap the zero curve from initial zero rates.

        Args:
            zero_rates_init (array-like): Initial zero rates.
            dlt (array-like): DLT flags.
            compounding_in (str): Input compounding.
            cra (float): Credit risk adjustment in basis points.
            max_tenor (int): Maximum tenor.
            compounding_out (str): Output compounding.

        Returns:
            pd.DataFrame: DataFrame with zero, forward, and discount curves.
        """
        forward = np.zeros(max_tenor)
        discount = np.ones(max_tenor)
        zero = np.zeros(max_tenor)

        valid_tenors = []
        valid_vals = []
        for i in range(max_tenor):
            if dlt[i] == 1 and not isnan(zero_rates_init[i]):
                dec = zero_rates_init[i] / 100.0 - cra / 10000.0
                if compounding_in == 'A':
                    dec = log(1 + dec)
                valid_tenors.append(i)
                valid_vals.append(dec)

        if len(valid_tenors) == 0:
            results_dict = {
                'Tenors': np.arange(max_tenor, dtype=int),
                'Zero_CC': zero,
                'Forward_CC': forward,
                'Discount': discount
            }
            return pd.DataFrame(data=results_dict)

        first_idx = valid_tenors[0]
        first_val = valid_vals[0]
        for i in range(first_idx + 1):
            year = i + 1
            forward[i] = first_val
            zero[i] = forward[i]
            discount[i] = exp(-year * zero[i])

        for idx in range(1, len(valid_tenors)):
            left_i = valid_tenors[idx - 1]
            right_i = valid_tenors[idx]
            left_val = valid_vals[idx - 1]
            right_val = valid_vals[idx]
            left_year = left_i + 1
            right_year = right_i + 1
            m = right_year - left_year
            fwtemp = ((right_year * right_val) - (left_year * left_val)) / m

            for i in range(left_i + 1, right_i + 1):
                year = i + 1
                forward[i] = fwtemp
                discount[i] = discount[i - 1] * exp(-fwtemp)
                zero[i] = -log(discount[i]) / year

        last_idx = valid_tenors[-1]
        if last_idx < (max_tenor - 1):
            for i in range(last_idx + 1, max_tenor):
                year = i + 1
                forward[i] = forward[last_idx]
                discount[i] = discount[i - 1] * exp(-forward[i])
                zero[i] = -log(discount[i]) / year

        if compounding_out == 'A':
            for i in range(max_tenor):
                fval = forward[i]
                zval = zero[i]
                forward[i] = exp(fval) - 1
                zero[i] = exp(zval) - 1

        results_dict = {
            'Tenors': np.arange(max_tenor, dtype=int),
            'Zero_CC': zero,
            'Forward_CC': forward,
            'Discount': discount
        }
        return pd.DataFrame(data=results_dict)
