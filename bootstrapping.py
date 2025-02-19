"""
This module implements the Bootstrapping class which converts market input data 
(such as swap or zero rates) into a full zero rate curve. The class is initialized 
with all necessary market inputs and parameters.
"""

import numpy as np
import pandas as pd
from math import isnan


class Bootstrapping:
    """
    Bootstraps zero rate curves from market input data.

    The class is initialized with the instrument type, input rates, DLT flags, coupon frequency,
    compounding convention, credit risk adjustment (CRA) and the maximum tenor. It provides the 
    method `bootstrap()` which selects the appropriate bootstrap routine based on the instrument type.
    """

    def __init__(self, instrument: str, rates, dlt, coupon_freq: float, compounding_in: str, cra: float, max_tenor: int):
        """
        Initialize the Bootstrapping object.

        Args:
            instrument (str): Type of instrument ('Swap', 'Bond', or 'Zero').
            rates: Input rates (list or array) in percentage (for swaps) or decimal.
            dlt: Array of DLT flags (liquidity indicators).
            coupon_freq (float): Coupon frequency per year.
            compounding_in (str): Compounding convention of the input rates ('A' for annual, etc.).
            cra (float): Credit risk adjustment in basis points.
            max_tenor (int): Maximum tenor (number of periods) for which to bootstrap.
        """
        self.instrument = instrument
        self.rates = np.array(rates)
        self.dlt = np.array(dlt)
        self.coupon_freq = coupon_freq
        self.compounding_in = compounding_in
        self.cra = cra
        self.max_tenor = max_tenor

    def newton_raphson_forward_swap(self, fw_guess, swap_rate, m, c, max_iter=500, tol=1e-15):
        """
        Solve for the forward swap rate using the Newton–Raphson method.

        This iterative method finds the root of the swap valuation function.

        Args:
            fw_guess (float): Initial guess for the forward rate.
            swap_rate (float): Observed swap rate (converted to decimal).
            m (int): Number of coupon payments for the swap.
            c (float): Constant term in the swap valuation equation.
            max_iter (int, optional): Maximum number of iterations. Default is 500.
            tol (float, optional): Tolerance for convergence. Default is 1e-15.

        Returns:
            float: Converged forward swap rate.
        """
        fw = fw_guess
        for _ in range(max_iter):
            temp = (1 + fw) ** (-m)
            fx = swap_rate * (1 - temp) / fw + temp - c
            if abs(fx) < tol:
                break
            temp2 = temp / (1 + fw)
            dfx = swap_rate * ((1 + (m + 1) * fw) * temp2 -
                               1) / (fw ** 2) - m * temp2
            fw -= fx / dfx
        return fw

    def bootstrap(self) -> pd.DataFrame:
        """
        Bootstrap the zero rate curve based on the instrument type.

        Returns:
            pd.DataFrame: DataFrame containing bootstrapped curves with columns:
                          'Tenors', 'Zero_AC', 'Forward_AC', 'Zero_CC', 'Forward_CC', and 'Discount'.
        """
        if self.instrument in ['Swap', 'Bond']:
            return self._bootstrap_swap_to_zero_full()
        elif self.instrument == 'Zero':
            return self._bootstrap_zero_to_zero_full()
        else:
            raise ValueError(f"Instrument {self.instrument} not recognized.")

    def _bootstrap_swap_to_zero_full(self) -> pd.DataFrame:
        """
        Bootstraps the zero curve from swap rates.

        This method processes valid swap rates (where DLT==1), applies a Newton–Raphson routine
        to determine forward swap rates, computes discount factors and zero rates in both 
        annual and continuous compounding.

        Returns:
            pd.DataFrame: Bootstrapped zero curve.
        """
        forward_ac = np.zeros(self.max_tenor)
        discount = np.ones(self.max_tenor)
        zero_ac = np.zeros(self.max_tenor)
        valid_tenors = []
        valid_swaps = []
        # Collect valid market data points based on DLT flag and rate availability.
        for i in range(self.max_tenor):
            if self.dlt[i] == 1 and not isnan(self.rates[i]):
                val_dec = self.rates[i] / 100.0 - self.cra / 10000.0
                valid_tenors.append(i)
                valid_swaps.append(val_dec)
        if len(valid_tenors) == 0:
            return pd.DataFrame({
                'Tenors': np.arange(self.max_tenor, dtype=int),
                'Zero_AC': zero_ac,
                'Forward_AC': forward_ac,
                'Zero_CC': zero_ac,
                'Forward_CC': forward_ac,
                'Discount': discount
            })
        # Use the first valid tenor as the starting point.
        first_idx = valid_tenors[0]
        first_tenor = first_idx + 1
        guess_fw = valid_swaps[0] / self.coupon_freq
        fwtemp = self.newton_raphson_forward_swap(
            fw_guess=guess_fw,
            swap_rate=(valid_swaps[0] / self.coupon_freq),
            m=self.coupon_freq * first_tenor,
            c=1.0
        )
        # Fill in rates for tenors up to the first valid index.
        for i in range(first_idx + 1):
            year = i + 1
            forward_ac[i] = (1 + fwtemp) ** self.coupon_freq - 1
            discount[i] = 1 / (1 + forward_ac[i]
                               ) if i == 0 else discount[i - 1] / (1 + forward_ac[i])
            zero_ac[i] = (1.0 / discount[i]) ** (1.0 / year) - 1
        # Cumulative discounting for subsequent valid points.
        sumdiscount = (1 - (1 + fwtemp) **
                       (-self.coupon_freq * first_tenor)) / fwtemp
        last_idx = first_idx
        for idx in range(1, len(valid_tenors)):
            curr_idx = valid_tenors[idx]
            curr_tenor = curr_idx + 1
            swap_val = valid_swaps[idx] / self.coupon_freq
            guess_fw = forward_ac[last_idx] / self.coupon_freq
            m2 = (curr_tenor - (last_idx + 1)) * self.coupon_freq
            fwtemp = self.newton_raphson_forward_swap(
                fw_guess=guess_fw,
                swap_rate=swap_val,
                m=m2,
                c=(1 - valid_swaps[idx] * sumdiscount) / discount[last_idx]
            )
            dtemp = (1 + fwtemp) ** -1
            sumdiscount += discount[last_idx] * (1 - dtemp ** m2) / fwtemp
            # Fill in values between last valid index and current index.
            for i in range(last_idx + 1, curr_idx + 1):
                year = i + 1
                forward_ac[i] = (1 + fwtemp) ** self.coupon_freq - 1
                discount[i] = discount[i - 1] / (1 + forward_ac[i])
                zero_ac[i] = (1.0 / discount[i]) ** (1.0 / year) - 1
            last_idx = curr_idx
        # Extrapolate for tenors beyond the last valid index.
        if last_idx < (self.max_tenor - 1):
            for i in range(last_idx + 1, self.max_tenor):
                year = i + 1
                forward_ac[i] = forward_ac[last_idx]
                discount[i] = discount[i - 1] / (1 + forward_ac[i])
                zero_ac[i] = (1.0 / discount[i]) ** (1.0 / year) - 1
        # Convert annual compounding rates to continuous (CC) rates.
        zero_cc = np.log(1 + zero_ac)
        forward_cc = np.log(1 + forward_ac)
        results_dict = {
            'Tenors': np.arange(self.max_tenor, dtype=int),
            'Zero_AC': zero_ac,
            'Forward_AC': forward_ac,
            'Zero_CC': zero_cc,
            'Forward_CC': forward_cc,
            'Discount': discount
        }
        return pd.DataFrame(data=results_dict)

    def _bootstrap_zero_to_zero_full(self) -> pd.DataFrame:
        """
        Bootstraps the zero curve directly from input zero rates.

        This method is used when the input data already consists of zero rates.
        It converts the input into discount factors and then computes the forward
        and zero curves in both annual and continuous compounding.

        Returns:
            pd.DataFrame: Bootstrapped zero curve.
        """
        forward_cc = np.zeros(self.max_tenor)
        discount = np.ones(self.max_tenor)
        zero_cc = np.zeros(self.max_tenor)
        valid_tenors = []
        valid_vals = []
        # Collect valid input zero rates.
        for i in range(self.max_tenor):
            if self.dlt[i] == 1 and not isnan(self.rates[i]):
                dec = self.rates[i] / 100.0 - self.cra / 10000.0
                if self.compounding_in == 'A':
                    # Convert to continuous compounding if necessary.
                    dec = np.log(1 + dec)
                valid_tenors.append(i)
                valid_vals.append(dec)
        if len(valid_tenors) == 0:
            return pd.DataFrame({
                'Tenors': np.arange(self.max_tenor, dtype=int),
                'Zero_AC': zero_cc,
                'Forward_AC': forward_cc,
                'Zero_CC': zero_cc,
                'Forward_CC': forward_cc,
                'Discount': discount
            })
        first_idx = valid_tenors[0]
        first_val = valid_vals[0]
        for i in range(first_idx + 1):
            year = i + 1
            forward_cc[i] = first_val
            zero_cc[i] = forward_cc[i]
            discount[i] = np.exp(-year * zero_cc[i])
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
                forward_cc[i] = fwtemp
                discount[i] = discount[i - 1] * np.exp(-fwtemp)
                zero_cc[i] = -np.log(discount[i]) / year
        zero_ac = np.exp(zero_cc) - 1
        forward_ac = np.exp(forward_cc) - 1
        results_dict = {
            'Tenors': np.arange(1, self.max_tenor + 1, dtype=int),
            'Zero_AC': zero_ac,
            'Forward_AC': forward_ac,
            'Zero_CC': zero_cc,
            'Forward_CC': forward_cc,
            'Discount': discount
        }
        return pd.DataFrame(data=results_dict)
