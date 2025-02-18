import math
import numpy as np
import pandas as pd
from scipy import optimize


class ExtrapolationSW:
    """
    Class implementing the Smith–Wilson extrapolation method.
    This refactored version calibrates using only liquid market data,
    then preserves the observed (liquid) part of the curve and only extrapolates
    for tenors beyond the last liquid point.
    """

    def __init__(self):
        pass

    def _w(self, t1: float, t2: float, alpha: float, omega: float) -> float:
        """
        Smith–Wilson kernel function.
        
        Args:
            t1 (float): First tenor (in years).
            t2 (float): Second tenor (in years).
            alpha (float): Convergence parameter.
            omega (float): ln(1 + UFR).
            
        Returns:
            float: The kernel value.
        """
        a_min = alpha * min(t1, t2)
        a_max = alpha * max(t1, t2)
        return np.exp(-omega * (t1 + t2)) * (a_min - np.exp(-a_max) * 0.5 * (np.exp(a_min) - np.exp(-a_min)))


    def _w_vector(self, t: float, t_obs: np.ndarray, alpha: float, omega: float) -> np.ndarray:
        """
        Compute the vector of kernel values between tenor t and each calibration tenor.
        
        Args:
            t (float): The tenor at which to evaluate.
            t_obs (np.ndarray): Array of calibration tenors.
            alpha (float): Convergence parameter.
            omega (float): ln(1 + UFR).
            
        Returns:
            np.ndarray: Vector with entries w(t, t_obs[i]).
        """
        return np.array([self._w(t, ti, alpha, omega) for ti in t_obs])


    def _convergence_criterion(
        self, alpha: float, t_obs: np.ndarray, D_obs: np.ndarray, omega: float, CP: float, CR: float
    ) -> float:
        """
        Compute the convergence criterion for a given alpha.

        Args:
            alpha (float): The candidate convergence parameter.
            t_obs (np.ndarray): Calibration tenors.
            D_obs (np.ndarray): Observed discount factors at t_obs.
            omega (float): ln(1 + UFR).
            CP (float): Convergence point (in years).
            CR (float): Convergence radius.

        Returns:
            float: The gap value (should be zero when convergence is achieved).
        """
        n = len(t_obs)
        W = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                W[i, j] = self._w(t_obs[i], t_obs[j], alpha, omega)
        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            raise ValueError(f"Calibration matrix is singular for alpha={alpha}")
        diff = D_obs - np.exp(-omega * t_obs)
        weight_vec = W_inv.dot(diff)
        Qb = np.exp(-omega * t_obs) * weight_vec
        Qb_dot_reft = np.dot(alpha * t_obs, Qb)
        QB_dot_sinh = np.dot(np.sinh(alpha * t_obs), Qb)
        kappa = (1 + Qb_dot_reft) / QB_dot_sinh if QB_dot_sinh != 0 else np.inf
        gap = alpha / abs(1 - np.exp(alpha * CP) * kappa)
        return gap - CR


    def _find_alpha(
        self, t_obs: np.ndarray, D_obs: np.ndarray, omega: float, CP: float, CR: float, alpha_min: float
    ) -> float:
        """
        Determine the optimal alpha using a root finding procedure on the convergence criterion.
        
        Args:
            t_obs (np.ndarray): Calibration tenors.
            D_obs (np.ndarray): Observed discount factors at t_obs.
            omega (float): ln(1 + UFR).
            CP (float): Convergence point (in years).
            CR (float): Convergence radius.
            initial_guess (float): Starting value for alpha.

        Returns:
            float: The calibrated alpha.
        """
        alpha_ini = 0.1
        try:
            sol = optimize.root_scalar(
                lambda alpha: self._convergence_criterion(alpha, t_obs, D_obs, omega, CP, CR),
                x0=alpha_ini, bracket=[alpha_min, 0.5]
            )
            if sol.converged:
                return sol.root
            else:
                print(f'No root could be found in the range [{alpha_min},1]. Setting alpha = {alpha_min}.')
                return alpha_min
        except Exception as e:
            print(e.args)


    def smith_wilson_extrapolation(
        self,
        curve_data: pd.DataFrame,
        UFR: float,
        alpha_min: float,
        CR: float,
        CP: float,
    ) -> pd.DataFrame:
        """
        Perform Smith–Wilson extrapolation.

        Args:
            curve_data (pd.DataFrame): Bootstrapped curve data with columns
                'DLT', 'Tenor', 'Discount', 'Zero_CC'.
            UFR (float): Ultimate forward rate.
            alpha_min (float): Minimum (or starting) alpha value.
            CR (float): Convergence radius in basis points.
            CP (float): Convergence point in years.
        
        Returns:
            pd.DataFrame: DataFrame containing the extrapolated curve with columns:
                'Tenors', 'Zero_CC', 'Forward_CC', 'Discount', 'Zero_AC', 'Forward_AC'.
        """

        # Select liquid market data (where DLT == 1).
        liquid_data = curve_data[curve_data['DLT'] == 1]
        if liquid_data.empty:
            raise ValueError("No liquid data available for calibration.")

        # Extract calibration tenors and observed rates.
        # Assumes 'Tenor' is in years.
        t_obs = liquid_data['Tenors'].values.astype(float)
        r_obs = liquid_data['Zero_CC'].values.astype(float)
        D_obs = liquid_data['Discount'].values.astype(float)

        # Define omega = ln(1 + UFR)
        omega = math.log(1 + UFR)

        # Determine optimal alpha (using the quick method).
        alpha = self._find_alpha(t_obs, D_obs, omega, CP, CR, alpha_min)

        # Calibration: build the Smith–Wilson matrix for calibration points.
        n = len(t_obs)
        W = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                W[i, j] = self._w(t_obs[i], t_obs[j], alpha, omega)
        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            raise ValueError("Calibration matrix is singular; cannot invert.")

        diff = D_obs - np.exp(-omega * t_obs)
        weight_vec = W_inv.dot(diff)

        # Define tenor grid.
        tenors = np.arange(0, len(tenors) + 1, dtype=float)

        # For the liquid part (t <= max(t_obs)), use the observed rates via interpolation.
        interp_func = lambda t: np.interp(t, t_obs, r_obs)

        Zero_CC = np.zeros_like(tenors)
        Discount = np.zeros_like(tenors)
        Forward_CC = np.zeros_like(tenors)

        for idx, t in enumerate(tenors):
            if t == 0:
                # At t = 0 define discount factor 1 and use UFR as the instantaneous rate.
                Discount[idx] = 1.0
                Zero_CC[idx] = UFR
            elif t <= t_obs.max():
                # For the liquid segment, preserve observed rates.
                Zero_CC[idx] = interp_func(t)
                Discount[idx] = np.exp(-Zero_CC[idx] * t)
            else:
                # For t beyond the liquid segment, compute using the SW formula.
                w_vec = self._w_vector(t, t_obs, alpha, omega)
                D_t = np.exp(-omega * t) + np.dot(weight_vec, w_vec)
                Discount[idx] = D_t
                Zero_CC[idx] = -np.log(D_t) / t

        # Compute forward rates from discount factors.
        Forward_CC[0] = Zero_CC[0]
        for idx in range(1, len(tenors)):
            if Discount[idx] > 0 and Discount[idx - 1] > 0:
                Forward_CC[idx] = np.log(Discount[idx - 1] / Discount[idx])
            else:
                Forward_CC[idx] = 0.0

        # For this version, set the "accumulated" (AC) values equal to the continuous compounding (CC) ones.
        Zero_AC = np.exp(Zero_CC) - 1
        Forward_AC = np.exp(Forward_CC) - 1

        output_dict = {
            'Tenors': tenors.astype(int),
            'Zero_CC': Zero_CC,
            'Forward_CC': Forward_CC,
            'Discount': Discount,
            'Zero_AC': Zero_AC,
            'Forward_AC': Forward_AC,
        }
        return pd.DataFrame(data=output_dict)

    def getInputwithVA(self, zero_rates_extrapolated_ac, LLP, VA_value, curve_data: pd.DataFrame) -> pd.DataFrame:
        """
        Incorporate VA into the extrapolated zero curve.
        
        For tenors up to LLP, adjust the zero rates by a parallel shift equal to the VA spread.
        
        Args:
            zero_rates_extrapolated_ac (array-like): Extrapolated (accumulated) zero rates.
            LLP (int): Last liquid point.
            VA_value (float): VA spread in basis points.
            curve_data (pd.DataFrame): Original curve data (used for column names).
            
        Returns:
            pd.DataFrame: DataFrame with VA-adjusted zero curve.
        """
        dlt = np.ones(LLP, dtype=int)
        tenors = np.arange(1, LLP + 1, dtype=int)
        # Add VA spread (converted from basis points) to the extrapolated zeros over the liquid segment.
        zero_rate_withVA = zero_rates_extrapolated_ac[1:LLP + 1] + VA_value / 10000.0
        return pd.DataFrame(data=list(zip(dlt, tenors, zero_rate_withVA)), columns=curve_data.columns)
