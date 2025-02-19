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
        self.alpha = float

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
            raise ValueError("W matrix is singular and cannot be inverted.")
        diff = D_obs - np.exp(-omega * t_obs)
        weight_vec = W_inv.dot(diff)
        Qb = np.exp(-omega * t_obs) * weight_vec
        Qb_dot_reft = np.dot(alpha * t_obs, Qb)
        QB_dot_sinh = np.dot(np.sinh(alpha * t_obs), Qb)
        kappa = (1 + Qb_dot_reft) / QB_dot_sinh if QB_dot_sinh != 0 else np.inf
        gap = alpha / abs(1 - np.exp(alpha * CP) * kappa)
        return gap / (CR * 1e-4) - 1

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
                x0=alpha_ini, bracket=[alpha_min, 1]
            )
            if sol.converged:
                print(f'A root could be found. alpha = {sol.root}.')
                return sol.root
            else:
                print(f'No root could be found in the range [{alpha_min}, 1]. Setting alpha = {alpha_min}.')
                return alpha_min
        except Exception as e:
            raise ValueError(e.args)


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
        t_obs = liquid_data['Tenors'].values.astype(int)
        r_obs = liquid_data['Zero_CC'].values.astype(float)
        f_obs = liquid_data['Forward_CC'].values.astype(float)
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
        tenors = len(curve_data['DLT'])
        zerocc = np.zeros(tenors)
        discount = np.zeros(tenors)
        forwardcc = np.zeros(tenors)

        for t in range(tenors):
            year = t + 1
            if t < t_obs[-1]:
                # For the liquid segment, preserve observed rates.
                zerocc[t] = r_obs[t]
                discount[t] = D_obs[t]
                forwardcc[t] = f_obs[t]
            else:
                # For t beyond the liquid segment, compute using the SW formula.
                w_vec = self._w_vector(year, t_obs, alpha, omega)
                D_t = np.exp(-omega * year) + np.dot(weight_vec, w_vec)
                discount[t] = D_t
                zerocc[t] = -np.log(D_t) / year

        # Compute forward rates from discount factors.
        forwardcc[0] = zerocc[0]
        for idx in range(1, tenors):
            if discount[idx] > 0 and discount[idx - 1] > 0:
                forwardcc[idx] = np.log(discount[idx - 1] / discount[idx])
            else:
                forwardcc[idx] = 0.0

        zeroac = np.exp(zerocc) - 1
        forwardac = np.exp(forwardcc) - 1

        output_dict = {
            'Tenors': np.arange(1, tenors+1, dtype=int),
            'Zero_CC': zerocc,
            'Forward_CC': forwardcc,
            'Discount': discount,
            'Zero_AC': zeroac,
            'Forward_AC': forwardac,
        }
        return pd.DataFrame(data=output_dict)

    def addVA(self, results_sw, LLP, VA_value, curve_data: pd.DataFrame) -> pd.DataFrame:
        """
        Incorporate VA into the extrapolated zero curve.

        For tenors up to LLP, adjust the zero and forward rates by a parallel shift equal to the VA spread.
        Adjust the discounting accordingly.

        Args:
            results_sw (df): Extrapolated SW rates.
            LLP (int): Last liquid point.
            VA_value (float): VA spread in basis points.
            curve_data (pd.DataFrame): Original curve data (used for column names).
            
        Returns:
            pd.DataFrame: DataFrame with VA-adjusted zero curve.
        """
        # Add VA spread (converted from basis points) to the extrapolated zeros over the liquid segment.
        results_sw_withVA = results_sw.copy()

        results_sw_withVA.loc[:LLP - 1, 'Zero_AC'] += VA_value / 10000.0
        results_sw_withVA.loc[:LLP - 1, 'Forward_AC'] += VA_value / 10000.0
        results_sw_withVA.loc[:LLP - 1, 'Zero_CC'] = np.log(1 + results_sw_withVA.loc[:LLP - 1, 'Zero_AC'])
        results_sw_withVA.loc[:LLP - 1, 'Forward_CC'] = np.log(1 + results_sw_withVA.loc[:LLP - 1, 'Forward_AC'])
        results_sw_withVA.loc[:LLP - 1, 'Discount'] = np.exp(-results_sw_withVA.loc[:LLP - 1, 'Zero_CC'] * results_sw_withVA.loc[:LLP - 1, 'Tenors'])
        results_sw_withVA['DLT'] = curve_data['DLT'].values
        return results_sw_withVA
