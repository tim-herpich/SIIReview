"""
This module implements the ExtrapolationSW class that performs Smith–Wilson extrapolation.
The class is initialized with curve data and SW-specific parameters and provides methods 
to calibrate the convergence parameter (alpha), generate the full zero curve, and apply 
VA adjustments.
"""

import math
import numpy as np
import pandas as pd
from scipy import optimize


class ExtrapolationSW:
    """
    Implements the Smith–Wilson extrapolation method.

    The class calibrates the convergence parameter using only liquid market data 
    (where DLT==1), and then extrapolates the zero curve for all tenors. It also 
    supports the application of a Valuation Adjustment (VA) to the liquid segment.
    """

    def __init__(self, curve_data: pd.DataFrame, UFR: float, alpha_min: float, CR: float, CP: float):
        """
        Initialize the ExtrapolationSW object.

        Args:
            curve_data (pd.DataFrame): DataFrame containing bootstrapped curve data with columns 
                                       including 'DLT', 'Tenors', 'Zero_CC', 'Discount', and optionally 'Forward_CC'.
            UFR (float): Ultimate Forward Rate.
            alpha_min (float): Minimum (or starting) value for the convergence parameter alpha.
            CR (float): Convergence radius (in basis points).
            CP (float): Convergence point (in years).
        """
        self.curve_data = curve_data
        self.UFR = UFR
        self.alpha_min = alpha_min
        self.CR = CR
        self.CP = CP

    def _w(self, t1: float, t2: float, alpha: float, omega: float) -> float:
        """
        Compute the Smith–Wilson kernel function for two tenors.

        The kernel function is defined as:

            w(t1, t2) = exp(-omega*(t1+t2)) * [alpha*min(t1,t2) - exp(-alpha*max(t1,t2)) * 0.5 * (exp(alpha*min(t1,t2)) - exp(-alpha*min(t1,t2)))]

        Args:
            t1 (float): First tenor (in years).
            t2 (float): Second tenor (in years).
            alpha (float): Convergence parameter.
            omega (float): ln(1 + UFR).

        Returns:
            float: Kernel function value.
        """
        a_min = alpha * min(t1, t2)
        a_max = alpha * max(t1, t2)
        return np.exp(-omega * (t1 + t2)) * (a_min - np.exp(-a_max) * 0.5 * (np.exp(a_min) - np.exp(-a_min)))

    def _w_vector(self, t: float, t_obs: np.ndarray, alpha: float, omega: float) -> np.ndarray:
        """
        Compute the kernel function vector for a given tenor against an array of calibration tenors.

        Args:
            t (float): The tenor for which the vector is computed.
            t_obs (np.ndarray): Array of calibration tenors.
            alpha (float): Convergence parameter.
            omega (float): ln(1 + UFR).

        Returns:
            np.ndarray: Array of kernel function values.
        """
        return np.array([self._w(t, ti, alpha, omega) for ti in t_obs])

    def _convergence_criterion(self, alpha: float, t_obs: np.ndarray, D_obs: np.ndarray, omega: float) -> float:
        """
        Evaluate the convergence criterion for a given candidate alpha.

        This function builds the calibration matrix W, computes the weight vector, 
        and calculates the gap between the computed value and the target convergence.

        Args:
            alpha (float): Candidate convergence parameter.
            t_obs (np.ndarray): Calibration tenors.
            D_obs (np.ndarray): Observed discount factors at t_obs.
            omega (float): ln(1 + UFR).

        Returns:
            float: The gap (should be zero at convergence), scaled by the convergence radius.
        """
        n = len(t_obs)
        W = np.empty((n, n))
        # Build the calibration matrix W using the kernel function.
        for i in range(n):
            for j in range(n):
                W[i, j] = self._w(t_obs[i], t_obs[j], alpha, omega)
        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            raise ValueError("W matrix is singular.")
        # Calculate the difference between observed discount factors and the target function.
        diff = D_obs - np.exp(-omega * t_obs)
        weight_vec = W_inv.dot(diff)
        # Compute Qb vector as part of the calibration.
        Qb = np.exp(-omega * t_obs) * weight_vec
        Qb_dot_reft = np.dot(alpha * t_obs, Qb)
        QB_dot_sinh = np.dot(np.sinh(alpha * t_obs), Qb)
        kappa = (1 + Qb_dot_reft) / QB_dot_sinh if QB_dot_sinh != 0 else np.inf
        gap = alpha / abs(1 - np.exp(alpha * self.CP) * kappa)
        # Scale the gap by the convergence radius (converted from basis points) and subtract 1.
        return gap / (self.CR * 1e-4) - 1

    def _find_alpha(self, t_obs: np.ndarray, D_obs: np.ndarray, omega: float) -> float:
        """
        Calibrate the convergence parameter alpha using a root-finding procedure.

        This method uses SciPy's root_scalar to solve the equation defined by the convergence criterion.

        Args:
            t_obs (np.ndarray): Calibration tenors.
            D_obs (np.ndarray): Observed discount factors.
            omega (float): ln(1 + UFR).

        Returns:
            float: Calibrated alpha value.
        """
        alpha_ini = 0.1  # Initial guess
        sol = optimize.root_scalar(
            lambda a: self._convergence_criterion(a, t_obs, D_obs, omega),
            x0=alpha_ini, bracket=[self.alpha_min, 1]
        )
        return sol.root if sol.converged else self.alpha_min

    def extrapolate(self) -> pd.DataFrame:
        """
        Perform Smith–Wilson extrapolation of the zero curve.

        The method uses only the liquid data (where DLT==1) for calibration, determines the optimal alpha,
        and then extrapolates the zero rate for all tenors using the Smith–Wilson formula.

        Returns:
            pd.DataFrame: DataFrame containing the extrapolated zero curve with columns:
                          'Tenors', 'Zero_CC', 'Forward_CC', 'Discount', 'Zero_AC', and 'Forward_AC'.

        Raises:
            ValueError: If no liquid data is available or if the calibration matrix is singular.
        """
        # Select only the liquid points for calibration.
        liquid_data = self.curve_data[self.curve_data['DLT'] == 1]
        if liquid_data.empty:
            raise ValueError("No liquid data available for calibration.")
        t_obs = liquid_data['Tenors'].values.astype(int)
        r_obs = liquid_data['Zero_CC'].values.astype(float)
        D_obs = liquid_data['Discount'].values.astype(float)
        omega = math.log(1 + self.UFR)
        # Calibrate alpha.
        alpha = self._find_alpha(t_obs, D_obs, omega)
        n = len(t_obs)
        W = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                W[i, j] = self._w(t_obs[i], t_obs[j], alpha, omega)
        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            raise ValueError("Calibration matrix is singular.")
        diff = D_obs - np.exp(-omega * t_obs)
        weight_vec = W_inv.dot(diff)
        total_tenors = len(self.curve_data)
        zerocc = np.zeros(total_tenors)
        discount = np.zeros(total_tenors)
        forwardcc = np.zeros(total_tenors)
        # For each tenor, either preserve the liquid data or compute the extrapolated values.
        for t in range(total_tenors):
            year = t + 1
            if t < t_obs[-1]:
                zerocc[t] = r_obs[t]
                discount[t] = D_obs[t]
                forwardcc[t] = self.curve_data.iloc[t]['Forward_CC'] if 'Forward_CC' in self.curve_data.columns else 0
            else:
                w_vec = self._w_vector(year, t_obs, alpha, omega)
                D_t = np.exp(-omega * year) + np.dot(weight_vec, w_vec)
                discount[t] = D_t
                zerocc[t] = -np.log(D_t) / year
        # Compute forward rates from discount factors.
        forwardcc[0] = zerocc[0]
        for t in range(1, total_tenors):
            if discount[t] > 0 and discount[t-1] > 0:
                forwardcc[t] = np.log(discount[t-1] / discount[t])
            else:
                forwardcc[t] = 0
        zeroac = np.exp(zerocc) - 1
        forwardac = np.exp(forwardcc) - 1
        output_dict = {
            'Tenors': np.arange(1, total_tenors + 1, dtype=int),
            'Zero_CC': zerocc,
            'Forward_CC': forwardcc,
            'Discount': discount,
            'Zero_AC': zeroac,
            'Forward_AC': forwardac
        }
        return pd.DataFrame(data=output_dict)

    def add_va(self, LLP: int, VA_value: float) -> pd.DataFrame:
        """
        Apply a Valuation Adjustment (VA) to the extrapolated zero curve.

        This method adds a parallel VA shift (converted from basis points) to the zero 
        and forward rates for tenors up to the Last Liquid Point (LLP) and recalculates 
        the corresponding discount factors.

        Args:
            LLP (int): Last Liquid Point (number of periods) up to which VA is applied.
            VA_value (float): VA spread in basis points.

        Returns:
            pd.DataFrame: VA-adjusted zero curve.
        """
        df = self.extrapolate().copy()
        # Apply the VA spread adjustment (convert bps to decimal by dividing by 10000)
        df.loc[:LLP-1, 'Zero_AC'] += VA_value / 10000.0
        df.loc[:LLP-1, 'Forward_AC'] += VA_value / 10000.0
        # Recalculate continuous compounding rates based on the VA-adjusted annual rates.
        df.loc[:LLP-1, 'Zero_CC'] = np.log(1 + df.loc[:LLP-1, 'Zero_AC'])
        df.loc[:LLP-1, 'Forward_CC'] = np.log(1 + df.loc[:LLP-1, 'Forward_AC'])
        # Recalculate discount factors.
        df.loc[:LLP-1, 'Discount'] = np.exp(-df.loc[:LLP-1,
                                            'Zero_CC'] * df.loc[:LLP-1, 'Tenors'])
        # Reinstate the DLT column from the original data.
        df['DLT'] = self.curve_data['DLT'].values
        return df
