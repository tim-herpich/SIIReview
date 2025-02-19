"""
Obsolete Module for Smith-Wilson extrapolation of zero rate curves according to EIOPA Excel tool.
"""

import numpy as np
import pandas as pd


class ExtrapolationSWExcel:
    """
    Class implementing the Smith-Wilson extrapolation method.
    """

    def __init__(self):
        pass

    def hh(self, z):
        """
        Helper function for Hmat calculation.

        Args:
            z (float): Input value.

        Returns:
            float: Computed hh value.
        """
        return (z + np.exp(-z)) / 2

    def Hmat(self, u, v):
        """
        Compute the H matrix element.

        Args:
            u (float): First parameter.
            v (float): Second parameter.

        Returns:
            float: H matrix element.
        """
        return self.hh(u + v) - self.hh(abs(u - v))

    def galpha(self, alpha, Q, mm, umax, nrofcoup, CP, convergence_radius):
        """
        Compute the galpha function and gamma vector.

        Args:
            alpha (float): Convergence parameter.
            Q (np.ndarray): Q matrix.
            mm (int): Number of liquid rates.
            umax (float): Maximum tenor.
            nrofcoup (int): Number of coupon payments.
            CP (float): Convergence point.
            convergence_radius (float): Convergence radius.

        Returns:
            tuple: (output1, gamma) where output1 is used for alpha adjustment and gamma is the gamma vector.
        """
        Q_cols = Q.shape[1]
        h = np.zeros((int(umax * nrofcoup), int(umax * nrofcoup)))
        for i in range(1, int(umax * nrofcoup) + 1):
            for j in range(1, int(umax * nrofcoup) + 1):
                h[i - 1, j - 1] = self.Hmat(alpha * i / nrofcoup, alpha * j / nrofcoup)

        temp1 = 1 - np.sum(Q, axis=1).reshape((mm, 1))
        QH = np.matmul(Q, h)
        QHQT = np.matmul(QH, Q.T)
        try:
            QHQT_inv = np.linalg.inv(QHQT)
        except np.linalg.LinAlgError:
            raise ValueError("QHQT matrix is singular and cannot be inverted.")

        b = np.matmul(QHQT_inv, temp1)
        gamma = np.matmul(Q.T, b)
        indices = np.arange(1, Q_cols + 1)
        temp2 = np.sum(gamma.flatten() * indices / nrofcoup)
        temp3 = np.sum(gamma.flatten() * np.sinh(alpha * indices / nrofcoup))
        kappa = (1 + alpha * temp2) / temp3

        denominator = abs(1 - kappa * np.exp(CP * alpha))
        if denominator == 0:
            raise ValueError("Denominator in alpha calculation is zero.")
        output1 = alpha / denominator - convergence_radius
        output2 = gamma.flatten()

        return output1, output2

    def alpha_scan(self, lastalpha, stepsize, Q, mm, umax, nrofcoup, CP, convergence_radius):
        """
        Scan for the optimal alpha.

        Args:
            lastalpha (float): Last alpha value.
            stepsize (float): Step size.
            Q (np.ndarray): Q matrix.
            mm (int): Number of rates.
            umax (float): Maximum tenor.
            nrofcoup (int): Number of coupon payments.
            CP (float): Convergence point.
            convergence_radius (float): Convergence radius.

        Returns:
            tuple: (alpha, gamma)
        """
        step = stepsize / 10.0
        start_alpha = lastalpha - 0.9 * stepsize
        end_alpha = lastalpha
        alphas = np.arange(start_alpha, end_alpha + step, step)
        for alpha in alphas:
            output1, gamma = self.galpha(alpha, Q, mm, umax, nrofcoup, CP, convergence_radius)
            if output1 <= 0:
                return alpha, gamma
        return end_alpha, gamma

    def smith_wilson_extrapolation(self, instrument, curve_data, coupon_freq, CRA, UFR, alpha_min, CR, CP):
        """
        Perform Smith-Wilson extrapolation.

        Args:
            instrument (str): Instrument type.
            curve_data (pd.DataFrame): Input curve data.
            coupon_freq (float): Coupon frequency.
            CRA (float): Credit risk adjustment.
            UFR (float): Ultimate forward rate.
            alpha_min (float): Minimum alpha.
            CR (float): Convergence radius (in basis points).
            CP (float): Convergence point.

        Returns:
            pd.DataFrame: DataFrame containing extrapolated curves.
        """
        data = curve_data
        data_liquid = data[data['DLT'] == 1]
        nrofrates = len(data_liquid)
        u = data_liquid['Tenor'].values
        r = data_liquid['Input Rates'].values - CRA / 10000.0
        umax = np.max(u)

        if instrument == "Zero":
            coupon_freq = 1
        Q_cols = int(umax * coupon_freq)
        Q = np.zeros((nrofrates, Q_cols))
        lnUFR = np.log(1 + UFR)

        if instrument == "Zero":
            for i in range(nrofrates):
                maturity = int(u[i])
                if maturity < 1 or maturity > Q_cols:
                    raise ValueError(f"Maturity {maturity} out of Q matrix bounds.")
                Q[i, maturity - 1] = np.exp(-lnUFR * u[i]) * ((1 + r[i]) ** u[i])
        elif instrument in ["Swap", "Bond"]:
            for i in range(nrofrates):
                maturity = int(u[i] * coupon_freq)
                for j in range(1, maturity):
                    Q[i, j - 1] = np.exp(-lnUFR * j / coupon_freq) * r[i] / coupon_freq
                if maturity <= Q_cols:
                    Q[i, maturity - 1] = np.exp(-lnUFR * maturity / coupon_freq) * (1 + r[i] / coupon_freq)
                else:
                    raise ValueError(f"Maturity {maturity} out of Q matrix bounds.")
        else:
            raise ValueError("Instrument must be 'Zero', 'Bond', or 'Swap'.")

        CR = CR / 10000.0
        output1, gamma = self.galpha(alpha_min, Q, nrofrates, umax, coupon_freq, CP, CR)

        if output1 <= 0:
            alpha = alpha_min
        else:
            stepsize = 0.1
            found = False
            for alpha in np.arange(alpha_min + stepsize, 20 + stepsize, stepsize):
                output1, gamma = self.galpha(alpha, Q, nrofrates, umax, coupon_freq, CP, CR)
                if output1 <= 0:
                    found = True
                    break
            if not found:
                raise ValueError("Optimal alpha not found within range.")
            precision = 6
            for _ in range(precision - 1):
                alpha, gamma = self.alpha_scan(alpha, stepsize, Q, nrofrates, umax, coupon_freq, CP, CR)
                stepsize /= 10.0

        max_v = 121
        h_matrix = np.zeros((max_v + 1, Q_cols))
        g_matrix = np.zeros((max_v + 1, Q_cols))
        for i in range(0, max_v + 1):
            for j in range(1, Q_cols + 1):
                h_matrix[i, j - 1] = self.Hmat(alpha * i, alpha * j / coupon_freq)
                if (j / coupon_freq) > i:
                    g_matrix[i, j - 1] = alpha * (1 - np.exp(-alpha * j / coupon_freq) * np.cosh(alpha * i))
                else:
                    g_matrix[i, j - 1] = alpha * np.exp(-alpha * i) * np.sinh(alpha * j / coupon_freq)

        temptempdiscount = np.matmul(h_matrix, gamma).flatten()
        temptempintensity = np.matmul(g_matrix, gamma).flatten()
        tempdiscount = temptempdiscount.copy()
        tempintensity = temptempintensity.copy()
        indices = np.arange(1, Q_cols + 1)
        temp = np.sum((1 - np.exp(-alpha * indices / coupon_freq)) * gamma)
        zerocc = np.zeros(max_v + 1)
        fwintensity = np.zeros(max_v + 1)
        discountcc = np.zeros(max_v + 1)
        zeroac = np.zeros(max_v + 1)
        forwardac = np.zeros(max_v + 1)
        forwardcc = np.zeros(max_v + 1)

        zerocc[0] = lnUFR - alpha * temp
        fwintensity[0] = zerocc[0]
        discountcc[0] = 1

        if Q_cols >= 1:
            zerocc[1] = lnUFR - np.log(1 + tempdiscount[1])
            fwintensity[1] = lnUFR - tempintensity[1] / (1 + tempdiscount[1])
            discountcc[1] = np.exp(-lnUFR) * (1 + tempdiscount[1])
            zeroac[1] = (1 / discountcc[1]) ** 1 - 1 if discountcc[1] != 0 else 0
            forwardac[1] = zeroac[1]
        else:
            raise ValueError("Q_cols must be at least 1.")

        for i in range(2, max_v):
            zerocc[i] = lnUFR - np.log(1 + tempdiscount[i]) / i
            fwintensity[i] = lnUFR - tempintensity[i] / (1 + tempdiscount[i])
            discountcc[i] = np.exp(-lnUFR * i) * (1 + tempdiscount[i])
            zeroac[i] = (1 / discountcc[i]) ** (1 / i) - 1 if discountcc[i] != 0 else 0
            forwardac[i] = discountcc[i - 1] / discountcc[i] - 1 if (discountcc[i - 1] != 0 and discountcc[i] != 0) else 0

        zerocc[max_v] = 0
        fwintensity[max_v] = 0
        zeroac[max_v] = 0
        forwardac[max_v] = 0
        discountcc[max_v] = alpha

        zerocc_arr = np.zeros(max_v + 1)
        for i in range(1, max_v):
            forwardcc[i] = np.log(1 + forwardac[i])
            zerocc_arr[i] = np.log(1 + zeroac[i])

        output_dict = {
            'Tenors': np.arange(max_v + 1, dtype=int),
            'Zero_CC': zerocc,
            'Forward_CC': forwardcc,
            'Discount': discountcc,
            'Zero_AC': zeroac,
            'Forward_AC': forwardac,
        }
        return pd.DataFrame(data=output_dict)

    def getInputwithVA(self, zero_rates_extrapolated_ac, LLP, VA_value, curve_data):
        """
        Incorporate VA into the extrapolated zero curve.

        Args:
            zero_rates_extrapolated_ac (array-like): Extrapolated zero rates.
            LLP (int): Last Liquid Point.
            VA_value (float): VA spread in basis points.
            curve_data (pd.DataFrame): Original curve data (used for column names).

        Returns:
            pd.DataFrame: DataFrame with VA-adjusted zero curve.
        """
        dlt = np.ones(LLP, dtype=int)
        tenors = np.arange(1, LLP + 1, dtype=int)
        zero_rate_withVA = zero_rates_extrapolated_ac[1:LLP + 1] + VA_value / 10000
        return pd.DataFrame(data=list(zip(dlt, tenors, zero_rate_withVA)), columns=curve_data.columns)
