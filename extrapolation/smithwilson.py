import numpy as np
import pandas as pd


class ExtrapolationSW:
    def __init__(self):
        pass

    def hh(self, z):
        """
        Help function for Hmat.
        hh = (z + exp(-z)) / 2
        """
        return (z + np.exp(-z)) / 2

    def Hmat(self, u, v):
        """
        Hmat function.
        Hmat = hh(u + v) - hh(abs(u - v))
        """
        return self.hh(u + v) - self.hh(abs(u - v))

    def galpha(self, alpha, Q, mm, umax, nrofcoup, CP, convergence_radius):
        """
        Galpha function.
        Calculates:
            output1: g(alpha) - convergence_radius
            output2: gamma
        """
        # Create h matrix
        Q_cols = Q.shape[1]
        h = np.zeros((int(umax * nrofcoup), int(umax * nrofcoup)))
        for i in range(1, int(umax * nrofcoup) + 1):
            for j in range(1, int(umax * nrofcoup) + 1):
                h[i-1, j-1] = self.Hmat(alpha * i /
                                        nrofcoup, alpha * j / nrofcoup)

        # temp1 = 1 - sum of Q rows
        temp1 = 1 - np.sum(Q, axis=1).reshape((mm, 1))  # mm x1

        # b = (Q H Q^T)^-1 (1 - Q 1)
        QH = np.matmul(Q, h)  # mm x Q_cols
        QHQT = np.matmul(QH, Q.T)  # mm x mm
        try:
            QHQT_inv = np.linalg.inv(QHQT)
        except np.linalg.LinAlgError:
            raise ValueError("QHQT matrix is singular and cannot be inverted.")

        b = np.matmul(QHQT_inv, temp1)  # mm x1

        # gamma = Q^T b
        gamma = np.matmul(Q.T, b)  # Q_cols x1

        # Compute kappa
        indices = np.arange(1, Q_cols + 1)
        temp2 = np.sum(gamma.flatten() * indices / nrofcoup)
        temp3 = np.sum(gamma.flatten() * np.sinh(alpha * indices / nrofcoup))
        kappa = (1 + alpha * temp2) / temp3

        # output1 = alpha / abs(1 - kappa * exp(CP * alpha)) - convergence_radius
        denominator = abs(1 - kappa * np.exp(CP * alpha))
        if denominator == 0:
            raise ValueError("Denominator in alpha calculation is zero.")
        output1 = alpha / denominator - convergence_radius

        # output2 = gamma
        output2 = gamma.flatten()

        return output1, output2

    def alpha_scan(self, lastalpha, stepsize, Q, mm, umax, nrofcoup, CP, convergence_radius):
        """
        alphaScan function.
        Scans for the optimal alpha by decreasing stepsize.
        """
        step = stepsize / 10.0
        start_alpha = lastalpha - 0.9 * stepsize
        end_alpha = lastalpha
        alphas = np.arange(start_alpha, end_alpha + step, step)
        for alpha in alphas:
            output1, gamma = self.galpha(
                alpha, Q, mm, umax, nrofcoup, CP, convergence_radius)
            if output1 <= 0:
                return alpha, gamma
        # If not found, return last alpha
        return end_alpha, gamma

    def smith_wilson_extrapolation(self, Instrument, curve_data, coupon_freq, CRA, UFR, alpha_min, CR, CP):
        """
        Main Smith-Wilson function
        Inputs:
            Instrument: "Zero", "Bond", "Swap"
            DataIn: DTL, Tenor, Input Rates
            nrofcoup: number of annual coupon payments
            CRA: credit risk adjustment (basis points)
            UFR: ultimate forward rate 
            alphamin: starting point for numerical solver of alpha
            convergence_radius: convergence bandwith
            CP: convergence point
        Returns:
            A dictionary containing six numpy arrays:
                zerocc, forwardcc, discountcc
        """
        # Convert DataIn to array
        data = curve_data
        # DataIn columns: 0: Indicator, 1: Maturity, 2: Rate
        # Select rows where Indicator=1
        data_liquid = data[data['DLT'] == 1]
        nrofrates = len(data_liquid)
        u = data_liquid['Tenor'].values
        r = data_liquid['Input Rates'].values - CRA / 10000.0
        umax = np.max(u)

        # Create Q matrix
        if Instrument == "Zero":
            coupon_freq = 1
        Q_cols = int(umax * coupon_freq)
        Q = np.zeros((nrofrates, Q_cols))
        lnUFR = np.log(1 + UFR)

        if Instrument == "Zero":
            for i in range(nrofrates):
                maturity = int(u[i])
                if maturity < 1 or maturity > Q_cols:
                    raise ValueError(
                        f"Maturity {maturity} out of Q matrix bounds.")
                Q[i, maturity - 1] = np.exp(-lnUFR * u[i]) * \
                    ((1 + r[i]) ** u[i])
        elif Instrument in ["Swap", "Bond"]:
            for i in range(nrofrates):
                maturity = int(u[i] * coupon_freq)
                for j in range(1, maturity):
                    Q[i, j - 1] = np.exp(-lnUFR * j /
                                         coupon_freq) * r[i] / coupon_freq
                if maturity <= Q_cols:
                    Q[i, maturity - 1] = np.exp(-lnUFR * maturity /
                                                coupon_freq) * (1 + r[i] / coupon_freq)
                else:
                    raise ValueError(
                        f"Maturity {maturity} out of Q matrix bounds.")
        else:
            raise ValueError("Instrument must be 'Zero', 'Bond', or 'Swap'.")

        # Convert convergence_radius from basis points to decimal
        CR = CR / 10000.0

        # Initial galpha calculation
        output1, gamma = self.galpha(
            alpha_min, Q, nrofrates, umax, coupon_freq, CP, CR)

        if output1 <= 0:
            alpha = alpha_min
        else:
            # Scanning for optimal alpha
            stepsize = 0.1
            found = False
            for alpha in np.arange(alpha_min + stepsize, 20 + stepsize, stepsize):
                output1, gamma = self.galpha(
                    alpha, Q, nrofrates, umax, coupon_freq, CP, CR)
                if output1 <= 0:
                    found = True
                    break
            if not found:
                raise ValueError("Optimal alpha not found within range.")
            # Refine alpha with decimal steps
            precision = 6
            for _ in range(precision - 1):
                alpha, gamma = self.alpha_scan(
                    alpha, stepsize, Q, nrofrates, umax, coupon_freq, CP, CR)
                stepsize /= 10.0

        # Now, alpha and gamma are determined
        # Compute H(v,u) and G(v,u) matrices for v=0 to121
        max_v = 121
        h_matrix = np.zeros((max_v + 1, Q_cols))
        g_matrix = np.zeros((max_v + 1, Q_cols))
        for i in range(0, max_v + 1):
            for j in range(1, Q_cols + 1):
                h_matrix[i, j -
                         1] = self.Hmat(alpha * i, alpha * j / coupon_freq)
                if (j / coupon_freq) > i:
                    g_matrix[i, j - 1] = alpha * \
                        (1 - np.exp(-alpha * j / coupon_freq) * np.cosh(alpha * i))
                else:
                    g_matrix[i, j - 1] = alpha * \
                        np.exp(-alpha * i) * np.sinh(alpha * j / coupon_freq)

        # Compute temptempdiscount and temptempintensity
        temptempdiscount = np.matmul(h_matrix, gamma).flatten()  # (122,)
        temptempintensity = np.matmul(g_matrix, gamma).flatten()  # (122,)

        # tempdiscount and tempintensity
        tempdiscount = temptempdiscount.copy()
        tempintensity = temptempintensity.copy()

        # Compute temp = sum((1 - exp(-alpha *i / nrofcoup)) * gamma[i] for i=1 to Q_cols
        indices = np.arange(1, Q_cols + 1)
        temp = np.sum((1 - np.exp(-alpha * indices / coupon_freq)) * gamma)

        # Initialize output arrays
        zerocc = np.zeros(max_v + 1)
        fwintensity = np.zeros(max_v + 1)
        discountcc = np.zeros(max_v + 1)
        zeroac = np.zeros(max_v + 1)
        forwardac = np.zeros(max_v + 1)
        forwardcc = np.zeros(max_v + 1)

        # yldintensity[0], fwintensity[0], discount[0]
        zerocc[0] = lnUFR - alpha * temp
        fwintensity[0] = zerocc[0]
        discountcc[0] = 1

        if Q_cols >= 1:
            # yldintensity[1]
            zerocc[1] = lnUFR - np.log(1 + tempdiscount[1])
            # fwintensity[1]
            fwintensity[1] = lnUFR - tempintensity[1] / (1 + tempdiscount[1])
            # discount[1]
            discountcc[1] = np.exp(-lnUFR) * (1 + tempdiscount[1])
            # zeroac[1]
            if discountcc[1] != 0:
                zeroac[1] = (1 / discountcc[1]) ** (1 / 1) - 1
            else:
                zeroac[1] = 0
            # forwardac[1]
            forwardac[1] = zeroac[1]
        else:
            raise ValueError("Q_cols must be at least 1.")

        # Loop from i=2 to120
        for i in range(2, max_v):
            zerocc[i] = lnUFR - np.log(1 + tempdiscount[i]) / i
            fwintensity[i] = lnUFR - tempintensity[i] / (1 + tempdiscount[i])
            discountcc[i] = np.exp(-lnUFR * i) * (1 + tempdiscount[i])
            if discountcc[i] != 0:
                zeroac[i] = (1 / discountcc[i]) ** (1 / i) - 1
            else:
                zeroac[i] = 0
            if discountcc[i - 1] != 0 and discountcc[i] != 0:
                forwardac[i] = discountcc[i - 1] / discountcc[i] - 1
            else:
                forwardac[i] = 0

        # After loop, set last index 121
        zerocc[max_v] = 0
        fwintensity[max_v] = 0
        zeroac[max_v] = 0
        forwardac[max_v] = 0
        discountcc[max_v] = alpha

        # Compute forwardcc and zerocc
        zerocc_arr = np.zeros(max_v + 1)
        for i in range(1, max_v):
            forwardcc[i] = np.log(1 + forwardac[i])
            zerocc_arr[i] = np.log(1 + zeroac[i])

        # Prepare output as a dictionary
        output_dict = {
            'Tenors': np.arange(max_v + 1, dtype=int),
            'Zero CC': zerocc,
            'Forward CC': forwardcc,
            'Discount CC': discountcc,
            'Zero AC': zeroac,
            'Forward AC': forwardac,
        }        
        return pd.DataFrame(data=output_dict)

    def getInputwithVA(self, zero_rates_extrapolated_ac, LLP, VA_value, curve_data):
        """
        Incorporate VA into zero curve function.
        Calculates:
            DLT array up to LLP
            Tenor array up to LLP
            zero curve with VA spread up to LLP
        """
        dlt = np.ones(LLP, dtype = int)
        tenors = np.arange(1, LLP+1, dtype = int)
        zero_rate_withVA = zero_rates_extrapolated_ac[1:LLP +
                                                      1] + VA_value / 10000

        return pd.DataFrame(data=list(zip(dlt, tenors, zero_rate_withVA)), columns=curve_data.columns)
