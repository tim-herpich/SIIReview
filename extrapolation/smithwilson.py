import numpy as np

class ExtrapolationSW:
    def __init__(self):
        pass

    def range_to_array(self, range_in):
        """
        Transforms a 2D list or numpy array to a numpy array.
        If single column, transposes to a 1D array.
        Otherwise, keeps as 2D array.
        """
        arr = np.array(range_in)
        if arr.ndim == 1:
            return arr
        elif arr.shape[1] == 1:
            return arr.flatten()
        else:
            return arr

    def hh(self, z):
        """
        Help function for Hmat.
        hh = (z + exp(-z)) / 2
        """
        return (z + np.exp(-z)) / 2

    def Hmat(self, u, v):
        """
        Hmat function as per VBA.
        Hmat = hh(u + v) - hh(abs(u - v))
        """
        return self.hh(u + v) - self.hh(abs(u - v))

    def galfa(self, alfa, Q, mm, umax, nrofcoup, T2, Tau):
        """
        Galfa function as per VBA.
        Calculates:
            output1: g(alfa) - tau
            output2: gamma
        """
        # Create h matrix
        Q_cols = Q.shape[1]
        h = np.zeros((int(umax * nrofcoup), int(umax * nrofcoup)))
        for i in range(1, int(umax * nrofcoup) + 1):
            for j in range(1, int(umax * nrofcoup) + 1):
                h[i-1, j-1] = self.Hmat(alfa * i / nrofcoup, alfa * j / nrofcoup)
        
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
        temp3 = np.sum(gamma.flatten() * np.sinh(alfa * indices / nrofcoup))
        kappa = (1 + alfa * temp2) / temp3
        
        # output1 = alfa / abs(1 - kappa * exp(T2 * alfa)) - Tau
        denominator = abs(1 - kappa * np.exp(T2 * alfa))
        if denominator == 0:
            raise ValueError("Denominator in alfa calculation is zero.")
        output1 = alfa / denominator - Tau
        
        # output2 = gamma
        output2 = gamma.flatten()
        
        return output1, output2

    def alfa_scan(self, lastalfa, stepsize, Q, mm, umax, nrofcoup, T2, Tau):
        """
        AlfaScan function as per VBA.
        Scans for the optimal alfa by decreasing stepsize.
        """
        step = stepsize / 10.0
        start_alfa = lastalfa - 0.9 * stepsize
        end_alfa = lastalfa
        alphas = np.arange(start_alfa, end_alfa + step, step)
        for alfa in alphas:
            output1, gamma = self.galfa(alfa, Q, mm, umax, nrofcoup, T2, Tau)
            if output1 <= 0:
                return alfa, gamma
        # If not found, return last alfa
        return end_alfa, gamma

    def smith_wilson_brute_force(self, Instrument, DataIn, nrofcoup, CRA, UFRac, alfamin, Tau, T2):
        """
        Main Smith-Wilson Brute Force function as per VBA.
        Inputs:
            Instrument: "Zero", "Bond", "Swap"
            DataIn: 2D list or numpy array, 50 x 3
            nrofcoup: integer
            CRA: float (basis points)
            UFRac: float
            alfamin: float
            Tau: float (basis points)
            T2: integer
        Returns:
            A dictionary containing six numpy arrays:
                discount, yldintensity, zeroac, fwintensity, forwardcc, forwardac
        """
        # Convert DataIn to array
        data = self.range_to_array(DataIn)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # DataIn columns: 0: Indicator, 1: Maturity, 2: Rate
        # Select rows where Indicator=1
        liquid_indices = np.where(data[:, 0] == 1)[0]
        nrofrates = len(liquid_indices)
        u = data[liquid_indices, 1]
        r = data[liquid_indices, 2] - CRA / 10000.0
        umax = np.max(u)
        
        # Create Q matrix
        if Instrument == "Zero":
            nrofcoup = 1
        Q_cols = int(umax * nrofcoup)
        Q = np.zeros((nrofrates, Q_cols))
        lnUFR = np.log(1 + UFRac)
        
        if Instrument == "Zero":
            for i in range(nrofrates):
                maturity = int(u[i])
                if maturity < 1 or maturity > Q_cols:
                    raise ValueError(f"Maturity {maturity} out of Q matrix bounds.")
                Q[i, maturity - 1] = np.exp(-lnUFR * u[i]) * ((1 + r[i]) ** u[i])
        elif Instrument in ["Swap", "Bond"]:
            for i in range(nrofrates):
                maturity = int(u[i] * nrofcoup)
                for j in range(1, maturity):
                    Q[i, j - 1] = np.exp(-lnUFR * j / nrofcoup) * r[i] / nrofcoup
                if maturity <= Q_cols:
                    Q[i, maturity - 1] = np.exp(-lnUFR * maturity / nrofcoup) * (1 + r[i] / nrofcoup)
                else:
                    raise ValueError(f"Maturity {maturity} out of Q matrix bounds.")
        else:
            raise ValueError("Instrument must be 'Zero', 'Bond', or 'Swap'.")
        
        # Convert Tau from basis points to decimal
        Tau = Tau / 10000.0
        
        # Initial galfa calculation
        output1, gamma = self.galfa(alfamin, Q, nrofrates, umax, nrofcoup, T2, Tau)
        
        if output1 <= 0:
            alfa = alfamin
        else:
            # Scanning for optimal alfa
            stepsize = 0.1
            found = False
            for alfa in np.arange(alfamin + stepsize, 20 + stepsize, stepsize):
                output1, gamma = self.galfa(alfa, Q, nrofrates, umax, nrofcoup, T2, Tau)
                if output1 <= 0:
                    found = True
                    break
            if not found:
                raise ValueError("Optimal alfa not found within range.")
            # Refine alfa with decimal steps
            precision = 6
            for _ in range(precision - 1):
                alfa, gamma = self.alfa_scan(alfa, stepsize, Q, nrofrates, umax, nrofcoup, T2, Tau)
                stepsize /= 10.0
        
        # Now, alfa and gamma are determined
        # Compute H(v,u) and G(v,u) matrices for v=0 to121
        max_v = 121
        h_matrix = np.zeros((max_v + 1, Q_cols))
        g_matrix = np.zeros((max_v + 1, Q_cols))
        for i in range(0, max_v + 1):
            for j in range(1, Q_cols + 1):
                h_matrix[i, j - 1] = self.Hmat(alfa * i, alfa * j / nrofcoup)
                if (j / nrofcoup) > i:
                    g_matrix[i, j - 1] = alfa * (1 - np.exp(-alfa * j / nrofcoup) * np.cosh(alfa * i))
                else:
                    g_matrix[i, j - 1] = alfa * np.exp(-alfa * i) * np.sinh(alfa * j / nrofcoup)
        
        # Compute temptempdiscount and temptempintensity
        temptempdiscount = np.matmul(h_matrix, gamma).flatten()  # (122,)
        temptempintensity = np.matmul(g_matrix, gamma).flatten()  # (122,)
        
        # tempdiscount and tempintensity
        tempdiscount = temptempdiscount.copy()
        tempintensity = temptempintensity.copy()
        
        # Compute temp = sum((1 - exp(-alfa *i / nrofcoup)) * gamma[i] for i=1 to Q_cols
        indices = np.arange(1, Q_cols + 1)
        temp = np.sum((1 - np.exp(-alfa * indices / nrofcoup)) * gamma)
        
        # Initialize output arrays
        yldintensity = np.zeros(max_v + 1)
        fwintensity = np.zeros(max_v + 1)
        discount = np.zeros(max_v + 1)
        zeroac = np.zeros(max_v + 1)
        forwardac = np.zeros(max_v + 1)
        forwardcc = np.zeros(max_v + 1)
        
        # yldintensity[0], fwintensity[0], discount[0]
        yldintensity[0] = lnUFR - alfa * temp
        fwintensity[0] = yldintensity[0]
        discount[0] = 1
        
        if Q_cols >= 1:
            # yldintensity[1]
            yldintensity[1] = lnUFR - np.log(1 + tempdiscount[1])
            # fwintensity[1]
            fwintensity[1] = lnUFR - tempintensity[1] / (1 + tempdiscount[1])
            # discount[1]
            discount[1] = np.exp(-lnUFR) * (1 + tempdiscount[1])
            # zeroac[1]
            if discount[1] != 0:
                zeroac[1] = (1 / discount[1]) ** (1 / 1) - 1
            else:
                zeroac[1] = 0
            # forwardac[1]
            forwardac[1] = zeroac[1]
        else:
            raise ValueError("Q_cols must be at least 1.")
        
        # Loop from i=2 to120
        for i in range(2, max_v):
            yldintensity[i] = lnUFR - np.log(1 + tempdiscount[i]) / i
            fwintensity[i] = lnUFR - tempintensity[i] / (1 + tempdiscount[i])
            discount[i] = np.exp(-lnUFR * i) * (1 + tempdiscount[i])
            if discount[i] != 0:
                zeroac[i] = (1 / discount[i]) ** (1 / i) - 1
            else:
                zeroac[i] = 0
            if discount[i - 1] != 0 and discount[i] != 0:
                forwardac[i] = discount[i - 1] / discount[i] - 1
            else:
                forwardac[i] = 0
        
        # After loop, set last index 121
        yldintensity[max_v] = 0
        fwintensity[max_v] = 0
        zeroac[max_v] = 0
        forwardac[max_v] = 0
        discount[max_v] = alfa
        
        # Compute forwardcc and zerocc
        zerocc_arr = np.zeros(max_v + 1)
        for i in range(1, max_v):
            forwardcc[i] = np.log(1 + forwardac[i])
            zerocc_arr[i] = np.log(1 + zeroac[i])
        
        # Prepare output as a dictionary
        output = {
            'discount': discount,
            'yldintensity': yldintensity,
            'zeroac': zeroac,
            'fwintensity': fwintensity,
            'forwardcc': forwardcc,
            'forwardac': forwardac
        }
        
        return output
