import numpy as np
import pandas as pd


class VASpreadCalculator:
    """
    A class to compute the new Valuation Adjustment (VA) spread for interest rate curves.
    """

    def __init__(self):

        # EIOPA reference portfolio as of 03/24
        self.wgov_cu = pd.DataFrame(data=np.array([0.03, 0.08, 0., 0., 0., 0., 0., 0., 0.01, 0.36, 0.15, 0., 0., 0., 0.01, 0.22, 0., 0., 0., 0., 0., 0.02,
                                                   0., 0.01, 0.01, 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                                    columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                             "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"])

        self.wother_cu = pd.DataFrame(data=np.array([0.15, 0.12, 0.27, 0.14, 0.01, 0., 0., 0.04, 0.04, 0.09, 0.13, 0.01, 0., 0.]),
                                      columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                               "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"])

        self.durgov_cu = pd.DataFrame(data=np.array([8.5, 8.5, 0., 0., 0., 0., 0., 0., 8.5, 8.5, 8.5, 0., 0., 0., 8.5, 8.5, 0., 0., 0., 0., 0., 8.5,
                                                     0., 5.6, 8.5, 0., 0., 0., 8.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                                      columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                               "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"])

        self.durother_cu = pd.DataFrame(data=np.array([7.3, 5.6, 5.6, 4.5, 3.3, 0., 0., 8.6, 8., 5.7, 4.8, 3.3, 0., 0.]),
                                        columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                                 "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"])

        self.wgov_co = pd.DataFrame(data=np.array([0.04, 0.08, 0., 0., 0., 0., 0., 0., 0.01, 0.12, 0.49, 0., 0., 0., 0.02, 0.02, 0., 0., 0., 0., 0., 0.02,
                                                   0., 0.01, 0., 0., 0.01, 0.01, 0.06, 0., 0., 0.01, 0.01, 0.01, 0.01, 0., 0.01, 0.06]),
                                    columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                             "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"])

        self.wother_co = pd.DataFrame(data=np.array([0.15, 0.12, 0.27, 0.16, 0.02, 0., 0., 0.06, 0.05, 0.08, 0.08, 0.01, 0., 0.]),
                                      columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                               "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"])

        self.durgov_co = pd.DataFrame(data=np.array([12.6, 16.1, 0., 0., 0., 0., 0., 0., 10.4, 14.8, 10.8, 0., 0., 0., 18.3, 8.1, 0., 0., 0., 0.,
                                                     0., 10.4, 0., 7., 0., 0., 10.9, 13., 13.2, 0., 0., 5.8, 4.9, 3.9, 4.5, 0., 0.8, 5.6]),
                                      columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                               "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"])

        self.durother_co = pd.DataFrame(data=np.array([8.3, 7.4, 7.2, 5.7, 0., 0., 0., 11.7, 11.1, 7.9, 6.4, 0., 0., 0.]),
                                        columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                                 "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"])

        self.ltsgov = pd.DataFrame(data=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                                  [-0.03, 0.06, 0.11, 0.16, 0.21, 0.26, 0.30, 0.34, 0.36, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37,
                                                   0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]]))

        self.ltsother = pd.DataFrame(data=np.array([
            ['Issuer', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            ['EUR_Financials_0', 0.17, 0.17, 0.21, 0.25, 0.29, 0.32, 0.35, 0.35, 0.36, 0.38, 0.39, 0.40, 0.41,
             0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
            ['EUR_Financials_1', 0.54, 0.54, 0.61, 0.68, 0.75, 0.81, 0.87, 0.89, 0.91, 0.94, 0.97, 0.99, 1.01,
             1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02],
            ['EUR_Financials_2', 1.18, 1.18, 1.23, 1.31, 1.45, 1.55, 1.60, 1.60, 1.60, 1.61, 1.62, 1.62, 1.62,
             1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62, 1.62],
            ['EUR_Financials_3', 3.08, 3.08, 2.94, 3.01, 3.10, 3.23, 3.30, 3.28, 3.26, 3.25, 3.26, 3.26, 3.26,
             3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26, 3.26],
            ['EUR_Financials_4', 6.37, 6.37, 6.27, 6.25, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24,
             6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24, 6.24],
            ['EUR_Financials_5', 14.31, 14.31, 14.21, 14.19, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18,
             14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18],
            ['EUR_Financials_6', 14.31, 14.31, 14.21, 14.19, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18,
             14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18, 14.18],
            ['EUR_Non-Financials_0', 0.04, 0.04, 0.03, 0.04, 0.07, 0.11, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.20,
             0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
            ['EUR_Non-Financials_1', 0.38, 0.38, 0.40, 0.43, 0.49, 0.56, 0.63, 0.65, 0.68, 0.70, 0.72, 0.72, 0.73,
             0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73],
            ['EUR_Non-Financials_2', 0.57, 0.57, 0.64, 0.72, 0.81, 0.87, 0.93, 0.96, 1.01, 1.05, 1.07, 1.08, 1.08,
             1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08],
            ['EUR_Non-Financials_3', 1.16, 1.16, 1.30, 1.41, 1.47, 1.56, 1.62, 1.68, 1.76, 1.81, 1.84, 1.84, 1.84,
             1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84],
            ['EUR_Non-Financials_4', 4.44, 4.44, 4.31, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25,
             4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25],
            ['EUR_Non-Financials_5', 7.12, 7.12, 6.99, 6.90, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88,
             6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88],
            ['EUR_Non-Financials_6', 7.12, 7.12, 6.99, 6.90, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88,
             6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88, 6.88]
        ], dtype=object))

    def round_duration_tenors(self):
        """
        Rounds the duration floats in all df to integer years.
        Args:
        Returns: 
        """
        self.durgov_cu = self.durgov_cu.round(0).astype(int)
        self.durother_cu = self.durgov_cu.round(0).astype(int)
        self.durgov_co = self.durgov_co.round(0).astype(int)
        self.durother_co = self.durother_co.round(0).astype(int)        

    def compute_average_gov_cu_spread(self, spreads):
        """
        Computes the average gov spread S_gov for a currency. Floored with zero.
        Args:
        Returns:
            S_gov_cu
        """
        S_gov_cu = max((self.wgov_cu * spreads).sum(axis=1),0)
        return S_gov_cu

    def compute_average_gov_co_spread(self, spreads):
        """
        Computes the average gov spread S_gov for a country. Floored with zero.
        Args:
        Returns:
            S_gov_co
        """
        S_gov_co = max((self.wgov_co * spreads).sum(axis=1),0)
        return S_gov_co

    def compute_average_other_cu_spread(self, spreads):
        """
        Computes the average other spread S_other for a currency. Floored with zero.
        Args:
        Returns:
            S_other_cu
        """
        S_other_cu = max((self.wother_cu * spreads).sum(axis=1),0)
        return S_other_cu

    def compute_average_other_co_spread(self, spreads):
        """
        Computes the average other spread S_other for a country. Floored with zero.
        Args:
        Returns:
            S_other_co
        """
        S_other_co = max((self.wother_co * spreads).sum(axis=1),0)
        return S_other_co

    def compute_rc_gov(self, spread_avg, ltas_avg):
        """
        Computes the risk-correction for gov (RC_gov)
        Args:
        Returns:
            RC_gov
        """
        RC_gov = min( 0.3*min(spread_avg, ltas_avg) + 0.2*max(0,min(spread_avg-ltas_avg, ltas_avg)) + 0.15*max(0,spread_avg-2*ltas_avg), 1.05*ltas_avg)
        return RC_gov

    def compute_rc_other(self, spread_avg, ltas_avg):
        """
        Computes the risk-correction for other (RC_other)
        Args:
        Returns:
            RC_other
        """
        RC_other = min( 0.5*min(spread_avg, ltas_avg) + 0.4*max(0,min(spread_avg-ltas_avg, ltas_avg)) + 0.35*max(0,spread_avg-2*ltas_avg), 1.95*ltas_avg)
        return RC_other



    def compute_cssr_cu(self):
        """
        Computes the currency-specific credit spread sensitive ratio (CSSR)
        Args:
        Returns:
            Currency-specific CSSR
        """

        # compute max(min( [(FI_A-FI_A*)/VA*]/(BE_L-BE_L*)/VA*,1),0)

        return 1.0


    def compute_rcs_co(self):
        """
        Computes the country-specific risk-corrected spread (RCS)
        Args:
        Returns:
            Country-specific RCS
        """

        # compute S_co - RC_co

        return 1.0

    def compute_rc_gov(self):
        """
        Computes the risk-corrected spread for GOV (RC_gov)
        Args:
        Returns:
            RC_gov
        """

        # compute S_co - RC_co

        return 1.0

    def compute_w_co(self):
        """
        Computes the country-specific adjustment factor w_co
        Args:
        Returns:
            Country-specific w_co
        """

        # w_co = max( min([self.compute_rcs_co * FI / Assets - 0.6%] / 0.3%, 1),0)

        return 1.0

    def compute_total_va(self):
        """
        Computes the VA spread based on the provided zero rates and other parameters.
        Args:
        Returns:
            VA spread in bp
        """
        return self.compute_macro_va_() + self.compute_currency_va_()

    def compute_macro_va_(self):
        """
        Computes the macroeconomic VA spread based on the provided zero rates and other parameters.
        Args:
        Returns:
            Macro VA spread in bp
        """

        # va_macro = 0.85*self.compute_cssr_cu()*max(self.compute_rcs_co-1.3*self.compute_rcs_cu, 0)*w_co

        return 5.0

    def compute_currency_va_(self):
        """
        Computes the currency VA spread based on the currency-specific credit spread sensitive ratio and the risk-corrected spread
        Args:
        Returns:
            Currency-VA spread in bp
        """
        # va_cu = 0.85*self.compute_cssr_cu()*self.compute_rcs_cu()

        return 5.0
