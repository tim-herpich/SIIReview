"""
Module for computing the Valuation Adjustment (VA) spread.
"""

import numpy as np
import pandas as pd


class VaSpreadCalculator:
    """
    Class to compute the new VA spread for interest rate curves.
    """

    def __init__(self, va_spreads_df, fi_asset_size, liability_size, pvbp_fi_assets, pvbp_liabs):
        """
        Initialize the VASpreadCalculator.

        Args:
            va_spreads_df (pd.DataFrame): DataFrame containing VA spread data.
            fi_asset_size (float): Fixed income asset size.
            liability_size (float): Liability size.
            pvbp_fi_assets (float): PVBP of fixed income assets.
            pvbp_liabs (float): PVBP of liabilities.
        """

        # EIOPA reference portfolio as of 03/24
        self.w_cu = pd.DataFrame(data=np.array([0.241, 0.377]).reshape(
            1, -1), columns=["GOV", "OTHER",], index=['EUR'])

        self.w_co = pd.DataFrame(data=np.array([0.15, 0.40]).reshape(
            1, -1), columns=["GOV", "OTHER",], index=['DE'])

        self.wgov_cu = pd.DataFrame(data=np.array([0.03, 0.08, 0., 0., 0., 0., 0., 0., 0.01, 0.36, 0.15, 0., 0., 0., 0.01, 0.22, 0., 0., 0., 0., 0., 0.02,
                                                   0., 0.01, 0.01, 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(1, -1),
                                    columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                             "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"], index=['EUR'])

        self.wother_cu = pd.DataFrame(data=np.array([0.15, 0.12, 0.27, 0.14, 0.01, 0., 0., 0.04, 0.04, 0.09, 0.13, 0.01, 0., 0.]).reshape(1, -1),
                                      columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                               "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"], index=['EUR'])

        self.durgov_cu = pd.DataFrame(data=np.array([8.5, 8.5, 0., 0., 0., 0., 0., 0., 8.5, 8.5, 8.5, 0., 0., 0., 8.5, 8.5, 0., 0., 0., 0., 0., 8.5,
                                                     0., 5.6, 8.5, 0., 0., 0., 8.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(1, -1),
                                      columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                               "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"], index=['EUR'])

        self.durother_cu = pd.DataFrame(data=np.array([7.3, 5.6, 5.6, 4.5, 3.3, 0., 0., 8.6, 8., 5.7, 4.8, 3.3, 0., 0.]).reshape(1, -1),
                                        columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                                 "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"], index=['EUR'])

        self.wgov_co = pd.DataFrame(data=np.array([0.04, 0.08, 0., 0., 0., 0., 0., 0., 0.01, 0.12, 0.49, 0., 0., 0., 0.02, 0.02, 0., 0., 0., 0., 0., 0.02,
                                                   0., 0.01, 0., 0., 0.01, 0.01, 0.06, 0., 0., 0.01, 0.01, 0.01, 0.01, 0., 0.01, 0.06]).reshape(1, -1),
                                    columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                             "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"])

        self.wother_co = pd.DataFrame(data=np.array([0.15, 0.12, 0.27, 0.16, 0.02, 0., 0., 0.06, 0.05, 0.08, 0.08, 0.01, 0., 0.]).reshape(1, -1),
                                      columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                               "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"])

        self.durgov_co = pd.DataFrame(data=np.array([12.6, 16.1, 0., 0., 0., 0., 0., 0., 10.4, 14.8, 10.8, 0., 0., 0., 18.3, 8.1, 0., 0., 0., 0.,
                                                     0., 10.4, 0., 7., 0., 0., 10.9, 13., 13.2, 0., 0., 5.8, 4.9, 3.9, 4.5, 0., 0.8, 5.6]).reshape(1, -1),
                                      columns=["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI",
                                               "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "UK", "AU", "CA", "CN", "HK", "JP", "US"])

        self.durother_co = pd.DataFrame(data=np.array([8.3, 7.4, 7.2, 5.7, 0., 0., 0., 11.7, 11.1, 7.9, 6.4, 0., 0., 0.]).reshape(1, -1),
                                        columns=["Finan_0", "Finan_1", "Finan_2", "Finan_3", "Finan_4", "Finan_5", "Finan_6",
                                                 "Nonfinan_0", "Nonfinan_1", "Nonfinan_2", "Nonfinan_3", "Nonfinan_4", "Nonfinan_5", "Nonfinan_6"])

        self.ltsgov = pd.DataFrame(data=np.array([-0.03, 0.06, 0.11, 0.16, 0.21, 0.26, 0.30, 0.34, 0.36, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37,
                                                  0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]).reshape(1, -1),
                                   columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], index=['EUR'])

        self.ltsother = pd.DataFrame(data=np.array([
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
        ], dtype=object), columns=['Issuer', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        self.ltsother.set_index('Issuer', inplace=True)

        self.relevant_ltas_gov_df = self._extract_relevant_ltas_gov()
        self.relevant_ltas_other_df = self._extract_relevant_ltas_other()
        self.lts_gov_avg = self._compute_long_term_average_gov_spread()
        self.lts_other_avg = self._compute_long_term_average_other_spread()

        self.va_spreads_df = va_spreads_df
        self.fi_asset_size = fi_asset_size
        self.liability_size = liability_size
        self.pvbp_fi_assets = pvbp_fi_assets
        self.pvbp_liabs = pvbp_liabs

        self.Sgov_cu = self._compute_average_sub_spread(
            self.wgov_cu, self.durgov_cu)
        self.Sgov_co = self._compute_average_sub_spread(
            self.wgov_co, self.durgov_co)
        self.Sother_cu = self._compute_average_sub_spread(
            self.wother_cu, self.durother_cu)
        self.Sother_co = self._compute_average_sub_spread(
            self.wother_co, self.durother_co)
        self.S_cu = self._compute_average_spread(
            self.Sgov_cu, self.Sother_cu, self.w_cu)
        self.S_co = self._compute_average_spread(
            self.Sgov_co, self.Sother_co, self.w_co)

        self.Rc_gov_cu = self._compute_rc_gov(self.Sgov_cu)
        self.Rc_other_cu = self._compute_rc_gov(self.Sgov_cu)
        self.Rc_gov_co = self._compute_rc_gov(self.Sgov_co)
        self.Rc_other_co = self._compute_rc_gov(self.Sgov_co)
        self.Rc_cu = self._compute_average_spread(
            self.Rc_gov_cu, self.Rc_other_cu, self.w_cu)
        self.Rc_co = self._compute_average_spread(
            self.Rc_gov_co, self.Rc_other_co, self.w_co)

        self.Rcs_cu = self._compute_rcs(self.S_cu, self.Rc_cu)
        self.Rcs_co = self._compute_rcs(self.S_co, self.Rc_co)

        self.Cssr_cu = self._compute_cssr_cu(pvbp_fi_assets, pvbp_liabs)
        self.w_co = self._compute_w_co(
            self.Rcs_co, fi_asset_size, liability_size)

    def _extract_relevant_ltas_gov(self):
        """
        Extract relevant LTAS for government bonds.

        Returns:
            pd.DataFrame: DataFrame with relevant LTAS for government.
        """
        relevant_ltas_gov = []
        for duration in self.durgov_cu.values.round(0).astype(int)[0]:
            if duration > 0 and duration in self.ltsgov.columns:
                spread = self.ltsgov.iloc[0, duration]
            else:
                spread = 0
            relevant_ltas_gov.append(spread)
        return pd.DataFrame(data=[relevant_ltas_gov], columns=self.durgov_cu.columns, index=self.durgov_cu.index)

    def _extract_relevant_ltas_other(self):
        """
        Extract relevant LTAS for other bonds.

        Returns:
            pd.DataFrame: DataFrame with relevant LTAS for others.
        """
        relevant_ltas_other = []
        rows = 0
        for duration in self.durother_cu.values.round(0).astype(int)[0]:
            if duration > 0 and duration in self.ltsother.columns:
                spread = self.ltsother.iloc[rows, duration]
            else:
                spread = 0
            relevant_ltas_other.append(spread)
            rows += 1
        return pd.DataFrame(data=[relevant_ltas_other], columns=self.durother_cu.columns, index=self.durother_cu.index)

    def _compute_long_term_average_gov_spread(self):
        """
        Compute the long-term average government spread.

        Returns:
            float: Long-term average government spread.
        """
        ltas_gov_avg = max(
            (self.wgov_cu * self.relevant_ltas_gov_df).sum(axis=1)[0], 0)
        return ltas_gov_avg

    def _compute_long_term_average_other_spread(self):
        """
        Compute the long-term average other spread.

        Returns:
            float: Long-term average other spread.
        """
        ltas_other_avg = max(
            (self.wother_cu * self.relevant_ltas_other_df).sum(axis=1)[0], 0)
        return ltas_other_avg

    def _compute_average_sub_spread(self, weights_df, dur_df):
        """
        Compute the average sub-spread.

        Returns:
            float: Average sub-spread.
        """
        spreads_rows_df = self.va_spreads_df.loc[weights_df.columns.values.tolist(
        )]
        relevant_spreads = []
        rows = 0
        for duration in dur_df.values.round(0).astype(int)[0]:
            if duration > 0 and duration in spreads_rows_df.columns:
                spread = spreads_rows_df.iloc[rows, duration]
            else:
                spread = 0
            relevant_spreads.append(spread)
            rows += 1
        S_sub = max(np.dot(weights_df.values,
                    np.array(relevant_spreads))[0], 0)
        return S_sub

    def _compute_average_spread(self, spread_gov, spread_other, w_df):
        """
        Compute the average spread.

        Returns:
            float: Average spread.
        """
        spread_avg = max(
            spread_gov * w_df.loc[:, 'GOV'][0] + spread_other * w_df.loc[:, 'OTHER'][0], 0)
        return spread_avg

    def _compute_rc_gov(self, spread_avg_gov):
        """
        Compute risk correction for government bonds.

        Returns:
            float: Risk correction for government bonds.
        """
        RC_gov = min(
            0.3 * min(spread_avg_gov, self.lts_gov_avg) +
            0.2 * max(0, min(spread_avg_gov - self.lts_gov_avg, self.lts_gov_avg)) +
            0.15 * max(0, spread_avg_gov - 2 * self.lts_gov_avg),
            1.05 * self.lts_gov_avg
        )
        return RC_gov

    def _compute_rc_other(self, spread_avg_other):
        """
        Compute risk correction for other bonds.

        Returns:
            float: Risk correction for other bonds.
        """
        RC_other = min(
            0.5 * min(spread_avg_other, self.lts_other_avg) +
            0.4 * max(0, min(spread_avg_other - self.lts_other_avg, self.lts_other_avg)) +
            0.35 * max(0, spread_avg_other - 2 * self.lts_other_avg),
            1.95 * self.lts_other_avg
        )
        return RC_other

    def _compute_rcs(self, spread, rc):
        """
        Compute risk-corrected spread (RCS).

        Returns:
            float: Risk-corrected spread.
        """
        return spread - rc

    def _compute_cssr_cu(self, pvbp_fi_assets, pvbp_liabs):
        """
        Compute the currency-specific credit spread sensitive ratio (CSSR).

        Returns:
            float: CSSR for currency.
        """
        Cssr_cu = max(min(pvbp_fi_assets / pvbp_liabs, 1), 0)
        return Cssr_cu

    def _compute_w_co(self, Rcs_co, fi_asset_size, liability_size):
        """
        Compute the country-specific adjustment factor.

        Returns:
            float: Country-specific adjustment factor.
        """
        w_co = max(
            min((Rcs_co * fi_asset_size / liability_size - 0.006) / 0.003, 1), 0)
        return w_co

    def compute_macro_va_(self):
        """
        Compute the macroeconomic VA spread.

        Returns:
            float: Macroeconomic VA spread.
        """
        va_macro = 0.85 * self.Cssr_cu * \
            max(self.Rcs_co - 1.3 * self.Rcs_cu, 0) * self.w_co
        return va_macro

    def compute_currency_va_(self):
        """
        Compute the currency-specific VA spread.

        Returns:
            float: Currency-specific VA spread.
        """
        va_cu = 0.85 * self.Cssr_cu * self.Rcs_cu
        return va_cu

    def compute_total_va(self):
        """
        Compute the total VA spread (in basis points).

        Returns:
            float: Total VA spread in basis points.
        """
        va = (self.compute_macro_va_() + self.compute_currency_va_()) * 1e4
        return va
