"""
Module containing parameters for curve bootstrapping and extrapolation.
"""

class Parameters:
    """
    Container for curve parameters and market scenarios.
    """

    def __init__(self):
        # Generic parameters
        self.compounding_in = 'A'
        self.compounding_out = 'C'
        self.instrument = 'Zero'
        self.VA_value = float  # This value will be set by each scenario
        self.coupon_freq = 1.0

        # Parameters for alternative extrapolation
        self.FSP = 20
        self.alpha = 0.11 # seems to be currently assumed value

        # Parameters for SW and alternative calculations
        self.UFR = 0.033
        self.CRA = 10.0

        # Parameters for Smith-Wilson (SW)
        self.alpha_min_SW = 0.05
        self.CR_SW = 1.0
        self.CP_SW = 60
        self.LLP_SW = 20
        self.max_tenorofAlt = 150

        # VA and impact calculation parameters
        self.asset_size = 1e6
        self.asset_duration = 6.8
        self.liability_size = 0.8e6
        self.liability_duration = 10.0

        self.fi_asset_size = 0.62 * self.asset_size
        self.pvbp_fi_assets = 0.1 * self.asset_duration
        self.pvbp_liabs = 0.1 * self.liability_duration

        # Market scenarios
        self.scenarios = [
            {'name': 'low_interest_base_spreads', 'irshift': -200, 'csshift': 0, 'vaspread': 27},
            {'name': 'low_interest_high_spreads', 'irshift': -200, 'csshift': 100, 'vaspread': 45}
            ,
            {'name': 'base_interest_base_spreads', 'irshift': 0, 'csshift': 0, 'vaspread': 27},
            {'name': 'base_interest_high_spreads', 'irshift': 200, 'csshift': 100, 'vaspread': 45},
            {'name': 'high_interest_base_spreads', 'irshift': 200, 'csshift': 0, 'vaspread': 27},
            {'name': 'high_interest_high_spreads', 'irshift': 200, 'csshift': 100, 'vaspread': 45}
        ]
