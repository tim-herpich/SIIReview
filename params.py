class CurveParameters:
    """
    Hard-coded scalar inputs required
    Adjust the actual values to match your real workbook.
    """

    def __init__(self):
        # Covergence Point of SW (= LLP + Convergence Radius for SW || = LLP for Alternative Extrapolation Method)
        self.CP_SW = 50
        self.LLP_SW = 20     # Last liquid point
        self.FSP = 20         # First Smoothing Point
        self.UFR = 0.0375      # Ultimate Forward Rate (3.6%)
        self.alpha = 0.10     # speed of convergence / reversion to the mean
        self.VA_value = 10    # VA in [bp]
        self.coupon_freq = 1.0  # number of annual coupon payments
        self.alpha_min_SW = 0.05  # minimum covergence parameter of SW
        self.CR_SW = 1.0  # covergence radius around UFR in SW in basis points
        self.CRA = 0.0        # credit risk adjustment
        self.max_tenorofAlt = 150     # maximum tenor 150
        self.compounding = 'C'  # 'C' => continuous by default
        self.instrument = 'Swap'     # Input Rates can be "Zero", "Bond", or "Swap"
        self.asset_size = 1e6   # Size of Asset Portfolio
        self.fi_asset_size = 0.85 * self.asset_size   # FI-part of Asset Portfolio
        self.asset_duration = 7.1   # Duration of Asset Portfolio
        self.liability_size = 0.95e6   # Size of Liability Portfolio
        self.liability_duration = 6.9   # Duration of Liability Portfolio
        self.pvbp_fi_assets = 0.01 * self.asset_size   # PVBP of fixed_income asset
        self.pvbp_liabs = 0.01 * self.liability_size  # PVBP of liabilities
