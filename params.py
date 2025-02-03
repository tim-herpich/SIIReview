class CurveParameters:
    """
    Hard-coded scalar inputs required
    Adjust the actual values to match your real workbook.
    """

    def __init__(self, vaspread):

        # Generic
        self.compounding_in = 'A'  # compunding of input rates, 'A' => annual by default
        self.compounding_out = 'C'  # compunding of bootstrapped rates, 'C' => continuous by default
        self.instrument = 'Zero'     # Input Rates can be "Zero", "Bond", or "Swap"
        self.VA_value = vaspread   # VA in [bp]
        self.coupon_freq = 1.0  # number of annual coupon payments

        # Required for Alternative (Dutch) calculation
        self.FSP = 20         # First Smoothing Point
        self.alpha = 0.10     # speed of convergence / reversion to the mean

        # Required for SW and Alternative (Dutch) calculation
        self.UFR = 0.033      # Ultimate Forward Rate (3.3%)
        self.CRA = 10.0        # credit risk adjustment in bp

        # Required for SW calculation
        self.alpha_min_SW = 0.05  # minimum covergence parameter of SW
        self.CR_SW = 1.0  # covergence radius around UFR in SW in basis points
        self.CP_SW = 60  # Covergence Point of SW (= LLP + Convergence Radius for SW)
        self.LLP_SW = 20     # Last liquid point
        self.max_tenorofAlt = 150     # maximum tenor 150

        # Required for VA and impact calculation
        self.asset_size = 1e6  # Size of Asset Portfolio
        self.asset_duration = 6.8  # Duration of Asset Portfolio | Derivation from EIOPA reference portfolio 03/24
        self.liability_size = 0.8e6  # Size of Liability Portfolio (0.8e6)
        self.liability_duration = 10.0 # Duration of Liability Portfolio | BE (10.0)

        # Required for VA
        self.fi_asset_size = 0.62 * self.asset_size   # FI-part of Asset Portfolio | Derivation from EIOPA reference portfolio 03/24
        self.pvbp_fi_assets = 0.1 * self.asset_duration   # PVBP of fixed_income asset
        self.pvbp_liabs = 0.1 * self.liability_duration  # PVBP of liabilities
