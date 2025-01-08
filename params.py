class CurveParameters:
    """
    Hard-coded scalar inputs from 'Calculator'!H4:H7 and 'Calculator'!O4:O8.
    Adjust the actual values to match your real workbook.
    """

    def __init__(self):
        self.CP = 50         # Covergence Point (= LLP + Convergence Radius for SW || = LLP for Alternative Extrapolation Method)
        self.FSP = 20         # First Smoothing Point
        self.UFR = 0.0375      # Ultimate Forward Rate (3.6%)
        self.alpha = 0.10     # speed of convergence / reversion to the mean
        self.VA_value = 10    # VA in [bp]
        self.CRA = 0.0        # credit risk adjustment
        self.max_tenor = 150     # maximum tenor 150
        self.compounding = 'C'  # 'C' => continuous by default
        self.input = 'Swap'     # Input Rates are 'Swap' by default (else 'Zero')
