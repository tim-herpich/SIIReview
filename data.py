# data.py
import pandas as pd

class MarketData:
    """
    Loads market data from columns A:D in the 'Calculator' sheet.
    The columns are labeled as follows:
      - Column A => DLT            (e.g., 1 or 0)
      - Column B => Tenor          (e.g., 1, 2, 3, ...)
      - Column C => LFR Weights    (e.g., 0.2, 0.5, ...)
      - Column D => Input Rates    (as percentages, e.g., 2.0 for 2%)
    
    The last column E in Excel is NOT parsed (it's just a duplicate 
    percentage conversion) because we now do that conversion in Python.
    """

    def __init__(self, filepath, sheet_name="Calculator", parse_excel=True):
        """
        :param filepath: path to Excel or CSV file
        :param sheet_name: If Excel, which sheet to parse
        :param parse_excel: True => read_excel, False => read_csv
        """
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.parse_excel = parse_excel

        # Will hold the parsed columns:
        self.dlt = []
        self.tenors = []
        self.llfr_weights = []
        self.input_rates = [] 

        self.load_data()

    def load_data(self):
        """
        Drops any NaN rows.
        """
        if self.parse_excel:
            df = pd.read_excel(
                self.filepath,
                sheet_name=self.sheet_name,
                usecols=[0,1,2,3],
                header=0
            )
        else:
            df = pd.read_csv(self.filepath, usecols=[0, 1, 2, 3], header=0)

        # Drop blank/NaN rows
        df.dropna(inplace=True)

        # Ensure column names are as expected
        expected_columns = ["DLT", "Tenor", "LLFR Weights", "Input Rates"]
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Expected columns: {expected_columns}, but got {list(df.columns)}")

        # Map data to attributes
        self.dlt = df["DLT"].astype(float).tolist()
        self.tenors = df["Tenor"].astype(float).tolist()
        self.llfr_weights = df["LLFR Weights"].astype(float).tolist()
        self.input_rates = (df["Input Rates"]).astype(float).tolist()
