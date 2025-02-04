import pandas as pd

class MarketData:

    def __init__(self, filepath):
        """
        Initialize the MarketData class.
        :param filepath: Path to the Excel file.
        """
        self.filepath = filepath
        self.workbook = None

    def open_workbook(self):
        """
        Open the Excel workbook.
        """
        try:
            self.workbook = pd.ExcelFile(self.filepath)
        except Exception as e:
            raise ValueError(f"Error opening workbook: {e}")

    def close_workbook(self):
        """
        Close the Excel workbook.
        """
        self.workbook = None

    def parse_sheet_to_df(self, sheet_name):
        """
        Parse data from a specific worksheet into a DataFrame.
        :param sheet_name: Name of the worksheet to parse.
        :return: Pandas DataFrame containing the data from the worksheet.
        """
        if not self.workbook:
            raise ValueError("Workbook is not opened. Call open_workbook() first.")

        try:
            df = self.workbook.parse(sheet_name=sheet_name)
            return df
        except ValueError as e:
            raise ValueError(f"Error parsing sheet '{sheet_name}': {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing sheet '{sheet_name}': {e}")
