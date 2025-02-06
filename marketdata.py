"""
Module for handling market data from Excel workbooks.
"""

import pandas as pd


class MarketData:
    """
    A class to load and parse market data from an Excel workbook.
    """

    def __init__(self, filepath):
        """
        Initialize the MarketData object.

        Args:
            filepath (str): Path to the Excel file.
        """
        self.filepath = filepath
        self.workbook = None

    def open_workbook(self):
        """
        Open the Excel workbook specified by the filepath.
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
        Parse a worksheet into a Pandas DataFrame.

        Args:
            sheet_name (str): Name of the worksheet to parse.

        Returns:
            pd.DataFrame: DataFrame containing the worksheet data.
        """
        if self.workbook is None:
            raise ValueError("Workbook is not opened. Call open_workbook() first.")
        try:
            df = self.workbook.parse(sheet_name=sheet_name)
            return df
        except ValueError as e:
            raise ValueError(f"Error parsing sheet '{sheet_name}': {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing sheet '{sheet_name}': {e}")
