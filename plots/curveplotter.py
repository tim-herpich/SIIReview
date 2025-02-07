"""
Module for plotting discount curves and exporting curve data.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce


class CurvePlotter:
    """
    Class for plotting discount curves (e.g., zero rate curves) across scenarios.
    """

    def __init__(self, curves: dict):
        """
        Initialize the CurvePlotter.

        Args:
            curves (dict): Dictionary mapping scenario (or curve) names to their corresponding DataFrames.
                Each DataFrame should contain at least the columns 'Tenors' and 'Zero_CC'.
        """
        self.scenario_curves_dict = curves

    def align_dataframes(self, *dfs: pd.DataFrame) -> list:
        """
        Align multiple DataFrames by their common 'Tenors' column.

        Args:
            *dfs (pd.DataFrame): A variable number of DataFrames to align.

        Returns:
            list: A list of aligned DataFrames.
        """
        if not dfs:
            return []
        # Find the common 'Tenors' across all DataFrames
        common_tenors = reduce(
            lambda x, y: x[['Tenors']].merge(y[['Tenors']], on='Tenors', how='inner'),
            dfs
        )
        aligned_dfs = [df.merge(common_tenors, on='Tenors', how='inner') for df in dfs]
        return aligned_dfs

    def plot_curves(self, pairs: list, scenario: str, output_path: str = None) -> None:
        """
        Plot specified pairs of curves for comparison.

        Args:
            pairs (list of tuples): Each tuple contains two keys from the curves dictionary to compare.
            scenario (str): The scenario name (used in the plot title and filename).
            output_path (str, optional): Directory to save the plots. If not provided, plots are displayed.
        """
        for curve1_name, curve2_name in pairs:
            # Retrieve curves from the dictionary
            curve1 = self.scenario_curves_dict[curve1_name]
            curve2 = self.scenario_curves_dict[curve2_name]

            # Align the data based on common tenors
            curve1_aligned, curve2_aligned = self.align_dataframes(curve1, curve2)

            plt.figure(figsize=(10, 6))
            plt.plot(curve1_aligned['Tenors'], curve1_aligned['Zero_CC'],
                     label=curve1_name, linestyle='-')
            plt.plot(curve2_aligned['Tenors'], curve2_aligned['Zero_CC'],
                     label=curve2_name, linestyle='-')

            plt.xlabel('Tenors', fontsize=12)
            plt.ylabel('Zero Rates', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                # Append 'with_VA' in filename if both curves contain "VA" in their names
                suffix = "_with_VA" if ("VA" in curve1_name and "VA" in curve2_name) else ""
                filename = os.path.join(output_path, f"{scenario}{suffix}.png")
                plt.savefig(filename, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

    def plot_curves_cs_combined(self, output_path: str = None) -> None:
        """
        Plot all credit spread scenarios (i.e. curves with 'VA' in the name) for each interest rate level in one figure.

        The method first identifies unique interest rate levels based on scenario names.

        Args:
            output_path (str, optional): Directory to save the plots. If not provided, plots are displayed.
        """
        # Determine unique interest rate levels from the scenario names
        interest_levels = set()
        for scenario in self.scenario_curves_dict.keys():
            if "low_interest" in scenario:
                interest_levels.add("low_interest")
            elif "base_interest" in scenario:
                interest_levels.add("base_interest")
            elif "high_interest" in scenario:
                interest_levels.add("high_interest")

        # For each interest level, collect curves with 'VA' in the name and plot them together
        for interest_level in interest_levels:
            curves_to_align = []
            curve_labels = []
            curve_styles = []
            for scenario, curves in self.scenario_curves_dict.items():
                if interest_level in scenario:
                    for curve_name, curve_data in curves.items():
                        if 'VA' in curve_name:
                            curves_to_align.append(curve_data)
                            # curve_label = ' '.join(word.capitalize() for word in scenario.split('_'))
                            curve_labels.append(' '.join(word.capitalize() for word in scenario.split('_')) + f" - {curve_name}")
                            line_style = '-' if 'base_spreads' in scenario else '--'
                            curve_styles.append(line_style)  # Store the line style


            if not curves_to_align:
                continue

            aligned_curves = self.align_dataframes(*curves_to_align)
            plt.figure(figsize=(10, 6))
            for curve_data, label, curve_line_style in zip(aligned_curves, curve_labels, curve_styles):
                plt.plot(curve_data['Tenors'], curve_data['Zero_CC'],
                         label=label, linestyle=curve_line_style)

            plt.xlabel('Tenors', fontsize=12)
            plt.ylabel('Zero Rates', fontsize=12)
            plt.legend(fontsize=8)
            plt.grid(True)

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(output_path, f"{interest_level}_with_VA.png")
                plt.savefig(filename, bbox_inches="tight")
            else:
                plt.show()
            plt.close()

    def export_curve_data(self, output_path: str = None) -> None:
        """
        Export all curve data to CSV files.

        Each scenario is saved as a separate CSV file with filenames in the format:
        "{scenario_name}_{curve_name}.csv".

        Args:
            output_path (str, optional): Directory to save the CSV files.
        """
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        for scenario, curves in self.scenario_curves_dict.items():
            for curve_name, df in curves.items():
                filename = f"{scenario}_{curve_name}.csv".replace(" ", "_")
                if output_path:
                    file_path = os.path.join(output_path, filename)
                    df.to_csv(file_path, index=False)
                else:
                    print(f"Exporting {filename}...")
                    df.to_csv(filename, index=False)
