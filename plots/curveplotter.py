"""
Module for plotting discount curves and exporting curve data.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from itertools import combinations


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
        self.curve_diff_dict = {}

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
            lambda x, y: x[['Tenors']].merge(
                y[['Tenors']], on='Tenors', how='inner'),
            dfs
        )
        aligned_dfs = [
            df.merge(common_tenors, on='Tenors', how='inner') for df in dfs]
        return aligned_dfs

    def plot_curves(self, llp: int, output_path: str = None) -> None:
        """
        Plot specified pairs of curves for comparison across all scenarios.

        Args:
            scenario_curves_dict (dict): Dictionary containing all scenarios and their curves.
            output_path (str, optional): Directory to save the plots. If not provided, plots are displayed.
        """
        # Define curve pairs for comparison
        curve_pairs = [
            ('Alternative Extrapolation with VA',
             'Smith-Wilson Extrapolation with VA'),
            ('Alternative Extrapolation', 'Smith-Wilson Extrapolation')
        ]

        # Iterate over scenarios
        for scenario_name, curves in self.scenario_curves_dict.items():
            for curve1_name, curve2_name in curve_pairs:
                # Ensure both curves exist in the dictionary before proceeding
                if curve1_name in curves and curve2_name in curves:
                    curve1 = curves[curve1_name]
                    curve2 = curves[curve2_name]

                    # Align the data based on common tenors
                    curve1_aligned, curve2_aligned = self.align_dataframes(
                        curve1, curve2)

                    plt.figure(figsize=(10, 6))
                    plt.plot(curve1_aligned['Tenors'], curve1_aligned['Zero_CC'],
                             label=f"{scenario_name} - {curve1_name}", linestyle='-')
                    plt.plot(curve2_aligned['Tenors'], curve2_aligned['Zero_CC'],
                             label=f"{scenario_name} - {curve2_name}", linestyle='--')

                    plt.axvline(x=llp, color='black',
                                linestyle='dashed', linewidth=1)
                    ymin, ymax = plt.ylim()
                    plt.text(llp, (0.95*ymax), "FSP/LLP", color='black',
                             fontsize=10, ha='right', va='center')
                    plt.xlabel('Tenors', fontsize=12)
                    plt.ylabel('Zero Rates', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(True)

                    # Save or display the plot
                    if output_path:
                        os.makedirs(output_path, exist_ok=True)
                        filename = os.path.join(
                            output_path, f"{scenario_name}_{curve1_name}_vs_{curve2_name}.png")
                        plt.savefig(filename, bbox_inches='tight')
                    else:
                        plt.show()
                    plt.close()

    def plot_curves_cs_combined(self, llp: int, output_path: str = None) -> None:
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
                            curve_labels.append(' '.join(
                                word.capitalize() for word in scenario.split('_')) + f" - {curve_name}")
                            line_style = '-' if 'base_spreads' in scenario else '--'
                            # Store the line style
                            curve_styles.append(line_style)

            if not curves_to_align:
                continue

            aligned_curves = self.align_dataframes(*curves_to_align)
            plt.figure(figsize=(10, 6))
            for curve_data, label, curve_line_style in zip(aligned_curves, curve_labels, curve_styles):
                plt.plot(curve_data['Tenors'], curve_data['Zero_CC'],
                         label=label, linestyle=curve_line_style)

            plt.axvline(x=llp, color='black', linestyle='dashed', linewidth=1)
            ymin, ymax = plt.ylim()
            plt.text(llp, (0.95*ymax), "LLP", color='black',
                     fontsize=10, ha='right', va='center')
            plt.xlabel('Tenor (Years)', fontsize=12)
            plt.ylabel('Zero Rates', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(
                    output_path, f"{interest_level}_with_VA.png")
                plt.savefig(filename, bbox_inches="tight")
            else:
                plt.show()
            plt.close()

    def compute_curve_differences(self):
        """
        Computes curve differences for:
        1. Same scenario, different extrapolation methods.
        2. Same extrapolation method, different scenarios.

        Stores results in `self.curve_diff_dict`.
        """
        curve_diff_dict = {}

        # Compute differences for same scenario (Alternative vs. Smith-Wilson)
        for scenario_name, curves in self.scenario_curves_dict.items():
            if 'Alternative Extrapolation with VA' in curves and 'Smith-Wilson Extrapolation with VA' in curves:
                curve_diff_dict[f"{scenario_name}_Alternative_vs_Smith-Wilson_with_VA"] = self.compute_curve_difference(
                    curves['Alternative Extrapolation with VA'],
                    curves['Smith-Wilson Extrapolation with VA']
                )

            if 'Alternative Extrapolation' in curves and 'Smith-Wilson Extrapolation' in curves:
                curve_diff_dict[f"{scenario_name}_Alternative_vs_Smith-Wilson"] = self.compute_curve_difference(
                    curves['Alternative Extrapolation'],
                    curves['Smith-Wilson Extrapolation']
                )

        # Compute differences for the same method across different scenarios
        scenario_names = list(self.scenario_curves_dict.keys())
        for scenario_1, scenario_2 in combinations(scenario_names, 2):
            if 'Alternative Extrapolation with VA' in self.scenario_curves_dict[scenario_1] and 'Alternative Extrapolation with VA' in self.scenario_curves_dict[scenario_2]:
                curve_diff_dict[f"{scenario_1}_vs_{scenario_2}_Alternative_with_VA"] = self.compute_curve_difference(
                    self.scenario_curves_dict[scenario_1]['Alternative Extrapolation with VA'],
                    self.scenario_curves_dict[scenario_2]['Alternative Extrapolation with VA']
                )

            if 'Smith-Wilson Extrapolation with VA' in self.scenario_curves_dict[scenario_1] and 'Smith-Wilson Extrapolation with VA' in self.scenario_curves_dict[scenario_2]:
                curve_diff_dict[f"{scenario_1}_vs_{scenario_2}_Smith-Wilson_with_VA"] = self.compute_curve_difference(
                    self.scenario_curves_dict[scenario_1]['Smith-Wilson Extrapolation with VA'],
                    self.scenario_curves_dict[scenario_2]['Smith-Wilson Extrapolation with VA']
                )

            if 'Alternative Extrapolation' in self.scenario_curves_dict[scenario_1] and 'Alternative Extrapolation' in self.scenario_curves_dict[scenario_2]:
                curve_diff_dict[f"{scenario_1}_vs_{scenario_2}_Alternative"] = self.compute_curve_difference(
                    self.scenario_curves_dict[scenario_1]['Alternative Extrapolation'],
                    self.scenario_curves_dict[scenario_2]['Alternative Extrapolation']
                )

            if 'Smith-Wilson Extrapolation' in self.scenario_curves_dict[scenario_1] and 'Smith-Wilson Extrapolation' in self.scenario_curves_dict[scenario_2]:
                curve_diff_dict[f"{scenario_1}_vs_{scenario_2}_Smith-Wilson"] = self.compute_curve_difference(
                    self.scenario_curves_dict[scenario_1]['Smith-Wilson Extrapolation'],
                    self.scenario_curves_dict[scenario_2]['Smith-Wilson Extrapolation']
                )

        # Assign computed differences to the class variable
        self.curve_diff_dict = curve_diff_dict

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
                    print(f"No output path given.\n\nPrint: {df}")

    def export_curve_differences_data(self, output_path: str):
        """
        Exports computed curve differences as CSV files.

        Args:
            output_path (str): Directory where CSV files will be saved.
        """
        diff_data_path = os.path.join(output_path, "data/diffs/")
        os.makedirs(diff_data_path, exist_ok=True)

        for diff_name, diff_df in self.curve_diff_dict.items():
            diff_df.to_csv(os.path.join(
                diff_data_path, f"{diff_name}.csv"), index=False)

    def plot_curve_differences(self, llp: int, output_path: str):
        """
        Plots computed curve differences.

        Args:
            output_path (str): Directory where the plots will be saved.
        """
        diff_plot_path = os.path.join(output_path, "plots/diffs/")
        os.makedirs(diff_plot_path, exist_ok=True)

        for diff_name, curve_diffs in self.curve_diff_dict.items():
            plt.figure(figsize=(10, 6))
            plt.plot(curve_diffs['Tenors'], curve_diffs['Zero_CC'],
                     label=diff_name, linestyle='-')
            plt.xlabel('Tenors', fontsize=12)
            plt.ylabel('Zero Rate Differences', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.axvline(x=llp, color='black', linestyle='dashed', linewidth=1)
            ymin, ymax = plt.ylim()
            plt.text(llp, (0.95*ymax), "FSP/LLP", color='black',
                     fontsize=10, ha='right', va='center')

            filename = os.path.join(diff_plot_path, f"{diff_name}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    def compute_curve_difference(self, curve1, curve2):
        """
        Compute the difference between two curves (assumes same tenors).

        Args:
            curve1 (DataFrame): First curve.
            curve2 (DataFrame): Second curve.

        Returns:
            DataFrame: The difference between the two curves.
        """
        diff_df = curve1.copy()
        diff_df['Zero_CC'] = curve1['Zero_CC'] - curve2['Zero_CC']
        diff_df['Forward_CC'] = curve1['Forward_CC'] - curve2['Forward_CC']
        diff_df['Discount'] = curve1['Discount'] - curve2['Discount']
        return diff_df
