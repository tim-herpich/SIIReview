import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

class CurvePlotter:
    """
    A class to plot specified combinations of discount curves vs. tenors for given dataframes.
    """

    def __init__(self, curves):
        """
        Initialize the CurvePlotter with the dataframes.

        Args:
            curves (dict): A dictionary containing scenario names and their corresponding dataframes.
        """
        self.scenario_curves_dict = curves

    def align_dataframes(self, *dfs):
        """
        Align multiple dataframes by their common tenors.

        Args:
            *dfs (pd.DataFrame): A variable number of dataframes to align.

        Returns:
            list: A list of aligned dataframes.
        """
        if not dfs:
            return []

        # Find the common tenors across all dataframes
        common_tenors = reduce(lambda x, y: x.merge(y, on='Tenors', how='inner'), dfs)[['Tenors']]

        # Align all dataframes to the common tenors
        aligned_dfs = [df.merge(common_tenors, on='Tenors', how='inner').reset_index(drop=True) for df in dfs]
        return aligned_dfs

    def plot_curves(self, pairs, scenario, output_path=None):
        """
        Plot specified 2-curve comparisons of Discount vs. Tenors.

        Args:
            pairs (list of tuples): List of tuples containing curve name pairs to compare.
            output_path (str, optional): Directory to save the plots. If None, plots are displayed.
        """
        for curve1_name, curve2_name in pairs:
            curve1 = self.scenario_curves_dict[curve1_name]
            curve2 = self.scenario_curves_dict[curve2_name]

            # Align the dataframes
            curve1, curve2 = self.align_dataframes(curve1, curve2)

            plt.figure(figsize=(10, 6))
            plt.plot(curve1['Tenors'], curve1['Zero_CC'], label=f'{curve1_name}', linestyle='-')
            plt.plot(curve2['Tenors'], curve2['Zero_CC'], label=f'{curve2_name}', linestyle='-')

            plt.xlabel('Tenors', fontsize=12)
            plt.ylabel('Zero Rates', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)

            if output_path:
                filename = f"{output_path}/{scenario}_{'with_VA' if 'VA' in curve1_name and 'VA' in curve2_name else ''}.png"
                plt.savefig(filename, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

    def plot_curves_cs_combined(self, output_path=None):
        """
        Plot all credit spread scenarios for a given interest rate level in one figure.

        Args:
            output_path (str, optional): Directory to save the plots. If None, plots are displayed.
        """
        # Extract unique interest rate levels
        interest_levels = set()
        for scenario in self.scenario_curves_dict.keys():
            if "low_interest" in scenario:
                interest_levels.add("low_interest")
            elif "base_interest" in scenario:
                interest_levels.add("base_interest")
            elif "high_interest" in scenario:
                interest_levels.add("high_interest")

        # Process each interest level
        for interest_level in interest_levels:
            curves_to_align = []
            curve_labels = []

            # Collect all VA curves for this interest level
            for scenario, curves in self.scenario_curves_dict.items():
                if interest_level in scenario:
                    for curve_name, curve_data in curves.items():
                        if 'VA' in curve_name:
                            curves_to_align.append(curve_data)
                            curve_labels.append(f"{scenario} - {curve_name}")

            if not curves_to_align:
                continue  # Skip if no curves exist for this interest level

            # Align all dataframes
            aligned_curves = self.align_dataframes(*curves_to_align)

            # --- PLOT ALL CURVES WITH VA ---
            plt.figure(figsize=(10, 6))
            for curve_data, label in zip(aligned_curves, curve_labels):
                plt.plot(curve_data['Tenors'], curve_data['Zero_CC'], label=label, linestyle='-')

            plt.xlabel('Tenors', fontsize=12)
            plt.ylabel('Zero Rates', fontsize=12)
            plt.legend(fontsize=8)
            plt.grid(True)

            if output_path:
                plt.savefig(f"{output_path}/{interest_level}_with_VA.png", bbox_inches="tight")
            else:
                plt.show()
            plt.close()
