"""
Module for plotting the impact analysis results as grouped bar charts.
Each chart displays, for a given scenario, the present value (PV) of a unit ZCB
for different maturities using two discount curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ImpactPlotter:
    """
    Class for visualizing the ZCB present value impact results.
    """

    def __init__(self, impact_data: dict):
        """
        Initialize the ImpactPlotter.

        Args:
            impact_data (dict): Dictionary where keys are scenario names and values are DataFrames
                                with columns: 'Maturity', 'PV_Alt' and 'PV_SW'.
        """
        self.impact_data = impact_data

    def plot_impact_barchart(self, output_path: str = None) -> None:
        """
        For each scenario in the impact data, plot a grouped bar chart where the x-axis shows
        the maturity and the y-axis shows the PV of a unit face value ZCB computed
        using the two discount curves.

        Three bars are plotted per maturity:
        - PV Alternative Extrapolation
        - PV Smith-Wilson Extrapolation
        - PV (Alt - SW) (difference between the two), colored green if positive, red if negative.

        Args:
            output_path (str, optional): Directory to save the plots. If not provided, the plots are displayed.
        """
        for scenario, scenario_df in self.impact_data.items():
            maturities = scenario_df['Maturity'].values
            pv_alt = scenario_df['PV Alternative Extrapolation'].values
            pv_sw = scenario_df['PV Smith-Wilson Extrapolation'].values
            pv_delta = scenario_df['PV (Alt - SW)'].values
            x = np.arange(len(maturities))
            width = 0.3  # Adjusted width to fit all 3 bars

            plt.figure(figsize=(12, 6))

            # Three grouped bars: Alternative, Smith-Wilson, and Delta
            plt.bar(x - width, pv_alt, width,
                    label='PV Alternative Curve')
            plt.bar(x, pv_sw, width, label='PV Smith-Wilson Curve')

            # Conditional colors for PV (Alt - SW): Green if positive, Red if negative
            delta_colors = ['#2ca02c' if delta >
                            0 else '#d62728' for delta in pv_delta]
            plt.bar(x + width, pv_delta, width,
                    label='PV (Alt - SW)', color=delta_colors)

            # Create a custom legend for PV Delta
            legend_handles = [
                mpatches.Patch(color='#1f77b4', label='PV Alternative Curve'),
                mpatches.Patch(color='#ff7f0e', label='PV Smith-Wilson Curve'),
                mpatches.Patch(color='#2ca02c',
                               label='PV (Alt - SW) (Positive)'),
                mpatches.Patch(color='#d62728',
                               label='PV (Alt - SW) (Negative)')
            ]
            plt.legend(handles=legend_handles, fontsize=10)

            plt.ylim(-0.1, 1.0)  # Expand slightly below min for readability
            plt.xlabel('Maturity (Years)', fontsize=12)
            plt.ylabel('PV at LLP of unit CF', fontsize=12)
            plt.xticks(x, maturities)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(output_path, f"impact_{scenario}.png")
                plt.savefig(filename, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()

    def export_impact_data(self, output_path: str = None) -> None:
        """
        Exports the scenario-specific impact data to a CSV file.

        Args:
            output_path (str, optional): Directory to save the CSV file.
        """
        for scenario, scenario_df in self.impact_data.items():
            if scenario_df.empty:
                raise ValueError(
                    f"No data available for scenario '{scenario}'.")
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(output_path, f"{scenario}.csv")
                scenario_df.to_csv(filename, index=False)
            else:
                print(f"No output path given.\n\nPrint: {scenario_df}")
