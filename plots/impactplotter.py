"""
Module for plotting the impact analysis results as grouped bar charts.
Each chart displays, for a given scenario, the present value (PV) of a unit ZCB
for different maturities using two discount curves, plus the difference (Alternative - Smith-Wilson Curve).
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
            impact_data (dict): Dictionary where keys are scenario names and
                                values are DataFrames with columns:
                                'Maturity',
                                'PV Alternative Extrapolation',
                                'PV Smith-Wilson Extrapolation',
                                'PV (Alternative - Smith-Wilson Curve)'.
        """
        self.impact_data = impact_data

    def plot_impact_barchart(self, output_path: str = None) -> None:
        """
        Plot a grouped bar chart for each scenario with a secondary axis for PV (Alternative - Smith-Wilson Curve),
        ensuring both axes' zero lines line up at the same vertical position, while also
        retaining each axis's data range. Extra margins are added to avoid truncation.
        """
        for scenario, scenario_df in self.impact_data.items():
            maturities = scenario_df['Maturity'].values
            pv_alt = scenario_df['PV Alternative Extrapolation'].values
            pv_sw = scenario_df['PV Smith-Wilson Extrapolation'].values
            pv_delta = scenario_df['PV (Alt - SW)'].values

            x = np.arange(len(maturities))
            width = 0.25

            fig, ax1 = plt.subplots(figsize=(14, 8))

            # Plot PV Alternative and PV Smith-Wilson on primary axis
            ax1.bar(
                x - width,
                pv_alt,
                width,
                label='PV Alternative Curve',
                color='#1f77b4'
            )
            ax1.bar(
                x,
                pv_sw,
                width,
                label='PV Smith-Wilson Curve',
                color='#ff7f0e'
            )

            # Create secondary axis for PV (Alternative - Smith-Wilson Curve)
            ax2 = ax1.twinx()
            ax2.bar(
                x + width,
                pv_delta,
                width,
                label='PV (Alternative - Smith-Wilson Curve)',
                color='#2ca02c'
            )

            # 1) Add margins so auto-scaling won't cut off top/bottom bars
            ax1.margins(y=0.25)

            # 2) Let Matplotlib finalize auto-limits for both axes
            plt.draw()

            # Get the auto-limits
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()

            # 3) Compute zero-line fraction for each axis
            # 4) Shift the axis whose zero is "lower" so that it lines up
            
            # catch special case where both axis ranges are disjoint 
            if abs(y1_min) <= 1e-6 and abs(y2_max) <= 1e-6:
                y1_min = y1_min - y1_max 
                y2_max = y2_max + abs(y2_min)
                ax1.set_ylim(y1_min, y1_max)
                ax2.set_ylim(y2_min, y2_max)

            else:
                def zero_fraction(ymin, ymax):
                    rng = (ymax - ymin) if (ymax != ymin) else 1e-12
                    return abs(ymin) / rng  # fraction from bottom to 0

                ax1_zero_pos = zero_fraction(y1_min, y1_max)
                ax2_zero_pos = zero_fraction(y2_min, y2_max)

                #    with the axis whose zero is "higher."
                if ax1_zero_pos > ax2_zero_pos:
                    # Shift ax2's entire range
                    delta = (ax1_zero_pos - ax2_zero_pos) * (y2_max - y2_min)
                    ax2.set_ylim(y2_min - delta, y2_max - delta)
                else:
                    # Shift ax1
                    delta = (ax2_zero_pos - ax1_zero_pos) * (y1_max - y1_min)
                    ax1.set_ylim(y1_min - delta, y1_max - delta)

            # 5) Draw horizontal zero lines
            ax1.axhline(0, color='black', linewidth=1)
            ax2.axhline(0, color='black', linewidth=1)

            # 6) Configure labels, grid, etc.
            ax1.set_xlabel('Maturity (Years)', fontsize=16)
            ax1.set_ylabel('PV of Unit CF', fontsize=16)
            ax2.set_ylabel('PV (Alternative - Smith-Wilson Curve)', fontsize=16, color='gray')
            ax1.tick_params(axis='both', labelsize=16)
            ax2.tick_params(axis='both', labelsize=16, labelcolor='gray')
            ax1.set_xticks(x)
            ax1.set_xticklabels(maturities)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)

            # Create legend above the plot
            legend_handles = [
                mpatches.Patch(color='#1f77b4', label='PV Alternative Curve'),
                mpatches.Patch(color='#ff7f0e', label='PV Smith-Wilson Curve'),
                mpatches.Patch(color='#2ca02c', label='PV (Alternative - Smith-Wilson Curve)')
            ]
            fig.legend(
                handles=legend_handles,
                fontsize=13.5,
                loc='upper center',
                ncol=3,
                frameon=True
            )

            # Adjust layout to fit the legend above the plot
            fig.tight_layout(rect=[0, 0.1, 1, 0.9])

            # Save or show plot
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
                raise ValueError(f"No data available for scenario '{scenario}'.")
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(output_path, f"{scenario}.csv")
                scenario_df.to_csv(filename, index=False)
            else:
                print(f"No output path given.\n\nPrint: {scenario_df}")

    