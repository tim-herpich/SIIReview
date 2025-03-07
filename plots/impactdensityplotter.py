"""
Module for visualizing sensitivity analysis results as a density-based heatmap.
Also provides an export function to save the sensitivity data.
"""

import os
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


class ImpactDensityPlotter:
    """
    Class for visualizing sensitivity analysis results as a density-based heatmap.
    The plot shows:
      - x-axis: Asset Size / Liability Size ratio,
      - y-axis: Asset Duration / Liability Duration ratio,
      - z-axis: Present Value (PV) of a unit cash flow at a selected maturity.
    Also provides an export method to save the sensitivity data to CSV.
    """

    def __init__(self, results_impact_df: pd.DataFrame):
        """
        Initializes the density plot visualizer.

        Args:
            results_impact_df (pd.DataFrame): DataFrame with columns:
              'Asset/Liability Ratio', 'Asset Duration/Liability Duration Ratio', and PV columns (e.g., 'PV (Alt - SW) at 10 years')
        """
        self.results_impact_df = results_impact_df

        colors = [(236/255, 124/255, 36/255), (4/255, 60/255, 88/255)]
        self.custom_cmap = LinearSegmentedColormap.from_list(
            "custom_gradient", colors, N=256)

    def create_impact_density_plot(self, selected_tenor: str, output_path: str = None):
        """
        Creates and saves/displays a density heatmap of the sensitivity analysis.

        Args:
            selected_tenor (str): The PV column to use (e.g., 'PV (Alt - SW) at 10 years').
            scenario (str): Scenario name (used in the title and filename).
            output_path (str): If provided, the plot is saved to this path; otherwise it is displayed.
        """
        x = self.results_impact_df['Asset/Liability Ratio'].values
        y = self.results_impact_df['Asset Duration/Liability Duration Ratio'].values
        z = self.results_impact_df[selected_tenor].values

        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        plt.figure(figsize=(10, 6))
        contour = plt.contourf(xi, yi, zi, levels=100, cmap=self.custom_cmap)

        # Format color bar to show 2 decimal places
        cbar = plt.colorbar(contour)
        cbar.set_label(selected_tenor)
        cbar.formatter = ticker.FormatStrFormatter('%.3f')
        cbar.update_ticks()

        plt.xlabel('Asset Size / Liability Size')
        plt.ylabel('Asset Duration / Liability Duration')
        x_ticks = np.linspace(x.min(), x.max(), 5)
        y_ticks = np.linspace(y.min(), y.max(), 5)
        plt.xticks(x_ticks, [f"{tick:.2f}" for tick in x_ticks])
        plt.yticks(y_ticks, [f"{tick:.2f}" for tick in y_ticks])
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            year = selected_tenor.split('at ')[1].split(' Y')[0]
            filename = f'portfolio_sensitivity_{year}_years.png'
            plt.savefig(os.path.join(output_path, filename),
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()

    def create_all_impact_density_plots(self, output_path: str = None):
        """
        Automatically plots density heatmaps for all PV columns in the dataset.

        Args:
            output_path (str): If provided, saves plots to this path; otherwise displays them.
        """
        # Identify PV columns dynamically
        pv_columns = [col for col in self.results_impact_df.columns if col.startswith(
            'PV (Alt - SW) at')]

        for tenor in pv_columns:
            print(f"Generating density plot for {tenor}...")
            self.create_impact_density_plot(
                selected_tenor=tenor, output_path=output_path)

    def export_data(self, output_path: str, filename: str = "portfolio_sensitivity.csv"):
        """
        Exports the sensitivity analysis data to a CSV file.

        Args:
            output_path (str): Directory path where to save the CSV file.
            filename (str): Name of the CSV file.
        """
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, filename)
        self.results_impact_df.to_csv(file_path, index=False)
