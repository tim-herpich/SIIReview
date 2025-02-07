"""
Module for plotting impact analysis results as heatmaps and line plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


class ImpactPlotter:
    """
    Class for visualizing impact analysis results.
    """

    def __init__(self, results_impact_df: pd.DataFrame, asset_size: float, asset_duration: float):
        """
        Initialize the ImpactPlotter.

        Args:
            results_impact_df (pd.DataFrame): DataFrame with impact results. Expected columns include:
                'Scenario', 'Liabilities', 'Liability Duration', and 'Own Funds Impact rel.'.
            asset_size (float): Reference asset size for scaling.
            asset_duration (float): Reference asset duration for scaling.
        """
        self.results_impact_df = results_impact_df
        self.asset_size = asset_size
        self.asset_duration = asset_duration

        # Define a custom colormap (Orange to Dark Blue)
        colors = [
            (236 / 255, 124 / 255, 36 / 255),  # Orange
            (4 / 255, 60 / 255, 88 / 255)      # Dark Blue
        ]
        self.custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=256)

    def create_impact_density_plot(self, scenario: str, output_path: str = None) -> None:
        """
        Create and save (or display) a density-based 2D heatmap for a given scenario.

        The x-axis shows the ratio of Liabilities to Asset Volume, the y-axis the ratio of
        Liability Duration to Asset Duration, and the color indicates the relative Own Funds Impact.

        Args:
            scenario (str): The scenario name to filter the data.
            output_path (str, optional): Directory to save the plot. If not provided, the plot is displayed.
        """
        # Filter the data for the specified scenario
        scenario_df = self.results_impact_df[self.results_impact_df['Scenario'] == scenario]
        if scenario_df.empty:
            raise ValueError(f"No data available for scenario '{scenario}'.")

        # Scale the x and y values by the asset size and duration
        x = scenario_df['Liabilities'].values / self.asset_size
        y = scenario_df['Liability Duration'].values / self.asset_duration
        z = scenario_df['Own Funds Impact rel.'].values

        # Create a regular grid over the data domain
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate z-values on the grid using cubic interpolation
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        plt.figure(figsize=(10, 6))
        contour = plt.contourf(xi, yi, zi, levels=100, cmap=self.custom_cmap)
        cbar = plt.colorbar(contour)
        cbar.set_label('Own Funds Impact rel.')

        plt.xlabel('Liability Volume / Asset Volume')
        plt.ylabel('Liability Duration / Asset Duration')

        # Set tick labels as fractions (formatted to two decimal places)
        x_ticks = np.linspace(x.min(), x.max(), 5)
        y_ticks = np.linspace(y.min(), y.max(), 5)
        plt.xticks(x_ticks, [f"{tick:.2f}" for tick in x_ticks])
        plt.yticks(y_ticks, [f"{tick:.2f}" for tick in y_ticks])

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filename = os.path.join(output_path, f"{scenario}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_liability_size_vs_impact_overlay(self, output_path: str = None) -> None:
        """
        Create overlay line plots of Own Funds Impact (relative) for each scenario
        at different liability sizes.

        For each unique liability size, the plot shows how the relative impact changes
        with (scaled) Liability Duration.

        Args:
            output_path (str, optional): Directory to save the plots. If not provided, the plots are displayed.
        """
        unique_liability_sizes = sorted(self.results_impact_df["Liabilities"].unique())

        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
        linestyles = ['-', '--', '-.', ':']

        for liability_size in unique_liability_sizes:
            plt.figure(figsize=(10, 6))
            filtered_df = self.results_impact_df[self.results_impact_df["Liabilities"] == liability_size]
            if filtered_df.empty:
                continue

            for idx, (scenario, group_df) in enumerate(filtered_df.groupby("Scenario")):
                if group_df.empty:
                    continue

                # Normalize liability duration
                x = group_df["Liability Duration"].values / self.asset_duration
                y = group_df["Own Funds Impact rel."].values

                marker_style = markers[idx % len(markers)]
                linestyle = linestyles[idx % len(linestyles)]
                plt.plot(x, y, marker=marker_style, linestyle=linestyle, label=scenario)

            plt.xlabel('Liability Duration / Asset Duration', fontsize=12)
            plt.ylabel('Own Funds Impact rel.', fontsize=12)
            plt.grid(True)
            plt.legend(fontsize=8)

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(output_path, f"liability_size_{liability_size}.png")
                plt.savefig(filename, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
                plt.close()

    def export_impact_data(self, scenario: str, output_path: str = None) -> None:
        """
        Export the impact data for a given scenario to a CSV file.

        Args:
            scenario (str): The scenario name.
            output_path (str, optional): Directory to save the CSV file.
        """
        scenario_df = self.results_impact_df[self.results_impact_df['Scenario'] == scenario]
        if scenario_df.empty:
            raise ValueError(f"No data available for scenario '{scenario}'.")
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filename = os.path.join(output_path, f"{scenario}.csv")
            scenario_df.to_csv(filename, index=False)
        else:
            print(f"No output path given.\n\nPrint: {scenario_df}")
