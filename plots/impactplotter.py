import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

class ImpactDensityPlotter:
    def __init__(self, results_impact_df, asset_size, asset_duration):
        """
        Initializes the heatmap visualizer with a DataFrame.
        :param results_impact_df: Pandas DataFrame with columns ['Liability Volume', 'Liability Duration', 'Own Funds Impact'].
        :param asset_size: Reference asset size for fraction scaling.
        :param asset_duration: Reference asset duration for fraction scaling.
        """
        self.results_impact_df = results_impact_df
        self.asset_size = asset_size
        self.asset_duration = asset_duration

        # Define custom colormap based on the provided RGB values
        colors = [(236/255, 124/255, 36/255),  # RGB(236,124,36) - Orange
                  (4/255, 60/255, 88/255)]    # RGB(4,60,88) - Dark Blue
        self.custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=256)

    def create_heatmap(self, scenario, output_path=None):
        """
        Creates and displays a density-based 2D heatmap from the results_impact_df DataFrame.
        """
        # Convert x and y values to fractions of asset_size and asset_duration
        x = self.results_impact_df['Liabilities'].values / self.asset_size
        y = self.results_impact_df['Liability Duration'].values / self.asset_duration
        z = self.results_impact_df['Own Funds Impact rel.'].values

        # Create a grid for interpolation
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate z-values onto the grid
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        plt.figure(figsize=(10, 6))

        # Use contourf to create the density heatmap with the custom colormap
        contour = plt.contourf(xi, yi, zi, levels=100, cmap=self.custom_cmap)

        # Add colorbar to indicate z-values
        plt.title('Own Funds Impact')
        cbar = plt.colorbar(contour)
        cbar.set_label('Own Funds Impact')

        # Set axis labels
        plt.xlabel('Liability Volume / Asset Volume')
        plt.ylabel('Liability Duration / Asset Duration')

        # Adjust tick labels to display fractions
        x_ticks = np.linspace(x.min(), x.max(), 5)
        y_ticks = np.linspace(y.min(), y.max(), 5)
        plt.xticks(x_ticks, [f"{tick:.2f}" for tick in x_ticks])
        plt.yticks(y_ticks, [f"{tick:.2f}" for tick in y_ticks])

        # Save or display the plot
        if output_path:
            filename = f'{scenario}.png'
            plt.savefig(f'{output_path}/{filename}', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()

    def export_data(self, scenario, output_path=None):
        """
        Exports the dataframe underlying the heat map
        """
        # Save or display the plot
        if output_path:
            filename = f'{scenario}.csv'
            self.results_impact_df.to_csv(f'{output_path}/{filename}', index=False)
