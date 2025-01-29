import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import griddata


class ImpactDensityPlotter:
    def __init__(self, results_impact_df):
        """
        Initializes the heatmap visualizer with a DataFrame.
        :param results_impact_df: Pandas DataFrame with columns ['Liability Size', 'Liability Duration', 'Own Funds Impact'].
        """
        self.results_impact_df = results_impact_df

    def create_heatmap(self, scenario, output_path=None):
        """
        Creates and displays a density-based 2D heatmap from the results_impact_df DataFrame.
        """
        x = self.results_impact_df['Liability Size'].values
        y = self.results_impact_df['Liability Duration'].values
        z = self.results_impact_df['Own Funds Impact'].values

        # Create a grid for interpolation
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate z-values onto the grid
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        plt.figure(figsize=(10, 6))

        # Use contourf to create the density heatmap
        contour = plt.contourf(xi, yi, zi, levels=100, cmap='coolwarm')

        # Add colorbar to indicate z-values
        plt.colorbar(contour, label='Own Funds Impact')
        plt.xlabel('Liability Size')
        plt.ylabel('Liability Duration')

        if output_path:
            filename = f'{scenario}.png'
            plt.savefig(f'{output_path}/{filename}', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()
