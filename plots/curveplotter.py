import matplotlib.pyplot as plt

class CurvePlotter:
    """
    A class to plot specified combinations of discount curves vs. tenors for given dataframes.
    """
    def __init__(self, curves):
        """
        Initialize the CurvePlotter with the dataframes.

        Args:
            curves (dict): A dictionary containing curve names and corresponding dataframes.
        """
        self.curves = curves

    def align_dataframes(self, curve1, curve2):
        """
        Ensure that two dataframes have the same length by aligning their tenors.

        Args:
            curve1 (pd.DataFrame): The first dataframe.
            curve2 (pd.DataFrame): The second dataframe.

        Returns:
            tuple: Aligned dataframes (curve1_aligned, curve2_aligned).
        """
        common_tenors = curve1['Tenors'].isin(curve2['Tenors']) & curve2['Tenors'].isin(curve1['Tenors'])
        curve1_aligned = curve1[curve1['Tenors'].isin(curve2['Tenors'])]
        curve2_aligned = curve2[curve2['Tenors'].isin(curve1['Tenors'])]
        return curve1_aligned.reset_index(drop=True), curve2_aligned.reset_index(drop=True)

    def plot_comparison(self, pairs, scenario, output_path=None):
        """
        Plot specified 2-curve comparisons of Discount vs. Tenors.

        Args:
            pairs (list of tuples): List of tuples containing curve name pairs to compare.
            output_path (str, optional): Directory to save the plots. If None, plots are displayed.
        """
        for curve1_name, curve2_name in pairs:
            curve1 = self.curves[curve1_name]
            curve2 = self.curves[curve2_name]

            # Align dataframes by common tenors
            curve1, curve2 = self.align_dataframes(curve1, curve2)

            plt.figure(figsize=(10, 6))
            plt.plot(curve1['Tenors'], curve1['Zero_CC'], label=f'{curve1_name}', marker='o')
            plt.plot(curve2['Tenors'], curve2['Zero_CC'], label=f'{curve2_name}', marker='x')

            # plt.title(f'Comparison: {curve1_name} vs. {curve2_name}', fontsize=14)
            plt.xlabel('Tenors', fontsize=12)
            plt.ylabel('Zero Rates', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)

            if output_path:
                filename = f'{scenario}_{curve1_name}_vs_{curve2_name}.png'
                plt.savefig(f'{output_path}/{filename}', bbox_inches='tight')
            else:
                plt.show()

            plt.close()
