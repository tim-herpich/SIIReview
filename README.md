# Interest Rate Curve Bootstrapping, Extrapolation & Impact Analysis

This project implements a comprehensive system for bootstrapping zero rate curves, extrapolating these curves (using both alternative and Smith–Wilson methods), computing valuation adjustments (VA) and finally assessing the impact on portfolio equity. In addition, it provides plotting functionality to visualize both the curves and the impact analysis.

## Features

- **Market Data Loader**: Read market data from Excel workbooks.
- **Parameters Container**: Store curve, scenario, and portfolio parameters.
- **Bootstrapping**: Build zero rate and forward curves from input rates.
- **Extrapolation Methods**: 
  - **Alternative Extrapolation** using a Dutch approach.
  - **Smith–Wilson Extrapolation** for more advanced curve fitting.
- **VA Spread Calculation**: Compute the new VA spread based on market inputs.
- **Impact Analysis**: Assess portfolio equity impact when discount curves change.
- **Plotting**: Generate plots for curves and impact analysis (heatmaps, line plots).

## Project Structure

Below is an example folder structure for the project:

## Requirements

- Python 3.8+
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [pytest](https://docs.pytest.org/) (for running unit tests)

You can install the Python dependencies with:

```bash
pip install -r requirements.txt
