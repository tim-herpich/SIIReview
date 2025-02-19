# Interest Rate Curve Extrapolation & Impact Analysis

This project generates and compares interest rate curves using the current **Solvency II (SII)** and the proposed **future Solvency II Review** methodologies. Explicitly, it implements both the **Smith-Wilson** and **Dutch (Alternative) extrapolation methods**. The **volatility adjustment (VA)** is treated as an input for Smith-Wilson, it computed dynamically for the Dutch method based on the proposed SII Review framework. Additionally, the project assesses the **impact on a unit face value ZCB** resulting from different interest rate curve extrapolation methodologies.

---

## 🚀 Features

- **Market Data Loader**: Parses market data from Excel workbooks.
- **Parameter Handling**: Manages curve settings, market scenarios, and portfolio parameters.
- **Bootstrapping**: Builds zero rate and forward rate curves from input market data.
- **Extrapolation Methods**: 
  - **Alternative Extrapolation** using the Dutch (Alternative) approach.
  - **Smith–Wilson Extrapolation** using the Smith-Wilson approach.
- **VA Spread Calculation**: Computes the Volatility Adjustment (VA) as per the SII Review.
- **Impact Analysis**: Assesses discounting impact under different extrapolation methodologies.
- **Scenario Runner**: A module that orchestrates data adjustments, bootstrapping, extrapolation (with and without VA) and impact analysis for each market scenario.
- **Plotting and Visualization**: Generates plots for:
  - **Extrapolated Yield Curves**: Comparison between extrapolation methods under different market scenarios.
  - **Impact Assessments** Impact assessments due to different extrapolation methods under different market scenarios.
---

## 📁 Project Structure

The repository is organized as follows:

```
SII REVIEW TOOL/
│
├── extrapolation/                   # Extrapolation methods implementation
│   ├── alternative.py                 # Dutch (Alternative) extrapolation method
│   ├── smithwilson.py                 # Smith-Wilson extrapolation method
│
├── inputs/                           # Input market data files
│   ├── rates.xlsx                      # Market interest rates
│   ├── spreads.xlsx                    # VA spread data
│
├── outputs/                          # Output results storage
│   ├── curves/                         # Generated interest rate curves
│   ├── impacts/                        # Own funds impact assessment results
│
├── plots/                            # Visualization and plotting scripts
│   ├── curveplotter.py                 # Generates extrapolated curve plots
│   ├── impactplotter.py                # Generates impact assessment visualizations
│
├── tests/                            # Unit tests for validation
│   ├── test_bootstrapping.py           # Tests for bootstrapping methods
│   ├── test_extrapolation_alt.py       # Tests for Alternative Extrapolation
│   ├── test_extrapolation_sw.py        # Tests for Smith-Wilson Extrapolation
│   ├── test_impact_calculator.py       # Tests for impact assessment calculations
│   ├── test_marketdata.py              # Tests for market data loader
│   ├── test_parameters.py              # Tests for parameter handling
│   ├── test_scenariorunner.py          # Tests for the ScenarioRunner class
│
├── .gitignore                         # Ignore file for Git
├── .gitlab-ci.yml                      # GitLab CI/CD pipeline configuration
├── bootstrapping.py                   # Bootstrapping methodology implementation
├── impact.py                          # Discounting impact assessment module
├── main.py                            # Main execution script
├── marketdata.py                      # Market data loader module
├── parameters.py                      # Configuration and parameters module
├── README.md                          # Project documentation (this file)
├── scenariorunner.py                  # Orchestrates scenario-specific processing (new)
├── va.py                              # Volatility Adjustment (VA) calculation module
```

---

## 📊 Output and Visualization

### **1. Interest Rate Curve Generation**
After running the main script, bootstrapped and extrapolated yield curve plots are saved in the `outputs/curves/plots` directory.
The underlying data is saved in the `outputs/curves/data` directory.

### **2. Impact Assessment Reports**
Impact assessment plots are generated in `outputs/impacts/plots`.
The underlying data is saved in the `outputs/impacts/data` directory.


---
