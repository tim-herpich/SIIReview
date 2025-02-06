# Interest Rate Curve Bootstrapping, Extrapolation & Impact Analysis

This project generates and compares interest rate curves using the current **Solvency II (SII)** and the proposed **future Solvency II Review** methodologies.  
Explicitly, it implements both the **Smith-Wilson** and **Dutch (Alternative) extrapolation methods**. The **volatility adjustment (VA)** is treated as an input for Smith-Wilson, it computed dynamically for the Dutch method based on the proposed SII Review framework.  
Additionally, the project assesses the **impact on own funds** resulting from different interest rate curve extrapolation methodologies.

---

## 🚀 Features

- **Market Data Loader**: Parses market data from Excel workbooks.
- **Parameter Handling**: Manages curve settings, market scenarios, and portfolio parameters.
- **Bootstrapping**: Builds zero rate and forward rate curves from input market data.
- **Extrapolation Methods**: 
  - **Alternative Extrapolation** using the Dutch (Alternative) approach.
  - **Smith–Wilson Extrapolation** using the Smith-Wilson approach.
- **VA Spread Calculation**: Computes the Volatility Adjustment (VA) as per the SII Review.
- **Impact Analysis**: Assesses own funds impact under different extrapolation methodologies.
- **Plotting and Visualization**: Generates plots for:
  - **Extrapolated Yield Curves**
  - **Impact Assessments** (CSV data exports available)

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
│
├── .gitignore                         # Ignore file for Git
├── .gitlab-ci.yml                      # GitLab CI/CD pipeline configuration
├── bootstrapping.py                   # Bootstrapping methodology implementation
├── impact.py                          # Own funds impact assessment module
├── main.py                            # Main execution script
├── marketdata.py                      # Market data loader module
├── parameters.py                      # Configuration and parameters module
├── README.md                          # Project documentation (this file)
├── va.py                              # Volatility Adjustment (VA) calculation module
```

---

## 📌 Usage

### **Running the Full Workflow**
To execute the full process (loading market data, extrapolation, and impact assessment), run:

```bash
python main.py
```

### **Running Tests**
To validate the implementation, execute:

```bash
pytest tests/
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

## 🏗️ Contributions

Contributions to this project are highly encouraged! Follow these steps to contribute:

1. **Fork** this repository.
2. Create a **new branch** for your feature or fix.
3. **Commit** your changes with a descriptive message.
4. **Push** to your branch.
5. Open a **Pull Request** for review.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📞 Contact

For questions, suggestions, or issues, please feel free to reach out via:

- **Email**: [tim.herpich@d-fine.com](tim.herpich@d-fine.com)

---

**Developed with for the Makret and Non-Financial Risk Cluster**

