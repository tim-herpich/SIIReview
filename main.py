"""
Main module for bootstrapping, extrapolation, impact analysis and sensitivity analysis.
"""

from marketdata import MarketData
from parameters import Parameters
from scenariorunner import ScenarioRunner
from portfoliosensitivity import PortfolioSensitivity
from plots.curveplotter import CurvePlotter
from plots.impactplotter import ImpactPlotter
from plots.impactdensityplotter import ImpactDensityPlotter
from extrapolationadditionalanalyzer import ExtrapolationAdditionalAnalyzer


def main():
    # Load market data from Excel using MarketData objects
    rates_md = MarketData(filepath="inputs/rates.xlsx")
    spreads_md = MarketData(filepath="inputs/spreads.xlsx")
    rates_md.open_workbook()
    spreads_md.open_workbook()

    # Parse sheets
    df_alt = rates_md.parse_sheet_to_df('zero_rates_alt')
    df_sw = rates_md.parse_sheet_to_df('zero_rates_sw')
    va_spreads_df = spreads_md.parse_sheet_to_df('spreads_va')
    va_spreads_df.set_index('Issuer', inplace=True)

    rates_md.close_workbook()
    spreads_md.close_workbook()

    # Load parameters (which include market scenarios)
    params = Parameters()

    # Process all scenarios
    scenario_curves_dict = {}
    scenario_impact_dict = {}
    for scenario in params.scenarios:
        print(f"Processing scenario: {scenario['name']}")
        runner = ScenarioRunner(scenario, df_alt, df_sw, va_spreads_df, params)
        curves, impact_df = runner.run()
        scenario_curves_dict[scenario['name']] = curves
        scenario_impact_dict[scenario['name']] = impact_df

    # Plot and export curves using the CurvePlotter
    print(f"Plot and export curves...")
    curve_plotter = CurvePlotter(scenario_curves_dict)
    curve_plotter.plot_curves(
        llp=params.LLP_SW, output_path='outputs/curves/plots/')
    curve_plotter.plot_curves_cs_combined(
        llp=params.LLP_SW, output_path='outputs/curves/plots/cs_combined/')
    curve_plotter.plot_low_high_no_va_curves(
        llp=params.LLP_SW, output_path='outputs/curves/plots/')
    curve_plotter.export_curve_data(output_path='outputs/curves/data/')
    curve_plotter.compute_curve_differences()
    curve_plotter.export_curve_differences_data(output_path='outputs/curves/')
    curve_plotter.plot_curve_differences(
        llp=params.LLP_SW, output_path='outputs/curves/')

    # Plot and export impact results using the ImpactPlotter
    print(f"Plot and export base impacts...")
    bar_impact_plotter = ImpactPlotter(scenario_impact_dict)
    bar_impact_plotter.plot_impact_barchart(
        output_path='outputs/impacts/plots/')
    bar_impact_plotter.export_impact_data(output_path='outputs/impacts/data/')

    # --- Run sensitivity analysis ---
    if params.analysis_parameters:
        print(f"Run sensitivity analysis...")
        analyzer = PortfolioSensitivity(df_alt, df_sw, va_spreads_df, params)
        sensitivity_df = analyzer.run_analysis()
        # --- Export sensitivity analysis results via the density plotter class ---
        density_plotter = ImpactDensityPlotter(sensitivity_df)
        density_plotter.export_data(
            output_path='outputs/sensitivity', filename="sensitivity_analysis.csv")
        # --- Plot density plots for sensitivity analysis ---
        print(f"Plot and export sensitivity analysis impacts...")
        density_plotter.create_all_impact_density_plots(
            output_path='outputs/sensitivity/plots')

    # --- Additional Extrapolation Analysis ---
    if params.additional_analysis:
        print("Running additional extrapolation analysis...")
        additional_analyzer = ExtrapolationAdditionalAnalyzer(
            cp=params,
            df_alt=df_alt,
            df_sw=df_sw
        )
        # The analyzer is responsible for bootstrapping internally based on the selected scenario.
        additional_results = additional_analyzer.run_additional_analysis()
        # Use an extended plotter method to plot the additional curves
        curve_plotter.plot_shifted_illiquid_curves(
            additional_results, llp=params.LLP_SW, output_path="outputs/additional_curves/plots")
        curve_plotter.export_shifted_illiquid_curves_data(
            additional_results, output_path="outputs/additional_curves/data")


if __name__ == '__main__':
    main()
