"""
Main module for bootstrapping, extrapolation and impact analysis.
"""

from marketdata import MarketData
from parameters import Parameters
from scenariorunner import ScenarioRunner
from plots.curveplotter import CurvePlotter
from plots.impactplotter import ImpactPlotter


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
    curve_plotter.export_curve_data(output_path='outputs/curves/data/')
    curve_plotter.compute_curve_differences()
    curve_plotter.export_curve_differences_data(output_path='outputs/curves/')
    curve_plotter.plot_curve_differences(
        llp=params.LLP_SW, output_path='outputs/curves/')

    # Plot and export impact results using the ImpactPlotter
    print(f"Plot and export impacts...")
    impact_plotter = ImpactPlotter(scenario_impact_dict)
    impact_plotter.plot_impact_barchart(output_path='outputs/impacts/plots/')
    impact_plotter.export_impact_data(output_path='outputs/impacts/data/')


if __name__ == '__main__':
    main()
