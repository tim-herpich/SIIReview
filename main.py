"""
Main module for the interest rate curve bootstrapping, extrapolation application and impact analysis.
"""

import numpy as np
import pandas as pd
from marketdata import MarketData
from parameters import Parameters
from bootstrapping import Bootstrapping
from extrapolation.alternative import ExtrapolationAlt
from extrapolation.smithwilson import ExtrapolationSW
from va import VaSpreadCalculator
from impact import ImpactCalculator
from plots.curveplotter import CurvePlotter
from plots.impactplotter import ImpactPlotter


def main():
    """
    Main function to run the curve bootstrapping, extrapolation and impact analysis.
    """
    # Load market data from Excel files
    input_rates = MarketData(filepath="inputs/rates.xlsx")
    input_spreads = MarketData(filepath="inputs/spreads.xlsx")
    input_rates.open_workbook()
    input_spreads.open_workbook()
    df_alt = input_rates.parse_sheet_to_df('zero_rates_alt')
    df_sw = input_rates.parse_sheet_to_df('zero_rates_sw')
    va_spreads_df = input_spreads.parse_sheet_to_df('spreads_va')
    va_spreads_df.set_index('Issuer', inplace=True)
    input_rates.close_workbook()
    input_spreads.close_workbook()

    # Instantiate business logic classes
    bootstr = Bootstrapping()
    ext_alt = ExtrapolationAlt()
    ext_sw = ExtrapolationSW()
    impact_calc = ImpactCalculator()

    # Load curve parameters and scenarios
    cp = Parameters()

    # Dictionary to store scenario curves and list for impact results
    scenario_curves_dict = {}
    results_impact_density = []

    # Iterate over market scenarios
    for scenario in cp.scenarios:
        print(f"Processing scenario: {scenario['name']}")

        # Create copies of data to adjust for scenario-specific shifts
        df_alt_shifted = df_alt.copy()
        df_sw_shifted = df_sw.copy()
        va_spreads_shifted = va_spreads_df.copy()

        # Apply interest rate shifts if required
        if 'high_interest' in scenario['name'] or 'low_interest' in scenario['name']:
            shift = scenario['irshift'] / 10000  # convert bps to decimal
            df_alt_shifted['Input Rates'] += shift
            df_sw_shifted['Input Rates'] += shift

        # Apply spread shifts if required
        if 'high_spreads' in scenario['name']:
            va_spreads_shifted += scenario['csshift'] / 10000  # convert bps to decimal

        # Set the legacy VA spread value from scenario
        cp.VA_value = scenario['vaspread']

        # Prepare arrays for bootstrapping
        max_tenor = cp.max_tenorofAlt
        dlt_array = np.zeros(max_tenor)
        rate_array = np.zeros(max_tenor)
        weight_array = np.zeros(max_tenor)

        for dlt_val, tenor, wgt, rt_dec in zip(
                df_alt_shifted['DLT'],
                df_alt_shifted['Tenor'],
                df_alt_shifted['LLFR Weights'],
                df_alt_shifted['Input Rates']):
            t = int(round(tenor))
            if 1 <= t <= max_tenor:
                idx = t - 1
                dlt_array[idx] = dlt_val
                rate_array[idx] = rt_dec
                weight_array[idx] = wgt

        # Bootstrap the zero curve
        boot_df = bootstr.bootstrap_to_zero_full(
            instrument=cp.instrument,
            rates=rate_array * 100.0,  # convert to percentage
            dlt=dlt_array,
            coupon_freq=cp.coupon_freq,
            compounding_in=cp.compounding_in,
            cra=cp.CRA,
            max_tenor=cp.max_tenorofAlt,
            compounding_out=cp.compounding_out
        )

        # Compute LLFR without VA
        llfr_noVA = ext_alt.get_llfr(
            zero_rates=boot_df['Zero_CC'].values,
            dlt=dlt_array,
            weights=weight_array
        )

        # Alternative extrapolation without VA
        results_alt = ext_alt.alternative_extrapolation(
            zero_rates=boot_df['Zero_CC'].values,
            FSP=cp.FSP,
            UFR=cp.UFR,
            LLFR=llfr_noVA,
            alpha=cp.alpha
        )

        # Calculate new VA spread and update the zero curve
        va_calc = VaSpreadCalculator(
            va_spreads_df=va_spreads_shifted,
            fi_asset_size=cp.fi_asset_size,
            liability_size=cp.liability_size,
            pvbp_fi_assets=cp.pvbp_fi_assets,
            pvbp_liabs=cp.pvbp_liabs
        )
        va_new = va_calc.compute_total_va()

        zero_boot_with_new_VA = ext_alt.zero_boot_withVA(
            fwd_boot_withVA=boot_df['Forward_CC'].copy().values,
            max_tenor=cp.max_tenorofAlt,
            FSP=cp.FSP,
            VA_value=va_new
        )

        # Compute LLFR with the new VA spread
        llfr_with_new_VA = ext_alt.get_llfr(
            zero_rates=zero_boot_with_new_VA,
            dlt=dlt_array,
            weights=weight_array
        )

        # Alternative extrapolation with new VA
        results_alt_with_new_VA = ext_alt.alternative_extrapolation(
            zero_rates=zero_boot_with_new_VA,
            FSP=cp.FSP,
            UFR=cp.UFR,
            LLFR=llfr_with_new_VA,
            alpha=cp.alpha
        )

        # Smith-Wilson extrapolation without VA
        results_sw = ext_sw.smith_wilson_extrapolation(
            instrument=cp.instrument,
            curve_data=df_sw_shifted,
            coupon_freq=cp.coupon_freq,
            CRA=cp.CRA,
            UFR=cp.UFR,
            alpha_min=cp.alpha_min_SW,
            CR=cp.CR_SW,
            CP=cp.CP_SW
        )

        # Smith-Wilson extrapolation with VA
        df_sw_withVA = ext_sw.getInputwithVA(
            zero_rates_extrapolated_ac=results_sw['Zero_AC'].copy(),
            LLP=cp.LLP_SW,
            VA_value=cp.VA_value,
            curve_data=df_sw_shifted
        )
        results_sw_withVA = ext_sw.smith_wilson_extrapolation(
            instrument='Zero',
            curve_data=df_sw_withVA,
            coupon_freq=cp.coupon_freq,
            CRA=0.0,
            UFR=cp.UFR,
            alpha_min=cp.alpha_min_SW,
            CR=cp.CR_SW,
            CP=cp.CP_SW
        )

        # Store the scenario curves in a dictionary
        scenario_curves_dict[scenario["name"]] = {
            'Alternative Extrapolation with VA': results_alt_with_new_VA,
            'Alternative Extrapolation': results_alt,
            'Smith-Wilson Extrapolation with VA': results_sw_withVA[:-1],
            'Smith-Wilson Extrapolation': results_sw[:-1]
        }

        # Plot the curves for this scenario
        curve_plotter = CurvePlotter(curves=scenario_curves_dict[scenario["name"]])
        pairs = [
            ('Alternative Extrapolation with VA', 'Smith-Wilson Extrapolation with VA'),
            ('Alternative Extrapolation', 'Smith-Wilson Extrapolation')
        ]
        curve_plotter.plot_curves(pairs=pairs, scenario=scenario["name"],
                                  output_path='outputs/curves/plots/')

        # Impact assessment over a range of liability sizes and durations
        for liability_size in np.arange(cp.liability_size_low_bound,
                                        cp.liability_size_high_bound,
                                        cp.liability_size_steps):
            for liability_duration in np.arange(cp.liability_duration_low_bound,
                                                cp.liability_duration_high_bound,
                                                cp.liability_duration_steps):

                va_calc_dynamic = VaSpreadCalculator(
                    va_spreads_df=va_spreads_shifted,
                    fi_asset_size=cp.fi_asset_size,
                    liability_size=liability_size,
                    pvbp_fi_assets=cp.pvbp_fi_assets,
                    pvbp_liabs=0.1 * liability_duration
                )
                va_new_dynamic = va_calc_dynamic.compute_total_va()

                zero_boot_with_new_VA_dynamic = ext_alt.zero_boot_withVA(
                    fwd_boot_withVA=boot_df['Forward_CC'].copy().values,
                    max_tenor=cp.max_tenorofAlt,
                    FSP=cp.FSP,
                    VA_value=va_new_dynamic
                )

                llfr_with_new_VA_dynamic = ext_alt.get_llfr(
                    zero_rates=zero_boot_with_new_VA_dynamic,
                    dlt=dlt_array,
                    weights=weight_array
                )

                results_alt_with_new_VA_dynamic = ext_alt.alternative_extrapolation(
                    zero_rates=zero_boot_with_new_VA_dynamic,
                    FSP=cp.FSP,
                    UFR=cp.UFR,
                    LLFR=llfr_with_new_VA_dynamic,
                    alpha=cp.alpha
                )

                impacts_calc_df = impact_calc.assess_impact(
                    asset_size=cp.asset_size,
                    asset_duration=cp.asset_duration,
                    liability_size=liability_size,
                    liability_duration=liability_duration,
                    zero_curve_SWWithVA=results_sw_withVA[['Tenors', 'Zero_CC']],
                    zero_curve_AltWithVA=results_alt_with_new_VA_dynamic[['Tenors', 'Zero_CC']],
                    zero_curve_assets=boot_df[['Tenors', 'Zero_CC']]
                )
                results_impact_density.append({
                    'Scenario': scenario["name"],
                    'Assets': cp.asset_size,
                    'Asset Duration': cp.asset_duration,
                    'Zero Rate Assets SW': impacts_calc_df['Zero Rate Assets SW'][0],
                    'Zero Rate Assets Alternative': impacts_calc_df['Zero Rate Assets Alternative'][0],
                    'Assets Reevaluated': impacts_calc_df['Assets Reevaluated'][0],
                    'Liabilities': liability_size,
                    'Liability Duration': liability_duration,
                    'Zero Rate Liabilities SW': impacts_calc_df['Zero Rate Liabilities SW'][0],
                    'Zero Rate Liabilities Alternative': impacts_calc_df['Zero Rate Liabilities Alternative'][0],
                    'Liabilities Reevaluated': impacts_calc_df['Liabilities Reevaluated'][0],
                    'Own Funds': impacts_calc_df['Own Funds'][0],
                    'Own Funds Reevaluated': impacts_calc_df['Own Funds Reevaluated'][0],
                    'Own Funds Impact': impacts_calc_df['Own Funds Impact'][0],
                    'Own Funds Impact rel.': impacts_calc_df['Own Funds Impact rel.'][0]
                })

    # Export and plot combined curve data
    curve_plotter_all = CurvePlotter(curves=scenario_curves_dict)
    curve_plotter_all.plot_curves_cs_combined(output_path='outputs/curves/plots/cs_combined/')
    curve_plotter_all.export_curve_data(output_path='outputs/curves/data/')

    # Plot and export impact data
    results_impacts_df = pd.DataFrame(results_impact_density)
    impact_plotter = ImpactPlotter(results_impact_df=results_impacts_df,
                                   asset_size=cp.asset_size,
                                   asset_duration=cp.asset_duration)
    impact_plotter.plot_liability_size_vs_impact_overlay(output_path='outputs/impacts/plots/')
    for scenario in cp.scenarios:
        impact_plotter.create_impact_density_plot(scenario=scenario["name"],
                                                  output_path='outputs/impacts/plots/')
        impact_plotter.export_impact_data(scenario=scenario["name"],
                                          output_path='outputs/impacts/data/')


if __name__ == "__main__":
    main()
