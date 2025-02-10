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
    scenario_impact_dict = {}

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
            # convert bps to decimal
            va_spreads_shifted += scenario['csshift'] / 10000

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
            max_tenor=cp.max_tenorofAlt
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

    # Impact analysis: Compute the PV of a unit ZCB for the different discount curves with VA
        impact_results = {
            "Maturity": [],
            "PV Alternative Extrapolation": [],
            "PV Smith-Wilson Extrapolation": [],
            "PV (Alt - SW)": []
        }
        for maturity in range(cp.LLP_SW, cp.CP_SW+1, 5):
            pv_alt = impact_calc.compute_zcb_pv(
                results_alt_with_new_VA, maturity, cp.LLP_SW)
            pv_sw = impact_calc.compute_zcb_pv(
                results_sw_withVA, maturity, cp.LLP_SW)
            impact_results["Maturity"].append(maturity)
            impact_results["PV Alternative Extrapolation"].append(pv_alt)
            impact_results["PV Smith-Wilson Extrapolation"].append(pv_sw)
            impact_results["PV (Alt - SW)"].append(pv_alt - pv_sw)
        impact_df = pd.DataFrame(impact_results)
        scenario_impact_dict[scenario["name"]] = impact_df

    # Export and plot combined curve data
    print(f"Plot and export curves...")
    curve_plotter = CurvePlotter(curves=scenario_curves_dict)
    curve_plotter.plot_curves_cs_combined(llp=cp.LLP_SW,
                                          output_path='outputs/curves/plots/cs_combined/')
    curve_plotter.export_curve_data(output_path='outputs/curves/data/')
    curve_plotter.plot_curves(
        llp=cp.LLP_SW, output_path='outputs/curves/plots/')
    curve_plotter.compute_curve_differences()
    curve_plotter.export_curve_differences_data(output_path='outputs/curves/')
    curve_plotter.plot_curve_differences(
        llp=cp.LLP_SW, output_path='outputs/curves/')

    # Plot and export impact data
    print(f"Plot and export impacts...")
    impact_plotter = ImpactPlotter(impact_data=scenario_impact_dict)
    impact_plotter.plot_impact_barchart(output_path='outputs/impacts/plots/')
    impact_plotter.export_impact_data(output_path='outputs/impacts/data/')


if __name__ == "__main__":
    main()
