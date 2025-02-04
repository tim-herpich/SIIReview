import numpy as np
from data import MarketData
from params import CurveParameters
from bootstrapping import Bootstrapping
from extrapolation.alternative import ExtrapolationAlt
from extrapolation.smithwilson import ExtrapolationSW
from va import VASpreadCalculator
from impact import OnwFundsImpactAssessor
from plots.curveplotter import CurvePlotter
from plots.impactplotter import ImpactPlotter
import pandas as pd


def main():
    # Load data from Excel
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
    impact_calc = OnwFundsImpactAssessor()

    # Parameters
    cp = CurveParameters()

    # Dictionary to store curves for different scenarios
    scenario_curves_dict = {}

    # Iterate over scenarios
    for scenario in cp.scenarios:
        print(f"Processing scenario: {scenario['name']}")

        # Apply global shift to curves for high interest rates or spreads
        df_alt_shifted = df_alt.copy()
        df_sw_shifted = df_sw.copy()
        va_spreads_shifted = va_spreads_df.copy()

        if 'high_interest' in scenario['name'] or 'low_interest' in scenario['name']:
            df_alt_shifted['Input Rates'] += scenario['irshift'] / \
                10000  # Convert bps to decimal
            df_sw_shifted['Input Rates'] += scenario['irshift'] / \
                10000  # Convert bps to decimal

        if 'high_spreads' in scenario['name']:
            # Convert bps to decimal
            va_spreads_shifted += scenario['csshift'] / 10000

        # set the legacy va spread value
        cp.VA_value = scenario['vaspread']

        # Prepare arrays for bootstrapping
        dlt_array = np.zeros(cp.max_tenorofAlt)
        rate_array = np.zeros(cp.max_tenorofAlt)
        weight_array = np.zeros(cp.max_tenorofAlt)

        for dlt_val, ten, wgt, rt_dec in zip(
                df_alt_shifted['DLT'], df_alt_shifted['Tenor'], df_alt_shifted['LLFR Weights'], df_alt_shifted['Input Rates']):
            t = int(round(ten))
            if 1 <= t <= cp.max_tenorofAlt:
                idx = t - 1
                dlt_array[idx] = dlt_val
                rate_array[idx] = rt_dec
                weight_array[idx] = wgt

        # Bootstrap input curves
        df_boot = bootstr.bootstrap_to_zero_full(instrument=cp.instrument,
                                                 rates=rate_array * 100.0,
                                                 dlt=dlt_array,
                                                 coupon_freq=cp.coupon_freq,
                                                 compounding_in=cp.compounding_in,
                                                 cra=cp.CRA,
                                                 max_tenor=cp.max_tenorofAlt,
                                                 compounding_out=cp.compounding_out)

        # Compute LLFR (No VA)
        llfr_noVA = ext_alt.get_llfr(
            zero_rates=df_boot['Zero_CC'].values,
            dlt=dlt_array,
            weights=weight_array
        )

        # Alternative Extrapolation (No VA)
        results_Alt = ext_alt.alternative_extrapolation(
            zero_rates=df_boot['Zero_CC'].values,
            FSP=cp.FSP,
            UFR=cp.UFR,
            LLFR=llfr_noVA,
            alpha=cp.alpha
        )

        # Include new VA
        va_calc = VASpreadCalculator(
            va_spreads_shifted, cp.fi_asset_size, cp.liability_size, cp.pvbp_fi_assets, cp.pvbp_liabs)
        va_new = va_calc.compute_total_va()
        # print(va_new)

        zero_boot_withNewVA = ext_alt.zero_boot_withVA(
            df_boot['Forward_CC'].copy().values, cp.max_tenorofAlt, cp.FSP, va_new)

        # Compute new LLFR with VA-laden zeros
        llfr_withNewVA = ext_alt.get_llfr(
            zero_rates=zero_boot_withNewVA,
            dlt=dlt_array,
            weights=weight_array
        )

        # Alternative Extrapolation with new VA
        results_Alt_withNewVA = ext_alt.alternative_extrapolation(
            zero_rates=zero_boot_withNewVA,
            FSP=cp.FSP,
            UFR=cp.UFR,
            LLFR=llfr_withNewVA,
            alpha=cp.alpha
        )

        # Smith-Wilson Extrapolation
        results_SW = ext_sw.smith_wilson_extrapolation(
            instrument=cp.instrument, curve_data=df_sw_shifted, coupon_freq=cp.coupon_freq,
            CRA=cp.CRA, UFR=cp.UFR, alpha_min=cp.alpha_min_SW, CR=cp.CR_SW, CP=cp.CP_SW)

        # Smith-Wilson Extrapolation with VA
        df_sw_withVA = ext_sw.getInputwithVA(
            zero_rates_extrapolated_ac=results_SW['Zero_AC'].copy(), LLP=cp.LLP_SW, VA_value=cp.VA_value, curve_data=df_sw_shifted)

        results_SW_withVA = ext_sw.smith_wilson_extrapolation(
            instrument='Zero', curve_data=df_sw_withVA, coupon_freq=cp.coupon_freq,
            CRA=0.0, UFR=cp.UFR, alpha_min=cp.alpha_min_SW, CR=cp.CR_SW, CP=cp.CP_SW)

       # Store curves in the dictionary
        scenario_curves_dict[scenario["name"]] = {
            'Alternative Extrapolation with VA': results_Alt_withNewVA,
            'Alternative Extrapolation': results_Alt,
            'Smith-Wilson Extrapolation with VA': results_SW_withVA[:-1],
            'Smith-Wilson Extrapolation': results_SW[:-1]
        }

        # Generate plots for this scenario
        pairs = [
            ('Alternative Extrapolation with VA',
             'Smith-Wilson Extrapolation with VA'),
            ('Alternative Extrapolation', 'Smith-Wilson Extrapolation')
        ]
        curve_plotter = CurvePlotter(scenario_curves_dict[scenario["name"]])
        curve_plotter.plot_curves(
            pairs, scenario=scenario["name"], output_path='outputs/curves/plots/')

        # Impact Assessment | Sensitivity Analysis w.r.t OF Sizes and Duration Gaps
        results_impact = []
        for liability_size in np.arange(0.75*cp.asset_size, cp.asset_size, cp.asset_size/40.0):
            for liability_duration in np.arange(cp.asset_duration, 1.5 * cp.asset_duration, cp.asset_duration/20.0):

                # Recompute new VA
                va_calc = VASpreadCalculator(
                    va_spreads_shifted, cp.fi_asset_size, liability_size=liability_size, pvbp_fi_assets=cp.pvbp_fi_assets, pvbp_liabs=0.1*liability_duration)
                va_new = va_calc.compute_total_va()

                zero_boot_withNewVA = ext_alt.zero_boot_withVA(
                    df_boot['Forward_CC'].copy().values, cp.max_tenorofAlt, cp.FSP, va_new)

                # Recompute LLFR with VA-laden zeros
                llfr_withNewVA = ext_alt.get_llfr(
                    zero_rates=zero_boot_withNewVA,
                    dlt=dlt_array,
                    weights=weight_array
                )

                # Recompute Alternative Extrapolation with recomputed VA
                results_Alt_withNewVA = ext_alt.alternative_extrapolation(
                    zero_rates=zero_boot_withNewVA,
                    FSP=cp.FSP,
                    UFR=cp.UFR,
                    LLFR=llfr_withNewVA,
                    alpha=cp.alpha
                )

                impacts_calc_df = impact_calc.assess_impact(
                    asset_size=cp.asset_size, asset_duration=cp.asset_duration, liability_size=liability_size, liability_duration=liability_duration, zero_curve_SWWithVA=results_SW_withVA[[
                        'Tenors', 'Zero_CC']], zero_curve_AltWithVA=results_Alt_withNewVA[['Tenors', 'Zero_CC']], zero_curve_assets=df_boot[['Tenors', 'Zero_CC']]
                )
                results_impact.append(
                    [cp.asset_size, impacts_calc_df['Zero Rate Assets SW'][0], impacts_calc_df['Zero Rate Assets Alternative'][0], impacts_calc_df['Assets Reevaluated'][0], cp.asset_duration,
                     liability_size, impacts_calc_df['Zero Rate Liabilities SW'][0], impacts_calc_df[
                         'Zero Rate Liabilities Alternative'][0], impacts_calc_df['Liabilities Reevaluated'][0], liability_duration,
                     impacts_calc_df['Own Funds'][0], impacts_calc_df['Own Funds Reevaluated'][0],
                     impacts_calc_df['Own Funds Impact'][0], impacts_calc_df['Own Funds Impact rel.'][0]])

        results_impacts_df = pd.DataFrame(results_impact, columns=[
            'Assets', 'Zero Rate Assets SW', 'Zero Rate Assets Alternative', 'Assets Reevaluated', 'Asset Duration', 'Liabilities', 'Zero Rate Liabilities SW', 'Zero Rate Liabilities Alternative',
            'Liabilities Reevaluated', 'Liability Duration', 'Own Funds', 'Own Funds Reevaluated', 'Own Funds Impact', 'Own Funds Impact rel.'])
        plotter = ImpactPlotter(
            results_impacts_df, cp.asset_size, cp.asset_duration)
        plotter.create_impact_density_plot(
            scenario=scenario["name"], output_path='outputs/impacts/plots/')
        plotter.export_impact_data(
            scenario=scenario["name"], output_path='outputs/impacts/data')
        results_impact

    # Instantiate CurvePlotter with all scenario curves for combined plotting
    curve_plotter_all = CurvePlotter(scenario_curves_dict)
    # Generate plots comparing different CS scenarios at the same interest rate level
    curve_plotter_all.plot_curves_cs_combined(
        output_path='outputs/curves/plots/cs_combined/')

    # Export curve data
    curve_plotter_all.export_curve_data(output_path='outputs/curves/data/')
    results_impact


if __name__ == "__main__":
    main()
