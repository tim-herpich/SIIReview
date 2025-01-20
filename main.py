import numpy as np
from data import MarketData
from params import CurveParameters
from bootstrapping import Bootstrapping
from extrapolation.alternative import ExtrapolationAlt
from extrapolation.smithwilson import ExtrapolationSW
from va import VASpreadCalculator
from impact import OnwFundsImpactAssessor


def main():

    # 1) Load data from Excel
    md = MarketData(filepath="inputs.xlsx")
    md.open_workbook()
    df_alt = md.parse_sheet_to_df('rates_alt')
    df_sw = md.parse_sheet_to_df('rates_sw')
    va_spreads_df = md.parse_sheet_to_df('spreads_va')
    va_spreads_df.set_index('Issuer', inplace=True)

    md.close_workbook()

    # 2) Parameters
    cp = CurveParameters()

    # 3) Instantiate business logic classes
    bootstr = Bootstrapping()
    ext_alt = ExtrapolationAlt()
    ext_sw = ExtrapolationSW()
    va_calc = VASpreadCalculator(
        va_spreads_df, cp.fi_asset_size, cp.liability_size, cp.pvbp_fi_assets, cp.pvbp_liabs)
    impact_calc = OnwFundsImpactAssessor()

    # 4) Prepare arrays for bootstrapping
    dlt_array = np.zeros(cp.max_tenorofAlt)
    rate_array = np.zeros(cp.max_tenorofAlt)
    weight_array = np.zeros(cp.max_tenorofAlt)

    for dlt_val, ten, wgt, rt_dec in zip(df_alt['DLT'], df_alt['Tenor'], df_alt['LLFR Weights'], df_alt['Input Rates']):
        t = int(round(ten))
        if 1 <= t <= cp.max_tenorofAlt:
            idx = t - 1
            dlt_array[idx] = dlt_val
            rate_array[idx] = rt_dec
            weight_array[idx] = wgt

    # 5) Bootstrap zero curves
    # Swap vs Zero input curves
    if cp.instrument == 'Swap' or cp.instrument == 'Bond':
        df_boot = bootstr.bootstrap_swap_to_zero_full(
            swap_rates=rate_array * 100.0,  # decimal => percentage as required by method
            dlt=dlt_array,
            coupon_freq=cp.coupon_freq,
            cra=cp.CRA,
            max_tenor=cp.max_tenorofAlt,
            compounding_out=cp.compounding
        )
    else:
        df_boot = bootstr.bootstrap_zero_to_zero_full(
            # decimal => percentage as required by method
            zero_rates_init=rate_array * 100.0,
            dlt=dlt_array,
            compounding_in='A',
            cra=cp.CRA,
            max_tenor=cp.max_tenorofAlt,
            compounding_out=cp.compounding
        )

    # 6) Compute LLFR (No VA)
    llfr_noVA = ext_alt.get_llfr(
        zero_rates=df_boot['Zero_CC'].values,
        dlt=dlt_array,
        weights=weight_array
    )

    # 7) Alternative Extrapolation (No VA)
    results_Alt = ext_alt.alternative_extrapolation(
        zero_rates=df_boot['Zero_CC'].values,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_noVA,
        alpha=cp.alpha
    )

    va_new = va_calc.compute_total_va()

    # 8) Include new VA
    # add new VA to zero curves
    zero_boot_withNewVA = ext_alt.zero_boot_withVA(
        df_boot['Forward_CC'].values, cp.max_tenorofAlt, cp.FSP, va_calc.compute_total_va())  # Alt extrapolation uses new VA method

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

    # 9) Smith-Wilson Extrapolation
    results_SW = ext_sw.smith_wilson_extrapolation(
        instrument=cp.instrument, curve_data=df_sw, coupon_freq=cp.coupon_freq,
        CRA=cp.CRA, UFR=cp.UFR, alpha_min=cp.alpha_min_SW, CR=cp.CR_SW, CP=cp.CP_SW)

    # Smith-Wilson Extrapolation with VA
    df_sw_withVA = ext_sw.getInputwithVA(
        zero_rates_extrapolated_ac=results_SW['Zero_AC'].copy(), LLP=cp.LLP_SW, VA_value=cp.VA_value, curve_data=df_sw)

    results_SW_withVA = ext_sw.smith_wilson_extrapolation(
        instrument='Zero', curve_data=df_sw_withVA, coupon_freq=cp.coupon_freq,
        CRA=cp.CRA, UFR=cp.UFR, alpha_min=cp.alpha_min_SW, CR=cp.CR_SW, CP=cp.CP_SW)

    # 10) Impact Assessment on Own Funds (Alternative Extrapolation + new VA vs. SW Extrapolation + VA)
    results_impact = impact_calc.assess_impact(asset_size=cp.asset_size, asset_duration=cp.asset_duration,
                                               liability_size=cp.liability_size, liability_duration=cp.liability_duration,
                                               discount_curve_SWWithVA=results_SW_withVA[[
                                                   'Tenors', 'Zero_CC']],
                                               discount_curve_AltWithVA=results_Alt_withNewVA[[
                                                   'Tenors', 'Zero_CC']],
                                               discount_curve_assets=df_boot[[
                                                   'Tenors', 'Zero_CC']]
                                               )

    results_impact


if __name__ == "__main__":
    main()
