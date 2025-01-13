import numpy as np
from data import MarketData
from params import CurveParameters
from bootstrapping import Bootstrapping
from extrapolation.alternative import ExtrapolationAlt
from extrapolation.smithwilson import ExtrapolationSW
from va import VASpreadCalculator


def main():

    # 1) Load data from Excel
    md = MarketData(filepath="inputs.xlsx")
    md.open_workbook()
    df_alt = md.parse_sheet_to_df('rates_alt')
    df_sw = md.parse_sheet_to_df('rates_sw')
    md.close_workbook()

    # 2) Parameters
    cp = CurveParameters()

    # 3) Instantiate business logic classes
    bootstr = Bootstrapping()
    ext_alt = ExtrapolationAlt()
    ext_sw = ExtrapolationSW()
    va_calc = VASpreadCalculator() 

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
    if cp.Instrument == 'Swap' or cp.Instrument == 'Bond':
        zero_boot, fwd_boot, disc_boot = bootstr.bootstrap_swap_to_zero_full(
            swap_rates=rate_array * 100.0,  # decimal => percentage as required by method
            dlt=dlt_array,
            coupon_freq=cp.coupon_freq,
            cra=cp.CRA,
            max_tenor=cp.max_tenorofAlt,
            compounding_out=cp.compounding
        )
    else:
        zero_boot, fwd_boot, disc_boot = bootstr.bootstrap_zero_to_zero_full(
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
        zero_rates=zero_boot,
        dlt=dlt_array,
        weights=weight_array
    )

    # 7) Alternative Extrapolation (No VA)
    zero_extrap_noVA, fwd_extrap_noVA, disc_extrap_noVA = ext_alt.alternative_extrapolation(
        zero_rates=zero_boot,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_noVA,
        alpha=cp.alpha,
        compounding_out=cp.compounding
    )

    # 8) Include VA
    # add new VA to zero curves
    zero_boot_withVA = ext_alt.zero_boot_withVA(
        fwd_boot, cp.max_tenorofAlt, cp.FSP, va_calc.compute_va_spread())
    # For calculation with old VA value
    # zero_boot_withVA = ext_alt.zero_boot_withVA(
    #     fwd_boot, cp.max_tenorofAlt, cp.FSP, cp.VA_value)

    # Compute new LLFR with VA-laden zeros
    llfr_withVA = ext_alt.get_llfr(
        zero_rates=zero_boot_withVA,
        dlt=dlt_array,
        weights=weight_array
    )

    # Alternative Extrapolation with VA
    zero_extrap_withVA, fwd_extrap_withVA, disc_extrap_withVA = ext_alt.alternative_extrapolation(
        zero_rates=zero_boot_withVA,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_withVA,
        alpha=cp.alpha,
        compounding_out=cp.compounding
    )

    # Smith-Wilson Extrapolation
    results_SW = ext_sw.smith_wilson_extrapolation(
        Instrument=cp.Instrument, curve_data=df_sw, coupon_freq=cp.coupon_freq,
        CRA=cp.CRA, UFR=cp.UFR, alpha_min=cp.alpha_min_SW, CR=cp.CR_SW, CP=cp.CP_SW)

    # Smith-Wilson Extrapolation with VA
    df_sw_withVA = ext_sw.getInputwithVA(
        zero_rates_extrapolated_ac=results_SW['Zero AC'].copy(), LLP=cp.LLP_SW, VA_value=cp.VA_value, curve_data=df_sw)

    results_SW_withVA = ext_sw.smith_wilson_extrapolation(
        Instrument='Zero', curve_data=df_sw_withVA, coupon_freq=cp.coupon_freq,
        CRA=cp.CRA, UFR=cp.UFR, alpha_min=cp.alpha_min_SW, CR=cp.CR_SW, CP=cp.CP_SW)

    results_SW_withVA

if __name__ == "__main__":
    main()
