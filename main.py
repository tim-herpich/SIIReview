import numpy as np
from data import MarketData
from params import CurveParameters
from bootstrapping import Bootstrapping
from extrapolation.alternative import ExtrapolationAlt
from extrapolation.smithwilson import ExtrapolationSW
import math

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

    # 4) Prepare arrays for bootstrapping
    dlt_array = np.zeros(cp.max_tenor)
    rate_array = np.zeros(cp.max_tenor)
    weight_array = np.zeros(cp.max_tenor)

    for dlt_val, ten, wgt, rt_dec in zip(df_alt['DLT'], df_alt['Tenor'], df_alt['LLFR Weights'], df_alt['Input Rates']):
        t = int(round(ten))
        if 1 <= t <= cp.max_tenor:
            idx = t - 1
            dlt_array[idx] = dlt_val
            rate_array[idx] = rt_dec
            weight_array[idx] = wgt

    # 5) Bootstrap zero curves
    # Swap vs Zero input curves
    if cp.input == 'Swap':
        print("Detected input = 'Swap' => bootstrap_swap_to_zero_full.")
        zero_boot, fwd_boot, disc_boot = bootstr.bootstrap_swap_to_zero_full(
            swap_rates=rate_array * 100.0,  # decimal => percentage as required by method
            dlt=dlt_array,
            coupon_freq=1,
            cra=cp.CRA,
            max_tenor=cp.max_tenor,
            compounding_out=cp.compounding
        )
    else:
        print("Detected input = 'Zero' => bootstrap_zero_to_zero_full.")
        zero_boot, fwd_boot, disc_boot = bootstr.bootstrap_zero_to_zero_full(
            zero_rates_init=rate_array * 100.0, # decimal => percentage as required by method
            dlt=dlt_array,
            compounding_in='A',
            cra=cp.CRA,
            max_tenor=cp.max_tenor,
            compounding_out=cp.compounding
        )

    # 6) Compute LLFR (No VA)
    llfr_noVA = ext_alt.get_llfr(
        zero_rates=zero_boot, 
        dlt=dlt_array,
        weights=weight_array
    )
    print(f"\nComputed LLFR (no VA) = {llfr_noVA:.6f}")

    # 7) Alternative Extrapolation (No VA)
    zero_extrap_noVA, fwd_extrap_noVA, disc_extrap_noVA = ext_alt.alternative_extrapolation(
        zero_rates=zero_boot,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_noVA,
        alpha=cp.alpha,
        compounding='C'
    )

    # 8) Include VA
    # add VA to zero curves
    zero_boot_withVA = ext_alt.zero_boot_withVA(fwd_boot, cp.max_tenor, cp.FSP, cp.VA_value)

    # Compute new LLFR with VA-laden zeros
    llfr_withVA = ext_alt.get_llfr(
        zero_rates=zero_boot_withVA,
        dlt=dlt_array,
        weights=weight_array
    )
    print(f"\n[With VA] Computed LLFR = {llfr_withVA:.6f}")

    # Extrapolated curves with VA
    zero_extrap_withVA, fwd_extrap_withVA, disc_extrap_withVA = ext_alt.alternative_extrapolation(
        zero_rates=zero_boot_withVA,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_withVA,
        alpha=cp.alpha,
        compounding='C'
    )

    # 9) Print results
    print("\n=== No VA (Extrapolated) ===")
    for i in range(cp.max_tenor):
        print(f"Year {i+1} | Zero={zero_extrap_noVA[i]:.6f}"
              f" | Fwd={fwd_extrap_noVA[i]:.6f}"
              f" | Disc={disc_extrap_noVA[i]:.6f}")

    print("\n=== With VA (Extrapolated) ===")
    for i in range(cp.max_tenor):
        year = i+1
        print(f"Year {year:2d} | Zero={zero_extrap_withVA[i]:.6f} "
              f"| Fwd={fwd_extrap_withVA[i]:.6f} "
              f"| Disc={disc_extrap_withVA[i]:.6f}")


    ######################################################## SW ########################################################

    # Instrument = "Swap"  # Can be "Zero", "Bond", or "Swap"
    
    # DataIn = pd.read_excel(filepath="inputs.xlsx", sheet_name='rates_sw')
    
    # nrofcoup = 2
    # CRA = 50.0       # Credit Risk Adjustment in basis points
    # UFRac = 0.03     # Ultimate Forward Rate
    # alfamin = 0.1
    # Tau = 20.0       # Tau in basis points
    # T2 = 10          # Convergence Maturity

    # # Initialize the class
    # sw = SmithWilson()

    # # Run the Smith-Wilson Brute Force method
    # results = sw.smith_wilson_brute_force(Instrument, DataIn, nrofcoup, CRA, UFRac, alfamin, Tau, T2)


if __name__ == "__main__":
    main()
