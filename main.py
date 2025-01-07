# main.py

import numpy as np
from data import MarketData
from params import CurveParameters
from curve import CurveLogic
import math

def main():
    # 1) Load data from Excel
    md = MarketData(filepath="inputs.xlsx", sheet_name="rates", parse_excel=True)

    # 2) Parameters
    cp = CurveParameters()
    max_tenor = cp.max_tenor

    # 3) Instantiate curve logic
    cl = CurveLogic()

    # 4) Prepare arrays for bootstrapping
    dlt_array = np.zeros(max_tenor)
    rate_array = np.zeros(max_tenor)
    weight_array = np.zeros(max_tenor)

    for dlt_val, ten, wgt, rt_dec in zip(md.dlt, md.tenors, md.llfr_weights, md.input_rates):
        t = int(round(ten))
        if 1 <= t <= max_tenor:
            idx = t - 1
            dlt_array[idx] = dlt_val
            rate_array[idx] = rt_dec
            weight_array[idx] = wgt

    # 5) Bootstrap zero curves
    # Swap vs Zero input curves
    if cp.input == 'Swap':
        print("Detected input = 'Swap' => bootstrap_swap_to_zero_full.")
        zero_boot, fwd_boot, disc_boot = cl.bootstrap_swap_to_zero_full(
            swap_rates=rate_array * 100.0,  # decimal => percentage as required by method
            dlt=dlt_array,
            coupon_freq=1,
            cra=cp.CRA,
            max_tenor=max_tenor,
            compounding_out=cp.compounding
        )
    else:
        print("Detected input = 'Zero' => bootstrap_zero_to_zero_full.")
        zero_boot, fwd_boot, disc_boot = cl.bootstrap_zero_to_zero_full(
            zero_rates_init=rate_array * 100.0, # decimal => percentage as required by method
            dlt=dlt_array,
            compounding_in='A',
            cra=cp.CRA,
            max_tenor=max_tenor,
            compounding_out=cp.compounding
        )

    # 6) Compute LLFR (No VA)
    llfr_noVA = cl.get_llfr(
        zero_rates=zero_boot, 
        dlt=dlt_array,
        weights=weight_array
    )
    print(f"\nComputed LLFR (no VA) = {llfr_noVA:.6f}")

    # 7) Alternative Extrapolation (No VA)
    zero_extrap_noVA, fwd_extrap_noVA, disc_extrap_noVA = cl.alternative_extrapolation(
        zero_rates_cc=zero_boot,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_noVA,
        alpha=cp.alpha,
        compounding='C'
    )

    # 8) Alternative Extrapolation (VA)
    zero_boot_withVA = np.zeros(max_tenor)
    fwd_boot_withVA = fwd_boot.copy()
    # compute zero curves with VA parallel shift up to FSP
    for i in range(cp.FSP):
        fwd_boot_withVA[i] += math.log(1+cp.VA_value/10000)
    for i in range(max_tenor):
        zero_boot_withVA[i] = np.mean(fwd_boot_withVA[:i+1])

    # Compute new LLFR with VA-laden zeros
    llfr_withVA = cl.get_llfr(
        zero_rates=zero_boot_withVA,
        dlt=dlt_array,
        weights=weight_array
    )
    print(f"\n[With VA] Computed LLFR = {llfr_withVA:.6f}")

    # Extrapolated curves (With VA)
    zero_extrap_withVA, fwd_extrap_withVA, disc_extrap_withVA = cl.alternative_extrapolation(
        zero_rates_cc=zero_boot_withVA,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_withVA,
        alpha=cp.alpha,
        compounding='C'
    )

    # 9) Print results
    print("\n=== No VA (Extrapolated) ===")
    for i in range(max_tenor):
        print(f"Year {i+1} | Zero={zero_extrap_noVA[i]:.6f}"
              f" | Fwd={fwd_extrap_noVA[i]:.6f}"
              f" | Disc={disc_extrap_noVA[i]:.6f}")

    print("\n=== With VA (Extrapolated) ===")
    for i in range(max_tenor):
        year = i+1
        print(f"Year {year:2d} | Zero={zero_extrap_withVA[i]:.6f} "
              f"| Fwd={fwd_extrap_withVA[i]:.6f} "
              f"| Disc={disc_extrap_withVA[i]:.6f}")


if __name__ == "__main__":
    main()
