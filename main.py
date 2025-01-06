# main.py

import numpy as np
from data import MarketData
from params import CurveParameters
from curve import CurveLogic

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

    # 6) Compute LLFR
    llfr_noVA = cl.get_llfr(
        zero_rates=zero_boot, 
        dlt=dlt_array,
        weights=weight_array
    )
    print(f"\nComputed LLFR (no VA) = {llfr_noVA:.6f}")

    # 7) Alternative Extrapolation
    zero_extrap_noVA, fwd_extrap_noVA, disc_extrap_noVA = cl.alternative_extrapolation(
        zero_rates_cc=zero_boot,
        FSP=cp.FSP,
        UFR=cp.UFR,
        LLFR=llfr_noVA,
        alpha=cp.alpha,
        compounding='C'
    )

    # 8) Print results
    print("\n=== No VA (Extrapolated) ===")
    for i in range(max_tenor):
        print(f"Year {i+1} | Zero={zero_extrap_noVA[i]:.6f}"
              f" | Fwd={fwd_extrap_noVA[i]:.6f}"
              f" | Disc={disc_extrap_noVA[i]:.6f}")

if __name__ == "__main__":
    main()


    # # 7. Example: With VA
    # #    - We'll first shift the short-end rates by cp.VA_value,
    # #      up to cp.LLP (or some logic).
    # va_rate_array = rate_array.copy()
    # for i in range(1, max_tenor):
    #     if i <= cp.LLP:
    #         # add cp.VA_value (in decimal) => e.g. 1.35% => 0.0135
    #         va_rate_array[i] += (cp.VA_value * 100)  # if original is in % ?

    # zero_withVA, fwd_withVA, disc_withVA = cl.bootstrap_zero_to_zero_full(
    #     zero_rates_init=va_rate_array,
    #     dlt=dlt,
    #     compounding_in='A',
    #     cra=0.0,
    #     max_tenor=max_tenor,
    #     compounding_out='C'
    # )

    # LLFR_withVA = zero_withVA[cp.FSP]
    # zero_extrap_withVA, fwd_extrap_withVA, disc_extrap_withVA = cl.alternative_extrapolation(
    #     zero_rates_cc=zero_withVA,
    #     FSP=cp.FSP,
    #     UFR=cp.UFR,
    #     LLFR=LLFR_withVA,
    #     alpha=cp.alpha,
    #     compounding='C'
    # )
