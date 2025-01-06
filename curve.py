# curve.py

import numpy as np
from math import exp, log, isnan

class CurveLogic:
    """
    0-based indexing:
      - swap_rates[i] => rate for year (i+1)
      - if dlt[i] == 1 => year (i+1) is valid
    """

    def __init__(self):
        pass

    ###############################################################
    # NewtonRaphsonForwardSwap
    ###############################################################
    def newton_raphson_forward_swap(self, fw_guess, swap_rate, m, c,
                                    max_iter=500, tol=1e-15):
        fw = fw_guess
        for _ in range(max_iter):
            temp = (1 + fw)**(-m)
            fx = swap_rate * (1 - temp) / fw + temp - c
            if abs(fx) < tol:
                break
            temp2 = temp / (1 + fw)
            dfx = swap_rate * ((1 + (m + 1)*fw)*temp2 - 1)/(fw**2) - m*temp2
            fw = fw - fx/dfx
        return fw

    ###############################################################
    # bootstrap_swap_to_zero_full
    ###############################################################
    def bootstrap_swap_to_zero_full(self, swap_rates, dlt, coupon_freq,
                                    cra, max_tenor, compounding_out):
        """
        swap_rates: length = max_tenor, index i => year (i+1)
        dlt[i]    : 1 => valid for year (i+1)
        """
        forward = np.zeros(max_tenor)
        discount = np.ones(max_tenor)
        zero = np.zeros(max_tenor)

        # 1) gather valid_tenors, valid_swaps
        valid_tenors = []  # store indices
        valid_swaps = []   # store decimal
        for i in range(max_tenor):
            if dlt[i] == 1 and not isnan(swap_rates[i]):
                # convert from e.g. 2.0 => 0.02 => minus CRA => final decimal
                val_dec = swap_rates[i]/100.0 - cra/10000.0
                valid_tenors.append(i)  
                valid_swaps.append(val_dec)

        if len(valid_tenors) == 0:
            return zero, forward, discount

        # 2) first valid tenor
        first_idx = valid_tenors[0]  
        first_tenor = first_idx + 1  # actual year
        guess_fw = valid_swaps[0]/coupon_freq

        fwtemp = self.newton_raphson_forward_swap(
            fw_guess=guess_fw,
            swap_rate=(valid_swaps[0]/coupon_freq),
            m=coupon_freq * first_tenor,
            c=1.0
        )

        # fill from i=0..first_idx
        # i => (i+1)-th year
        for i in range(first_idx+1):
            # actual year = i+1
            year = i+1
            forward[i] = (1 + fwtemp)**coupon_freq - 1
            if i == 0:
                discount[i] = 1/(1 + forward[i])
            else:
                discount[i] = discount[i-1]/(1 + forward[i])
            zero[i] = (1.0/discount[i])**(1.0/year) - 1

        sumdiscount = (1 - (1+fwtemp)**(-coupon_freq * first_tenor)) / fwtemp
        last_idx = first_idx

        # 3) solve for subsequent
        for idx in range(1, len(valid_tenors)):
            curr_idx = valid_tenors[idx]
            curr_tenor = curr_idx + 1
            swap_val = valid_swaps[idx]/coupon_freq
            guess_fw = forward[last_idx]/coupon_freq
            m2 = (curr_tenor - (last_idx+1)) * coupon_freq

            # note: c=...
            fwtemp = self.newton_raphson_forward_swap(
                fw_guess=guess_fw,
                swap_rate=swap_val,
                m=m2,
                c=(1 - valid_swaps[idx]*sumdiscount)/discount[last_idx]
            )
            dtemp = (1 + fwtemp)**-1
            sumdiscount += discount[last_idx]*(1 - dtemp**(m2))/fwtemp

            # fill from last_idx+1..curr_idx
            for i in range(last_idx+1, curr_idx+1):
                year = i+1
                forward[i] = (1 + fwtemp)**coupon_freq - 1
                discount[i] = discount[i-1]/(1 + forward[i])
                zero[i] = (1.0/discount[i])**(1.0/year) - 1

            last_idx = curr_idx

        # 4) extrapolate
        if last_idx < (max_tenor - 1):
            for i in range(last_idx+1, max_tenor):
                year = i+1
                forward[i] = forward[last_idx]
                discount[i] = discount[i-1]/(1 + forward[i])
                zero[i] = (1.0/discount[i])**(1.0/year) - 1

        # 5) convert to continuous if needed
        if compounding_out == 'C':
            for i in range(max_tenor):
                fval = forward[i]
                zval = zero[i]
                forward[i] = log(1 + fval) if fval > -1 else 0.0
                zero[i] = log(1 + zval) if zval > -1 else 0.0

        return zero, forward, discount

    ###############################################################
    # bootstrap_zero_to_zero_full
    ###############################################################
    def bootstrap_zero_to_zero_full(self, zero_rates_init, dlt,
                                    compounding_in, cra,
                                    max_tenor, compounding_out):
        forward = np.zeros(max_tenor)
        discount = np.ones(max_tenor)
        zero = np.zeros(max_tenor)

        valid_tenors = []
        valid_vals = []
        for i in range(max_tenor):
            if dlt[i] == 1 and not isnan(zero_rates_init[i]):
                dec = zero_rates_init[i]/100.0 - cra/10000.0
                if compounding_in == 'A':
                    dec = log(1 + dec)
                valid_tenors.append(i)
                valid_vals.append(dec)

        if len(valid_tenors) == 0:
            return zero, forward, discount

        # 1) from i=0..first_idx
        first_idx = valid_tenors[0]
        first_val = valid_vals[0]
        for i in range(first_idx+1):
            year = i+1
            forward[i] = first_val
            zero[i] = forward[i]
            discount[i] = exp(-year * zero[i])

        # 2) piecewise
        for idx in range(1, len(valid_tenors)):
            left_i = valid_tenors[idx-1]
            right_i = valid_tenors[idx]
            left_val = valid_vals[idx-1]
            right_val = valid_vals[idx]
            left_year = left_i + 1
            right_year = right_i + 1
            m = right_year - left_year
            fwtemp = ((right_year * right_val) - (left_year * left_val)) / m

            for i in range(left_i+1, right_i+1):
                year = i+1
                forward[i] = fwtemp
                discount[i] = discount[i-1]*exp(-fwtemp)
                zero[i] = -log(discount[i]) / year

        last_idx = valid_tenors[-1]
        # 3) extrapolate
        if last_idx < (max_tenor-1):
            for i in range(last_idx+1, max_tenor):
                year = i+1
                forward[i] = forward[last_idx]
                discount[i] = discount[i-1]*exp(-forward[i])
                zero[i] = -log(discount[i]) / year

        # 4) convert to annual if needed
        if compounding_out == 'A':
            for i in range(max_tenor):
                fval = forward[i]
                zval = zero[i]
                forward[i] = exp(fval) - 1
                zero[i] = exp(zval) - 1

        return zero, forward, discount

    ###############################################################
    # alternative_extrapolation
    ###############################################################
    def alternative_extrapolation(self, zero_rates_cc, FSP, UFR, LLFR,
                                  alpha, compounding):
        """
        zero_rates_cc: length = max_tenor, index i => year (i+1)
        FSP: e.g. 30 => means array index = 29
        """
        max_range = len(zero_rates_cc)
        discount = np.zeros(max_range)
        forward = np.zeros(max_range)
        zero = np.zeros(max_range)

        # init year=1 => index=0
        discount[0] = exp(-zero_rates_cc[0])
        zero[0] = zero_rates_cc[0]
        forward[0] = zero[0]

        # fill up to FSP-1 => index (FSP-1)
        for i in range(1, FSP):
            year = i+1
            zero[i] = zero_rates_cc[i]
            forward[i] = year*zero[i] - (year-1)*zero[i-1]
            discount[i] = discount[i-1]*exp(-forward[i])

        ln_ufr = log(1 + UFR)
        # from index=FSP..(max_range-1)
        for i in range(FSP, max_range):
            year = i+1
            fwtemp = ln_ufr + (LLFR - ln_ufr)*(1 - exp(-alpha*(year - FSP))) / (alpha*(year - FSP))
            zero[i] = (FSP*zero[FSP-1] + (year - FSP)*fwtemp)/year
            discount[i] = exp(-year*zero[i])
            forward[i] = log(discount[i-1]/discount[i]) if i > 0 else zero[i]

        if compounding == 'A':
            for i in range(max_range):
                forward[i] = exp(forward[i]) - 1
                zero[i] = exp(zero[i]) - 1

        return zero, forward, discount

    ###############################################################
    # get_llfr
    ###############################################################
    def get_llfr(self, zero_rates, dlt, weights):
        """
        zero_rates, dlt, weights => length=max_tenor, index i => year i+1
        """
        pos_idx = []
        pos_wts = []
        sum_w = 0.0
        n = len(weights)
        for i in range(n):
            if weights[i] > 0:
                pos_idx.append(i)
                pos_wts.append(weights[i])
                sum_w += weights[i]
        if abs(sum_w - 1.0) > 1e-12:
            raise ValueError("LLFR weights do not sum to 1.")

        if not pos_idx:
            return 0.0

        fsp_idx = pos_idx[0]
        # year = fsp_idx + 1
        # find LLPbeforeFSP => largest j < fsp_idx with dlt[j]==1
        LLPbefore = -1
        for j in range(fsp_idx):
            if dlt[j] == 1:
                LLPbefore = j

        # Weighted sum of forward segments
        # forward( A->B ) = [ B*Z(B) - A*Z(A) ] / ( B - A )
        # but B= (fsp_idx + 1), A= (LLPbefore + 1)
        llfr_val = 0.0
        if LLPbefore >= 0 and fsp_idx > LLPbefore:
            B = fsp_idx + 1
            A = LLPbefore + 1
            seg_fw = ((B*zero_rates[fsp_idx]) - (A*zero_rates[LLPbefore]))/(B - A)
            llfr_val += pos_wts[0] * seg_fw

        # subsequent segments => (fsp -> t_i)
        for idx in range(1, len(pos_idx)):
            i2 = pos_idx[idx]
            B = i2 + 1
            A = fsp_idx + 1
            seg_fw = ((B*zero_rates[i2]) - (A*zero_rates[fsp_idx]))/(B - A)
            llfr_val += pos_wts[idx] * seg_fw

        return llfr_val

    def get_first_fw_llfr(self, zero_rates, dlt, weights):
        """
        first forward from LLPbeforeFSP -> fsp_idx
        """
        pos_idx = [i for i in range(len(weights)) if weights[i]>0]
        if not pos_idx:
            return 0.0
        fsp_idx = pos_idx[0]

        LLPbefore = -1
        for j in range(fsp_idx):
            if dlt[j] == 1:
                LLPbefore = j

        if LLPbefore < 0 or fsp_idx <= LLPbefore:
            return 0.0

        B = fsp_idx + 1
        A = LLPbefore + 1
        seg_fw = ((B*zero_rates[fsp_idx]) - (A*zero_rates[LLPbefore]))/(B - A)
        return seg_fw
