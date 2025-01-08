import math
import numpy as np
from math import exp, log

class ExtrapolationAlt:

    def __init__(self):
        pass

    ###############################################################
    # alternative_extrapolation
    ###############################################################
    def alternative_extrapolation(self, zero_rates, FSP, UFR, LLFR,
                                  alpha, compounding):
        max_range = len(zero_rates)
        discount = np.zeros(max_range)
        forward = np.zeros(max_range)
        zero = np.zeros(max_range)

        # init year=1 => index=0
        discount[0] = exp(-zero_rates[0])
        zero[0] = zero_rates[0]
        forward[0] = zero[0]

        # fill up to FSP-1 => index (FSP-1)
        for i in range(1, FSP):
            year = i+1
            zero[i] = zero_rates[i]
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

    def zero_boot_withVA(self, fwd_boot_withVA, max_tenor, FSP, VA_value):
        zero_boot_withVA = np.zeros(max_tenor)
        # compute zero curves with VA parallel shift up to FSP
        for i in range(FSP):
            fwd_boot_withVA[i] += math.log(1+VA_value/10000) # VA in bp
        for i in range(max_tenor):
            zero_boot_withVA[i] = np.mean(fwd_boot_withVA[:i+1])
        return zero_boot_withVA

