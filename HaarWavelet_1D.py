# Aliakbar Zarkoob, AKA "XIV"
# Gmail: XIV.Aliakbar.Zarkoob@gmail.com
# Telegram: @XIVAliakbar

import numpy as np


def HaarWavelet_1D(Signal, Max_Dec_Level):
    
    n = Signal.shape[0]
    if np.mod(n, 2) != 0:
        raise ValueError('The length of signal must be a power of 2!')
    
    if Max_Dec_Level >= np.log2(n):
        raise ValueError(f'The maximum decomposition level for the given data is {int(np.log2(n)-1)}!')
    
    V = {}; W = {}; s = {}; d = {}; S = {}; D = {}
    for i in range(Max_Dec_Level):
        level = i+1
        coeff_level = 1/2**(level/2)
        if level == 1:
            coeff_pattern_s = np.array([[1], [1]])
            coeff_pattern_d = np.array([[1], [-1]])
        else:
            coeff_pattern_s = np.vstack((pre_coeff_pattern_s, pre_coeff_pattern_s))
            coeff_pattern_d = np.vstack((pre_coeff_pattern_d, -pre_coeff_pattern_d))
        pre_coeff_pattern_s = coeff_pattern_s
        pre_coeff_pattern_d = coeff_pattern_d
        coeff_s = coeff_pattern_s*coeff_level
        coeff_d = coeff_pattern_d*coeff_level
        
        V[str(level)] = np.kron(np.eye(n//(2**level)), coeff_s)
        W[str(level)] = np.kron(np.eye(n//(2**level)), coeff_d)
        s[str(level)] = Signal.T@V[str(level)]
        d[str(level)] = Signal.T@W[str(level)]
        S[str(level)] = np.sum(s[str(level)]*V[str(level)], 1)
        D[str(level)] = np.sum(d[str(level)]*W[str(level)], 1)
        # S[str(level)] = np.zeros((n, 1))
        # D[str(level)] = np.zeros((n, 1)) 
        # for j in range(s[str(level)].shape[1]):
        #     S[str(level)] = S[str(level)] + (s[str(level)][0, j]*V[str(level)][:,j]).reshape(-1, 1)
        #     D[str(level)] = D[str(level)] + (d[str(level)][0, j]*W[str(level)][:,j]).reshape(-1, 1)
        
    return S, D, s, d

        
