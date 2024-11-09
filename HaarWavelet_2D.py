# Aliakbar Zarkoob, AKA "XIV"
# Gmail: XIV.Aliakbar.Zarkoob@gmail.com
# Telegram: @XIVAliakbar

import numpy as np


def HaarWavelet_2D(Image, Max_Dec_Level):
        
    m = Image.shape[0]
    n = Image.shape[1]

    if np.mod(n, 2) != 0 or np.mod(m, 2) != 0:
        raise ValueError('The size of image must be a power of 2!')
    
    if Max_Dec_Level >= np.log2(min(n,m)):
        raise ValueError(f'The maximum decomposition level for the given image is {int(np.log2(min(n,m))-1)}!')

    print('\n Processing Wavelet Decomposition ...')
    Vn = {}; Wn = {}; Vm = {}; Wm = {}; SS = {}; SD = {}; DS = {}; DD = {}
    for level in range(Max_Dec_Level):
        level += 1
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

        Vn[str(level)] = np.kron(np.eye(n//(2**level)), coeff_s)
        Wn[str(level)] = np.kron(np.eye(n//(2**level)), coeff_d)
        Vm[str(level)] = np.kron(np.eye(m//(2**level)), coeff_s)
        Wm[str(level)] = np.kron(np.eye(m//(2**level)), coeff_d)

        for i in range (m//(2**level)):
            for j in range(n//(2**level)):
                
                tmp_ss = np.outer(Vm[str(level)][:, i], Vn[str(level)][:, j])
                tmp_sd = np.outer(Vm[str(level)][:, i], Wn[str(level)][:, j])
                tmp_ds = np.outer(Wm[str(level)][:, i], Vn[str(level)][:, j])
                tmp_dd = np.outer(Wm[str(level)][:, i], Wn[str(level)][:, j])
                                                    
                ss = np.sum(Image*tmp_ss)
                sd = np.sum(Image*tmp_sd)
                ds = np.sum(Image*tmp_ds)                
                dd = np.sum(Image*tmp_dd)
                
                if i == 0 and j == 0:
                    SS[str(level)] = ss*tmp_ss
                    SD[str(level)] = sd*tmp_sd
                    DS[str(level)] = ds*tmp_ds
                    DD[str(level)] = dd*tmp_dd
                else:
                    SS[str(level)] = SS[str(level)] + ss*tmp_ss
                    SD[str(level)] = SD[str(level)] + sd*tmp_sd
                    DS[str(level)] = DS[str(level)] + ds*tmp_ds
                    DD[str(level)] = DD[str(level)] + dd*tmp_dd
                    
        print(f'Decomposition Level {level} Completed!')

    return SS, SD, DS, DD