# Aliakbar Zarkoob, AKA "XIV"
# Gmail: XIV.Aliakbar.Zarkoob@gmail.com
# Telegram: @XIVAliakbar

import numpy as np


def Wavelet_2D(Image, Type, Max_Dec_Level):
    
    if Type.lower() == 'haar':
        h = np.array([1/np.sqrt(2), 1/np.sqrt(2)]).reshape(-1, 1)
        g = np.array([1/np.sqrt(2), -1/np.sqrt(2)]).reshape(-1, 1)
        TYPE = 'Haar'
    elif Type.lower() == 'db4':
        h = np.array([(1+np.sqrt(3))/(4*np.sqrt(2)), (3+np.sqrt(3))/(4*np.sqrt(2)), (3-np.sqrt(3))/(4*np.sqrt(2)), (1-np.sqrt(3))/(4*np.sqrt(2))]).reshape(-1, 1)
        g = np.array([h[3], -h[2], h[1], -h[0]]).reshape(-1, 1)
        TYPE = 'Daubechies4'
    elif Type.lower() == 'db6':
        h = np.array([0.3326705529500826, 0.8068915093110928, 0.4598775021184915, -0.1350110200102546, -0.0854412738822415, 0.0352262918857095]).reshape(-1, 1)
        g = np.array([0.0352262918857095, 0.0854412738822415, -0.1350110200102546, -0.4598775021184915, 0.8068915093110928, -0.3326705529500826]).reshape(-1, 1)
        TYPE = 'Daubechies6'
    elif Type.lower() == 'mexicanhat':
        h = np.array([(1-np.sqrt(7))/(16*np.sqrt(2)), (5+np.sqrt(7))/(16*np.sqrt(2)), (14+2*np.sqrt(7))/(16*np.sqrt(2)), (14-2*np.sqrt(7))/(16*np.sqrt(2)), (1-np.sqrt(7))/(16*np.sqrt(2)), (-3+np.sqrt(7))/(16*np.sqrt(2))]).reshape(-1, 1)
        g = np.array([h[5], -h[4], h[3], -h[2], h[1], -h[0]]).reshape(-1, 1)
        TYPE = 'Mexican Hat'
    elif Type.lower() == 'sym3':
        h = np.array([0.3326705529500826, 0.8068915093133388, 0.4598775021184915, -0.1350110200102546, -0.0854412738820267, 0.0352262918857095]).reshape(-1, 1)
        g = np.array([-0.0352262918857095, -0.0854412738820267, 0.1350110200102546, 0.4598775021184915, -0.8068915093133388, 0.3326705529500826]).reshape(-1, 1)
        TYPE = 'Symlet3'
    else:
        raise ValueError('The specified wavelet type is not supported in the current version of this function!')

    if np.mod(Image.shape[0], 2) != 0 or np.mod(Image.shape[1], 2) != 0:
        raise ValueError('The size of image must be a power of 2!')
    
    if Max_Dec_Level >= np.log2(min(Image.shape[:2])):
        raise ValueError(f'The maximum decomposition level for the given image is {int(np.log2(min(Image.shape[:2]))-1)}!')


    def CreateBasis(BasisLen, h, g):
        CoeffNum = h.shape[0]
        V = np.zeros((BasisLen, BasisLen//2))
        W = np.zeros((BasisLen, BasisLen//2))
        for i in range(BasisLen//2):
            V_tmp = np.zeros((BasisLen, 1))
            W_tmp = np.zeros((BasisLen, 1))
            if i*2+CoeffNum > BasisLen:
                diff = i*2+CoeffNum - BasisLen
                V_tmp[:diff] = h[-diff:]
                V_tmp[-diff:] = h[:diff]
                W_tmp[:diff] = g[-diff:]
                W_tmp[-diff:] = g[:diff]
            else:
                V_tmp[i*2:i*2+CoeffNum] = h
                W_tmp[i*2:i*2+CoeffNum] = g
            V[:, i] = V_tmp.flatten()
            W[:, i] = W_tmp.flatten()
        return V, W


    print(f'\nProcessing {TYPE} Wavelet Decomposition ...')
    SS = []; SD = []; DS = []; DD = []
    f = Image
    for level in range(Max_Dec_Level):
        
        level += 1
        m, n = f.shape[:2]
        Vn, Wn = CreateBasis(n, h, g)
        Vm, Wm = CreateBasis(m, h, g)
        ss = np.zeros((m//2, n//2, 3))
        sd = np.zeros((m//2, n//2, 3))
        ds = np.zeros((m//2, n//2, 3))
        dd = np.zeros((m//2, n//2, 3))
        for i in range(m//2):
            for j in range(n//2):
                tmp_ss = np.outer(Vm[:, i], Vn[:, j])
                tmp_sd = np.outer(Vm[:, i], Wn[:, j])
                tmp_ds = np.outer(Wm[:, i], Vn[:, j])
                tmp_dd = np.outer(Wm[:, i], Wn[:, j])
                
                ss[i, j, 0] = np.sum(f[:, :, 0]*tmp_ss); ss[i, j, 1] = np.sum(f[:, :, 1]*tmp_ss); ss[i, j, 2] = np.sum(f[:, :, 2]*tmp_ss)
                sd[i, j, 0] = np.sum(f[:, :, 0]*tmp_sd); sd[i, j, 1] = np.sum(f[:, :, 1]*tmp_sd); sd[i, j, 2] = np.sum(f[:, :, 2]*tmp_sd)
                ds[i, j, 0] = np.sum(f[:, :, 0]*tmp_ds); ds[i, j, 1] = np.sum(f[:, :, 1]*tmp_ds); ds[i, j, 2] = np.sum(f[:, :, 2]*tmp_ds)                
                dd[i, j, 0] = np.sum(f[:, :, 0]*tmp_dd); dd[i, j, 1] = np.sum(f[:, :, 1]*tmp_dd); dd[i, j, 2] = np.sum(f[:, :, 2]*tmp_dd)
                
        f = ss
        for k in range(level):
            m = Image.shape[0]//(2**(level-k-1))
            n = Image.shape[1]//(2**(level-k-1))
            Vn, Wn = CreateBasis(n, h, g)
            Vm, Wm = CreateBasis(m, h, g)
            for i in range(m//2):
                for j in range(n//2):
                    tmp_ss = np.outer(Vm[:, i], Vn[:, j])
                    tmp_sd = np.outer(Vm[:, i], Wn[:, j])
                    tmp_ds = np.outer(Wm[:, i], Vn[:, j])
                    tmp_dd = np.outer(Wm[:, i], Wn[:, j])
                    if k == 0:
                        if i == 0 and j == 0:
                            SS0 = ss[i, j, 0]*tmp_ss; SS1 = ss[i, j, 1]*tmp_ss; SS2 = ss[i, j, 2]*tmp_ss; 
                            SD0 = sd[i, j, 0]*tmp_sd; SD1 = sd[i, j, 1]*tmp_sd; SD2 = sd[i, j, 2]*tmp_sd; 
                            DS0 = ds[i, j, 0]*tmp_ds; DS1 = ds[i, j, 1]*tmp_ds; DS2 = ds[i, j, 2]*tmp_ds; 
                            DD0 = dd[i, j, 0]*tmp_dd; DD1 = dd[i, j, 1]*tmp_dd; DD2 = dd[i, j, 2]*tmp_dd; 
                        else:
                            SS0 = SS0 + ss[i, j, 0]*tmp_ss; SS1 = SS1 + ss[i, j, 1]*tmp_ss; SS2 = SS2 + ss[i, j, 2]*tmp_ss
                            SD0 = SD0 + sd[i, j, 0]*tmp_sd; SD1 = SD1 + sd[i, j, 1]*tmp_sd; SD2 = SD2 + sd[i, j, 2]*tmp_sd
                            DS0 = DS0 + ds[i, j, 0]*tmp_ds; DS1 = DS1 + ds[i, j, 1]*tmp_ds; DS2 = DS2 + ds[i, j, 2]*tmp_ds
                            DD0 = DD0 + dd[i, j, 0]*tmp_dd; DD1 = DD1 + dd[i, j, 1]*tmp_dd; DD2 = DD2 + dd[i, j, 2]*tmp_dd
                    else: 
                        if i == 0 and j == 0:
                            SS0_new = SS0[i, j]*tmp_ss; SS1_new = SS1[i, j]*tmp_ss; SS2_new = SS2[i, j]*tmp_ss
                            SD0_new = SD0[i, j]*tmp_ss; SD1_new = SD1[i, j]*tmp_ss; SD2_new = SD2[i, j]*tmp_ss
                            DS0_new = DS0[i, j]*tmp_ss; DS1_new = DS1[i, j]*tmp_ss; DS2_new = DS2[i, j]*tmp_ss
                            DD0_new = DD0[i, j]*tmp_ss; DD1_new = DD1[i, j]*tmp_ss; DD2_new = DD2[i, j]*tmp_ss
                        else:
                            SS0_new = SS0_new + SS0[i, j]*tmp_ss; SS1_new = SS1_new + SS1[i, j]*tmp_ss; SS2_new = SS2_new + SS2[i, j]*tmp_ss
                            SD0_new = SD0_new + SD0[i, j]*tmp_ss; SD1_new = SD1_new + SD1[i, j]*tmp_ss; SD2_new = SD2_new + SD2[i, j]*tmp_ss
                            DS0_new = DS0_new + DS0[i, j]*tmp_ss; DS1_new = DS1_new + DS1[i, j]*tmp_ss; DS2_new = DS2_new + DS2[i, j]*tmp_ss
                            DD0_new = DD0_new + DD0[i, j]*tmp_ss; DD1_new = DD1_new + DD1[i, j]*tmp_ss; DD2_new = DD2_new + DD2[i, j]*tmp_ss
            if k != 0:
                SS0 = SS0_new; SS1 = SS1_new; SS2 = SS2_new; 
                SD0 = SD0_new; SD1 = SD1_new; SD2 = SD2_new; 
                DS0 = DS0_new; DS1 = DS1_new; DS2 = DS2_new; 
                DD0 = DD0_new; DD1 = DD1_new; DD2 = DD2_new; 

        SS.append(np.stack([SS0, SS1, SS2], 2))
        SD.append(np.stack([SD0, SD1, SD2], 2))
        DS.append(np.stack([DS0, DS1, DS2], 2))
        DD.append(np.stack([DD0, DD1, DD2], 2))
                                
                    
        print(f'Decomposition Level {level} Completed!')

    return SS, SD, DS, DD