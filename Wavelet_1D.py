# Aliakbar Zarkoob, AKA "XIV"
# Gmail: XIV.Aliakbar.Zarkoob@gmail.com
# Telegram: @XIVAliakbar

import numpy as np


def Wavelet_1D(Signal, Type, Max_Dec_Level):
    
    if Type.lower() == 'haar':
        h = np.array([1/np.sqrt(2), 1/np.sqrt(2)]).reshape(-1, 1)
        g = np.array([1/np.sqrt(2), -1/np.sqrt(2)]).reshape(-1, 1)
    elif Type.lower() == 'db4':
        h = np.array([(1+np.sqrt(3))/(4*np.sqrt(2)), (3+np.sqrt(3))/(4*np.sqrt(2)), (3-np.sqrt(3))/(4*np.sqrt(2)), (1-np.sqrt(3))/(4*np.sqrt(2))]).reshape(-1, 1)
        g = np.array([h[3], -h[2], h[1], -h[0]]).reshape(-1, 1)
    elif Type.lower() == 'db6':
        h = np.array([0.3326705529500826, 0.8068915093110928, 0.4598775021184915, -0.1350110200102546, -0.0854412738822415, 0.0352262918857095]).reshape(-1, 1)
        g = np.array([0.0352262918857095, 0.0854412738822415, -0.1350110200102546, -0.4598775021184915, 0.8068915093110928, -0.3326705529500826]).reshape(-1, 1)
    elif Type.lower() == 'mexicanhat':
        h = np.array([(1-np.sqrt(7))/(16*np.sqrt(2)), (5+np.sqrt(7))/(16*np.sqrt(2)), (14+2*np.sqrt(7))/(16*np.sqrt(2)), (14-2*np.sqrt(7))/(16*np.sqrt(2)), (1-np.sqrt(7))/(16*np.sqrt(2)), (-3+np.sqrt(7))/(16*np.sqrt(2))]).reshape(-1, 1)
        g = np.array([h[5], -h[4], h[3], -h[2], h[1], -h[0]]).reshape(-1, 1)
    elif Type.lower() == 'sym3':
        h = np.array([0.3326705529500826, 0.8068915093133388, 0.4598775021184915, -0.1350110200102546, -0.0854412738820267, 0.0352262918857095]).reshape(-1, 1)
        g = np.array([-0.0352262918857095, -0.0854412738820267, 0.1350110200102546, 0.4598775021184915, -0.8068915093133388, 0.3326705529500826]).reshape(-1, 1)
    elif Type.lower() == 'sym2':
        h = np.array([-0.12940952255092145, 0.2241438680420134, 0.836516303737469, 0.48296291314469025]).reshape(-1, 1)
        g = np.array([-0.48296291314469025, 0.836516303737469, -0.2241438680420134, -0.12940952255092145]).reshape(-1, 1)
    elif Type.lower() == 'csd':
        # h = np.array([[-1], [-1], [2], [2], [-1], [-1]])
        h = np.ones((6,1))/np.sqrt(18)
        g = np.array([[1], [-1], [-2], [2], [1], [-1]])
    else:
        raise ValueError('The specified wavelet type is not supported in the current version of this function!')

    if np.mod(Signal.shape[0], 2) != 0:
        raise ValueError('The length of signal must be a power of 2!')
    
    if Max_Dec_Level >= np.log2(Signal.shape[0]):
        raise ValueError(f'The maximum decomposition level for the given data is {int(np.log2(n)-1)}!')

    def CreatBasis(q, h, g):
        CoeffNum = h.shape[0]
        V = np.zeros((q, q//2))
        W = np.zeros((q, q//2))
        for j in range(q//2):
            V_tmp = np.zeros((q, 1))
            W_tmp = np.zeros((q, 1))
            if j*2+CoeffNum > q:
                diff = j*2+CoeffNum - q
                l = len(h)
                V_tmp[:diff] = h[-diff:]
                V_tmp[-(l-diff):] = h[:(l-diff)]
                W_tmp[:diff] = g[-diff:]
                W_tmp[-(l-diff):] = g[:(l-diff)]
            else:
                V_tmp[j*2:j*2+CoeffNum] = h
                W_tmp[j*2:j*2+CoeffNum] = g
            V[:, j] = V_tmp.flatten()
            W[:, j] = W_tmp.flatten()
        return V, W

    S = []; D = []
    f = Signal
    for i in range(Max_Dec_Level):
        level = i + 1
        n = f.shape[0]
        V, W = CreatBasis(n, h, g)
        s = f.T@V
        d = f.T@W  
        f = s.T
        
        for k in range(level):
            n = Signal.shape[0]//(2**(level-k-1))
            V, W = CreatBasis(n, h, g)
                
            if k == 0:
                S_level = np.sum(V*s, 1)
                D_level = np.sum(W*d, 1)
            else:
                S_level = np.sum(V*S_level, 1)
                D_level = np.sum(V*D_level, 1)
        S.append(S_level)
        D.append(D_level)
                
    return S, D