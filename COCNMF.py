#=====================================================================
# Reference:
# C.-H. Lin, F. Ma, C.-Y. Chi, and C.-H. Hsieh,
# ``A convex optimization based coupled non-negative matrix factorization algorithm for hyperspectral and multispectral data fusion,"
# accepted by IEEE Trans. Geoscience and Remote Sensing, 2017.
#======================================================================
# A convex optimization based coupled NMF algorithm for hyperspectral superresolution via big data fusion
# Z_fused = ConvOptiCNMF(Yh,Ym,N,D,K)
#======================================================================
#  Input
#  Yh is low-spatial-resolution hyperspectral data cube of dimension rows_h*columns_h*M.
#       (rows_h: vertical spatial dimension; 
#        columns_h: horizontal spatial dimension; 
#        M: spectral dimension.)
#  Ym is high-spatial-resolution multispectral data cube of dimension rows_m*columns_m*Mm.
#       (rows_m: vertical spatial dimension; 
#        columns_m: horizontal spatial dimension; 
#        M_m: spectral dimension.)
#  N is the model order (should be greater than the number of endmembers).
#  D is spectral response transform matrix of dimension M_m*M.
#  K is blurring kernel of dimension k*k
#       (The first dimension of K corresponds to the vertical spatial dimension;
#        the second dimension of K corresponds to the horizontal spatial dimension.)
#----------------------------------------------------------------------
#  Output
#  Z_fused is super-resolved hyperspectral data cube of dimension rows_m*columns_m*M.
#========================================================================
from typing import Tuple
import numpy as np
from numpy.linalg import eig, det, norm, pinv
from scipy import sparse as sp
from scipy.sparse.linalg import inv, spsolve
from scipy import ndimage

# main program
def ConvOptiCNMF(Yh: np.ndarray, Ym: np.ndarray, N: int, D: np.ndarray, K: np.ndarray) -> np.ndarray:
    lambda1 = 0.001 # SSD regularization parameter
    lambda2 = 0.001 # L1-norm regularization parameter
    eta = 1 # penalty parameter in ADMM Algorithm 2
    tildeeta = 1 # penalty parameter in ADMM Algorithm 3
    mode = 1 # 0 uses Eq.(17)&(26) (for non-structured blur), 1 uses Lemmas 1 & 2 (for structured blur)
    # Max_iter = 30 # maximum iteration of Algorithm 1
    # iterS = 5 # maximum iteration of Algorithm 2
    # iterA = 5 # maximum iteration of Algorithm 3
    Max_iter = None # maximum iteration of Algorithm 1
    iterS = None # maximum iteration of Algorithm 2
    iterA = None # maximum iteration of Algorithm 3

    rows_m, cols_m, bands_m = Ym.shape
    rows_h, cols_h, M = Yh.shape
    Yh = Yh.reshape(-1, M, order='F').T
    M, Lh = Yh.shape
    r1 = rows_m / rows_h # vertical dimension
    r2 = cols_m / cols_h # horizontal dimension
    r = int(np.amax([np.ceil(r1), np.ceil(r2)]))
    Ym = imresize(Ym, [r * rows_h, r * cols_h])
    Ym = Ym.reshape(-1, bands_m, order='F').T
    Perm, g, B = Permutation(K, r, rows_h, cols_h)
    if mode == 1:
        Ym = Ym @ Perm.T

    I_M = sp.eye(M)
    P = sp.lil_matrix((int(M * N * (N - 1) / 2), M * N))
    x = 0
    for n in range(N - 1):
        e_n = sp.eye(N, format='csc')
        H_n = sp.kron(e_n[:, n].T, I_M)
        for m in range(n + 1, N):
            P_m = sp.kron(e_n[:, m].T, I_M)
            P[x:(x+M), :] = H_n - P_m
            x += M
    PtP = (P.T @ P).asformat('csr')

    A, S_h = HyperCSI_int(Yh, N) # initialization by HyperCSI [43]
    S = S_h @ sp.kron(sp.eye(Lh), np.ones(g.T.shape))
    # calculating loss
    CNMF = norm(Yh - A @ S @ B, ord='fro') + norm(Ym - D @ A @ S, ord='fro')
    phi1 = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            phi1 += norm(A[:, i] - A[:, j], ord=2)
    phi1 *= 0.5
    phi2 = norm(S, ord=1)
    loss = 0.5 * CNMF + lambda1 * phi1 + lambda2 * phi2

    k = 1
    while Max_iter is None or k <= Max_iter:
        print('Iteration', k)
        # store previous values
        loss_pre = loss
        # Two subproblems
        S = S_NonNeg_Lasso(N, Yh, Ym, A, S, D, g, B, iterS, lambda2, eta, mode) # Algorithm 2 (warm start)
        A = A_ICE(N, Yh, Ym, S, A, D, g, B, iterA, lambda1, tildeeta, PtP, mode) # Algorithm 3 (warm start)
        # calculating loss
        CNMF = norm(Yh - A @ S @ B, ord='fro') + norm(Ym - D @ A @ S, ord='fro')
        phi1 = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                phi1 += norm(A[:, i] - A[:, j], ord=2)
        phi1 *= 0.5
        phi2 = norm(S, ord=1)
        loss = 0.5 * CNMF + lambda1 * phi1 + lambda2 * phi2
        # stopping criteria
        change = np.absolute(loss - loss_pre)
        print(f'Iteration {k}: loss = {loss}, change = {change}')
        print()
        if change < 1e-3:
            break
        k += 1
    
    if mode == 0:
        Z_fused = A @ S
    else:
        Z_fused = A @ S @ Perm
    Z_fused = Z_fused.T.A.reshape(r * rows_h, r * cols_h, M, order='F')
    Z_fused = imresize(Z_fused, [rows_m, cols_m]) # resize back

    return Z_fused

# subprogram 1 (implementation of Algorithm 2, solving the non-negative LASSO)
def S_NonNeg_Lasso(N: int, Yh: np.ndarray, Ym: np.ndarray, A: np.ndarray, S_old: sp.spmatrix, D: np.ndarray, g: np.ndarray, B: sp.spmatrix, iterS: int, lambda2: float, eta: float, mode: int) -> sp.spmatrix:
    L = Ym.shape[1]
    y = np.hstack([Yh.reshape(-1, 1, order='F').T, Ym.reshape(-1, 1, order='F').T]).reshape(-1, 1, order='F')
    x = S_old.reshape(-1, 1, order='F')
    nu = sp.csc_matrix((N * L, 1))
    
    if mode == 1:
        C_bar1 = sp.kron(g.T, A).T
        C_bar2 = sp.kron(sp.eye(g.shape[0]) , D @ A).T
        C_bartC_bar = sp.hstack([C_bar1, C_bar2]) @ sp.hstack([C_bar1, C_bar2]).T
        C_bar_inv = C_bartC_bar + eta * sp.eye(C_bartC_bar.shape[0], C_bartC_bar.shape[1])
        C_bar_inv = inv(C_bar_inv.asformat('csc'))
        C_binv1 = C_bar_inv @ C_bar1
        C_binv2 = C_bar_inv @ C_bar2
        Ym_re = Ym.reshape(C_binv2.shape[1], -1, order='F')
        C_bYhYm = (C_binv1 @ Yh + C_binv2 @ Ym_re).reshape(-1, 1, order='F')

        j = 1
        while iterS is None or j <= iterS:
            # store previous values
            xp = x.copy()
            # s update
            xnu_re = (eta * x - nu).reshape(C_bar_inv.shape[1], -1, order='F')
            s = (C_bar_inv @ xnu_re).reshape(-1, 1, order='F') + C_bYhYm
            # x update
            x = s + nu / eta - lambda2 / eta
            x[x < 0] = 0
            # nu update
            nu = nu + eta * (s - x)
            # stopping criteria
            residual = norm(s - x)
            dual_residual = norm(-eta * (x - xp))
            print(f'S Iteration {j}: residual = {residual}, dual residual = {dual_residual}')
            if residual < 1e-3 and dual_residual < 1e-3:
                break
            j += 1
    
    else: # naive version
        C = sp.hstack([sp.kron(B.T, A).T, sp.kron(sp.eye(L), D @ A).T]).T
        CtC = C.T @ C
        Cty = C.T @ y
        Lbs = CtC + eta * sp.eye(N * L)
        
        j = 1
        while iterS is None or j <= iterS:
            # store previous values
            xp = x.copy()
            # s update
            q = Cty + eta * x - nu
            s = spsolve(Lbs, q.asformat('csc'))
            # x update
            x = s + nu / eta - lambda2 / eta
            x[x < 0] = 0
            # nu update
            nu = nu + eta *(s - x)
            # stopping criteria
            residual = norm(s - x)
            dual_residual = norm(-eta * (x - xp))
            print(f'S Iteration {j}: residual = {residual}, dual residual = {dual_residual}')
            if residual < 1e-3 and dual_residual < 1e-3:
                break
            j += 1

    S = x.reshape(N, L, order='F')

    return S

# subprogram 2 (implementation of Algorithm 3, solving the ICE-regularized problem)
def A_ICE(N: int, Yh: np.ndarray, Ym: np.ndarray, S: sp.spmatrix, A_old: np.ndarray, D: sp.spmatrix, g: np.ndarray, B: sp.spmatrix, iterA: int, lambda1: float, tildeeta: float, PtP: sp.spmatrix, mode: int) -> np.ndarray:
    M = Yh.shape[0]
    y = np.hstack([Yh.reshape(-1, 1, order='F').T, Ym.reshape(-1, 1, order='F').T]).reshape(-1, 1, order='F')
    z = A_old.reshape(-1, 1, order='F')
    nu = sp.csc_matrix((M * N, 1))
    
    if mode == 1:
        S_3d = S.A.reshape(N, g.shape[0], -1, order='F')
        S_perm = np.moveaxis(S_3d, 2, 1)
        S_re = S_perm.reshape(-1, g.shape[0], order='F')
        Sgv = (S_re @ g).reshape(N, -1, order='F')
        CtCb1 = sp.kron(Sgv @ Sgv.T, sp.eye(M))
        DtD = D.T @ D
        StS = S @ S.T
        CtCb2 = sp.kron(StS, DtD)
        CtC = CtCb1 + CtCb2
        CtYh = ((Sgv @ Yh.T).T).reshape(-1, 1, order='F')
        CtYm = ((S @ (D.T @ Ym).T).T).reshape(-1, 1, order='F')
        Cty = CtYh + CtYm
        CtCinv = CtC + lambda1 * PtP + tildeeta * sp.eye(M * N)
        CtCinv = inv(CtCinv.asformat('csc'))
        
        j = 1
        while iterA is None or j <= iterA:
            # store previous values
            zp = z.copy()
            # a update
            a = CtCinv @ (Cty + tildeeta * z - nu)
            # z update
            z = a + nu / tildeeta
            z[z < 0] = 0
            # nu update
            nu = nu + tildeeta * (a - z)
            # stopping criteria
            residual = norm(a - z)
            dual_residual = norm(-tildeeta * (z - zp))
            print(f'A Iteration {j}: residual = {residual}, dual residual = {dual_residual}')
            if residual < 1e-3 and dual_residual < 1e-3:
                break
            j += 1
    
    else: # naive version
        C = [sp.kron((S @ B).T, sp.eye(M)).T, sp.kron(S.T, D).T].T
        CtC = C.T @ C
        Cty = C.T @ y
        Lba = CtC + lambda1 * PtP + tildeeta * sp.eye(M * N)

        j = 1
        while iterA is None or j <= iterA:
            print('A Iteration', j)
            # store previous values
            zp = z.copy()
            # a update
            q = Cty + tildeeta * z - nu
            a = spsolve(Lba, q.asformat('csc'))
            # z update
            z = a + nu / tildeeta
            z[z < 0] = 0
            # nu update
            nu = nu + tildeeta * (a - z)
            # stopping criteria
            residual = norm(a - z)
            dual_residual = norm(-tildeeta * (z - zp))
            print(f'A Iteration {j}: residual = {residual}, dual residual = {dual_residual}')
            if residual < 1e-3 and dual_residual < 1e-3:
                break
            j += 1

    A = z.reshape(M, N, order='F')

    return A

# subprogram 3
# This function helps user automatically generate the spatial spread
# transform matrix B from the blurring kernel K, automatically permute the
# pixels in a specific order (so that our fast closed-form solutions can be
# applied), and then accordingly revise the spatial spread transform matrix.
# This function considers the following 4 types of K, and automatically adapts
# it to the desired form with the original properties (variance) remained unchanged:
#   1. K is symmetric Gaussian, and k=sqrt((rows_m*columns_m)/(rows_h*columns_h))=r (the desired form). 
#   2. K is symmetric Gaussian, but k does not equal to r.
#   3. K is not symmetric Gaussian but is uniform.
#   4. K is not symmetric Gaussian and is non-uniform.
def Permutation(K: np.ndarray, r: int, rows_h: int, cols_h: int) -> Tuple[sp.spmatrix, np.ndarray, sp.spmatrix]:
    blurkernel = K.shape[0]
    r_square=r ** 2
    isgusyK, sigmaK = isgusy(K)

    if np.all(K[:, :] == K[0, 0]) is True:
        Temp = np.ones([r, r]) / (r ** 2)
    elif blurkernel == r and isgusyK is True:
        Temp = K
    elif isgusyK is True:
        Temp = gaussion2D([r, r], sigmaK)
    else:
        zz = 0
        ubound = 150
        lbound = 0.05
        interval = 0.05
        a = np.zeros([((ubound - lbound) / interval) + 1, 1])
        for sigma in range(lbound, interval, ubound + interval):
            Temp = gaussion2D([blurkernel, blurkernel], sigma)
            a[zz, 0] = norm(Temp - K, ord='fro')
            zz += 1
        index = np.argmin(a)
        sigma = lbound + (index - 1) * interval
        Temp = gaussion2D([r, r], sigma)
    
    # generate B before permutation
    Hv = sp.eye(rows_h * cols_h, format='csc')
    B = sp.lil_matrix((rows_h * cols_h * r * r, rows_h * cols_h))
    for i in range(rows_h * cols_h):
        Hm = Hv[:, i].reshape(rows_h, cols_h, order='F')
        B[:, i] = sp.kron(Hm, Temp).reshape(-1, 1, order='F')
    B = B.asformat('csc')
    gv = Temp.reshape(-1, 1)
    
    # start permutation
    Pi = sp.lil_matrix((r_square, r_square))
    Pi = sp.kron(np.ones([1, rows_h * cols_h]), Pi, format='lil')
    Pi_temp = Pi.copy()
    vv, _, _ = sp.find(B[:, 0] > 0)
    for j in range(len(vv)):
        Pi[j, vv[j]] = 1
    for i in range(1, rows_h * cols_h):
        vv, _, _ = sp.find(B[:, i] > 0)
        Pi_new = Pi_temp.copy()
        for j in range(len(vv)):
            Pi_new[j, vv[j]] = 1
        Pi = sp.vstack([Pi, Pi_new], format='csr')

    return Pi, gv, B

# subprogram 4
# check if K is symmetric Gaussian
def isgusy(K: np.ndarray) -> Tuple[bool, float]:
    k = K.shape[0]
    issym = lambda x: np.array_equal(x, x.T)
    updown=np.sum(np.absolute(K - np.fliplr(K)), axis=None)
    leftright=np.sum(np.absolute(K - np.flipud(K)), axis=None)

    if (issym(K) is True and updown == 0 and leftright == 0): # check the symmetricity
        if (k % 2 == 0): # define the coordinate w.r.t the center
            index = (k / 2) - 0.5
        else:
            index = np.floor(k / 2)
        
        sigma = np.sqrt((2 * (index ** 2) - 2 * (index - 1) ** 2) / (2 * np.log(K[1, 1] / K[0, 0])))
        temp = gaussion2D([k, k], sigma)
        if np.sum(np.absolute(K - temp), axis=None) < 1e-10:
            E = True
        else:
            E = False
    else:
        E = False
    
    return E, sigma

# subprogram 5
# HyperCSI adapted for initialization
def HyperCSI_int(X: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    con_tol = 1e-8
    num_SPA_itr = N

    # Step 1
    M, L = X.shape
    d = np.mean(X, axis=1).reshape(-1, 1)
    U = X - d @ np.ones([1, L])
    D, eV = eig(U @ U.T)
    idx = D.argsort()[::-1]
    eV = eV[:, idx]
    C = eV[:, :(N-1)]
    Xd = C.T @ U
    
    # Step 2
    alpha_tilde = iterativeSPA(Xd, L, N, num_SPA_itr, con_tol)
    
    # Step 3
    bi_tilde = np.empty([N-1, N])
    for i in range(N):
        bi_tilde[:, i] = compute_bi(alpha_tilde, i, N)
    
    radius = 1e-8
    for i in range(0, N-1):
        for j in range(i+1, N):
            dist_ai_aj = norm(alpha_tilde[:, i] - alpha_tilde[:, j])
            if (1 / 2) * dist_ai_aj < radius:
                radius = (1 / 2) * dist_ai_aj
    
    Xd_divided_idx = np.zeros([L])
    radius_square = radius ** 2
    for i in range(N):
        ds = np.sum((Xd - alpha_tilde[:, i].reshape(-1, 1) @ np.ones([1, L])) ** 2, axis=0)
        IDX_alpha_i_tilde = np.argwhere(ds < radius_square)[:, 0]
        Xd_divided_idx[IDX_alpha_i_tilde] = i + 1
    
    # Step 4
    b_hat = np.empty([N-1, N])
    h = np.empty([N, 1])
    for i in range(N):
        Hi_idx = np.setdiff1d(range(N), [i])
        pi_k = np.zeros([N-1, N-1])
        for k in range(1*(N-1)):
            Ri_k = Xd[:, Xd_divided_idx == Hi_idx[k] + 1]
            idx = np.argmax(bi_tilde[:, i].T @ Ri_k)
            pi_k[:, k] = Ri_k[:, idx]
        b_hat[:, i] = compute_bi(np.hstack([pi_k, alpha_tilde[:, i].reshape(-1, 1)]), N-1, N)
        h[i, 0] = np.amax(b_hat[:, i].T @ pi_k[:, 0]) # actually it should be compared with all pi_k, I don't know why here omitted
    
    # Step 6
    alpha_hat = np.empty([N-1, N])
    for i in range(N):
        bbbb = np.delete(b_hat, i, axis=1)
        ccconst = np.delete(h, i, axis=0)
        alpha_hat[:, i] = (pinv(bbbb.T) @ ccconst).reshape(-1)
    
    A_est = C @ alpha_hat + d @ np.ones([1, N])
    
    # Step 7
    D1 = h @ np.ones([1, L]) - b_hat.T @ Xd
    D2 = (h - np.sum(b_hat * alpha_hat, axis=0).T.reshape(-1, 1)) @ np.ones([1, L])
    S_est = np.divide(D1, D2, out=D1.copy(), where=(D2 != 0))
    S_est[S_est < 0] = 0

    return A_est, S_est

# subprogram 6
# SPA with post-processing
def iterativeSPA(Xd: np.ndarray, L: int, N: int, num_SPA_itr: int, con_tol: float) -> np.ndarray:
    p = 2
    N_max = N

    Xd_t = np.vstack([Xd, np.ones([1, L])])
    pnorms = np.sum(np.absolute(Xd_t) ** p, axis=0) ** (1 / p)
    ind = np.argmax(pnorms)
    A_set = Xd_t[:, ind].reshape(-1, 1)
    index = [ind]
    for i in range(1, N):
        XX = (np.eye(N_max) - A_set @ pinv(A_set)) @ Xd_t
        pnorms = np.sum(np.absolute(XX) ** p, axis=0) ** (1 / p)
        ind = np.argmax(pnorms)
        A_set = np.hstack([A_set, Xd_t[:, ind].reshape(-1, 1)])
        index.append(ind)
    
    alpha_tilde = Xd[:, index]
    current_vol = det(alpha_tilde[:, :(N-1)] - alpha_tilde[:, N-1].reshape(-1, 1) @ np.ones([1, N-1]))
    for _ in range(num_SPA_itr):
        b = np.empty([N-1, N])
        for i in range(N):
            b[:, i] = compute_bi(alpha_tilde, i, N)
            b[:, i] = -b[:, i]
            idx = np.argmax(b[:, i].T @ Xd)
            alpha_tilde[:, i] = Xd[:, idx]
        new_vol = det(alpha_tilde[:, :(N-1)] - alpha_tilde[:, N-1].reshape(-1, 1) @ np.ones([1, N-1]))
        if (new_vol - current_vol) / current_vol < con_tol:
            break
    
    return alpha_tilde

# subprogram 7
# compute normal vectors
# modify: using 0 as the first index
def compute_bi(A: np.ndarray, i: int, N: int) -> np.ndarray:
    Hidx = np.setdiff1d(range(N), [i])
    A_Hidx = A[:, Hidx]
    P = A_Hidx[:, :(N-2)] - A_Hidx[:, N-2].reshape(-1, 1) @ np.ones([1, N-2])
    bi = A_Hidx[:, N-2] - A[:, i]
    bi = (np.eye(N-1) - P @ pinv(P.T @ P) @ P.T) @ bi
    bi = bi / norm(bi) if norm(bi) > 0 else bi

    return bi

# additional subprogram in Python
# 2D gaussian kernel - same as MATLAB's fspecial('gaussian',[shape],[sigma])
def gaussion2D(shape: Tuple[int] = [5, 5], sigma: float = 1.0) -> np.ndarray:
    m = (shape[0] - 1.0) / 2.0
    n = (shape[1] - 1.0) / 2.0
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    
    return h

# additional subprogram in Python
# resize the image array
def imresize(img: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    return ndimage.zoom(img, [shape[0] / img.shape[0], shape[1] / img.shape[1], 1])

# additional subprogram in Python
# Normalize the image color
def normColor(R: np.ndarray) -> np.ndarray:
    # nomalize the image R
    Z = (R - R.mean()) / R.std() # standardization

    return np.clip(Z, -2, 2) / 4 + 0.5