import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import HomogMatrix2twist, twist2HomogMatrix, projectPoints
from scipy.optimize import least_squares
import scipy.sparse as sp


def _build_sparsity(index_lists, BA_window, num_points_total):
    """
    x = [poses for frames 1..BA_window-1] (6*(BA_window-1)) + [all points] (3*num_points_total)
    residuals stacked by frame, each observation contributes 2 residuals
    """
    n_pose = 6 * (BA_window - 1)
    n_vars = n_pose + 3 * num_points_total
    m_res = sum(2 * len(idxs) for idxs in index_lists)

    # each obs => 2 rows; depends on: pose(6) (if not anchor) + point(3) => up to 18 nonzeros per obs
    nnz = 0
    for fi, idxs in enumerate(index_lists):
        nobs = len(idxs)
        nnz += 2 * nobs * 3                # point part always
        if fi > 0:
            nnz += 2 * nobs * 6            # pose part for non-anchor frames

    Jpat = sp.lil_matrix((m_res, n_vars), dtype=np.int8)

    row = 0
    for fi, idxs in enumerate(index_lists):
        nobs = len(idxs)

        # pose block (frames 1..)
        if fi > 0:
            c0 = 6 * (fi - 1)
            Jpat[row:row + 2*nobs, c0:c0+6] = 1

        # point blocks
        for j, pid in enumerate(idxs):
            r0 = row + 2*j
            c0 = n_pose + 3*int(pid)
            Jpat[r0:r0+2, c0:c0+3] = 1

        row += 2*nobs

    return Jpat.tocsr()

def _build_X_complete(X_list, preserve_order=True):
    """
    X_list: list of ndarrays, each shape (n_i, 3)
    returns:
      X_complete: ndarray shape (m, 3) with unique rows across all inputs
      index_lists: list of ndarrays, each shape (n_i,), mapping rows of X_i -> indices in X_complete
    """
    lengths = np.array([x.shape[0] for x in X_list], dtype=int)
    splits = np.cumsum(lengths)[:-1]

    X_all = np.vstack(X_list)

    if preserve_order:
        # unique() sorts by default; use return_index to restore first-appearance order
        uniq, first_idx, inv = np.unique(
            X_all, axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first_idx)          # order of first appearance
        X_complete = uniq[order]

        # remap inverse indices from "sorted unique" -> "first-appearance unique"
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        inv = remap[inv]
    else:
        X_complete, inv = np.unique(X_all, axis=0, return_inverse=True)

    index_lists = np.split(inv, splits)
    return X_complete, index_lists


def _build_X_complete_tol(X_list, decimals=6, preserve_order=True):
    '''
    X_complete: ndarray shape (m, 3)
    '''
    lengths = np.array([x.shape[0] for x in X_list], dtype=int)
    splits = np.cumsum(lengths)[:-1]

    X_all = np.vstack(X_list)

    # quantize for tolerance-based matching
    Xq = np.round(X_all, decimals=decimals)

    if preserve_order:
        uniq, first_idx, inv = np.unique(Xq, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first_idx)
        X_complete = X_all[first_idx[order]]  # keep an original representative (not rounded)

        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        inv = remap[inv]
    else:
        uniq, inv = np.unique(Xq, axis=0, return_inverse=True)
        # pick representatives from original values (first occurrence of each rounded bin)
        _, first_idx = np.unique(inv, return_index=True)
        X_complete = X_all[np.sort(first_idx)]

    index_lists = np.split(inv, splits)
    return X_complete, index_lists


def do_BundleAdjustment(S_list_full, T_CW_list_full, K, BA_window, max_nfev=20):

    start_timer = time.perf_counter()
    
    # get the relevant frames
    S_list = S_list_full[-BA_window:]
    T_CW_list = T_CW_list_full[-BA_window:]

    # convert to twist
    tau_list = []
    for T in T_CW_list:
        last_row = np.array([0,0,0,1])
        tau = HomogMatrix2twist(np.vstack([T, last_row]))
        tau_list.append(tau)
    
    X_list = []
    for S in S_list:
        X = S["X"]
        X_list.append(X.T)
    
    # X_complete (n, 3) contains all seen landmarks in the sliding window
    # index_lists describes which which points of X_complete are seen in which frame
    # example: index_lists[k] = [3,4,5] --> X_complete[3,4,5] = X_k (X of frame k)
    X_complete, index_lists = _build_X_complete_tol(X_list)
    
    tau_anchor = tau_list[0]  # first pose (wont be optimized, stays the same for consistency of global path)
    # tau_anchor = tau_list[:2]  # first pose (wont be optimized, stays the same for consistency of global path)
    # x_0 = tau_list[1:] + [X_complete]
    x_0 = np.hstack([np.concatenate(tau_list[1:], axis=0).reshape(1, -1), X_complete.reshape(1, -1)]).reshape(-1)
    # x_0 = np.hstack([np.concatenate(tau_list[2:], axis=0).reshape(1, -1), X_complete.reshape(1, -1)]).reshape(-1)
    def error_fun(x):
        '''
        x: [tau_list[1:], Landmarks]
        f, Y: (n*k, 1) n: landmarks, k: frames
        '''
        x = x.reshape(1, -1)

        # X_complete_fun = x[BA_window-1]  # =x[-1]
        X_complete_fun = x[:, (BA_window-1)*6:].reshape(-1,3)

        # convert to T-Matrix & invert
        # tau_list_fun = [tau_anchor] + x[:BA_window-1]
        n = x[:, :(BA_window-1)*6].shape[1] // 6
        tau_rest = [t.ravel() for t in np.hsplit(x[:, :(BA_window-1)*6], n)]
        tau_list_fun = [tau_anchor] + tau_rest
        # tau_list_fun = tau_anchor + tau_rest

        T_CW_list_fun = []
        for tau in tau_list_fun:
            T_CW = twist2HomogMatrix(tau)
            # T_CW = np.linalg.inv(T_WC)
            # T_CW = T_WC
            T_CW_list_fun.append(T_CW)

        f_list = []  # list of k  * (2n, 1) arrays
        for i_x in range(BA_window):
            X_W = X_complete_fun[index_lists[i_x]]
            X_C = T_CW_list_fun[i_x][:3, :3] @ X_W.T + T_CW_list_fun[i_x][:3, 3].reshape(-1,1)
            P_estimated = projectPoints(X_C.T, K)  # list of apparently (n,2) arrays
            f_list.append(P_estimated.flatten(order='C').reshape(-1,1))

        Y_list = []  # list of k * (2n, 1) arrays
        for S in S_list:
            P = S["P"]
            Y_list.append(P.T.flatten(order='C').reshape(-1,1)) 
        
        f = np.vstack(f_list)
        Y = np.vstack(Y_list) 
        error = f - Y
        return error.reshape(-1)
    
    Jpat = _build_sparsity(index_lists, BA_window, X_complete.shape[0])
    tol = 1e-12  # 1e-12
    res = least_squares(error_fun, x_0, max_nfev=max_nfev, verbose=2, jac_sparsity=Jpat, loss="huber", gtol=tol, xtol=tol, ftol=tol)  # loss="huber"
    # tau_list_adjusted =  [tau_anchor] + res.x[:BA_window-1]
    # X_complete_adjusted = res.x[BA_window-1]  # =x[-1]
    n = res.x.reshape(1,-1)[:, :(BA_window-1)*6].shape[1] // 6
    tau_rest = [t.ravel() for t in np.hsplit(res.x.reshape(1,-1)[:, :(BA_window-1)*6], n)]
    tau_list_adjusted = [tau_anchor] + tau_rest
    # tau_list_adjusted = tau_anchor + tau_rest
    X_complete_adjusted = res.x.reshape(1,-1)[:, (BA_window-1)*6:].reshape(-1,3)

    # convert to T-Matrix
    T_CW_list_adjusted = []
    for tau in tau_list_adjusted:
        T_CW = twist2HomogMatrix(tau)
        T_CW_list_adjusted.append(T_CW[:3, :])
    
    # update with adjusted frames
    T_CW_list_full[-BA_window:] = T_CW_list_adjusted

    S_list_adjusted = S_list
    T_CW_list_full_vectorized = np.stack([T_CW.reshape(-1) for T_CW in T_CW_list_full], axis=1)
    for i_x in range(BA_window):
        S = S_list_adjusted[i_x]

        # update X
        X = X_complete_adjusted[index_lists[i_x]]
        S["X"] = X.T

        # update T
        T = S["T"]
        T_it = S["T_it"]
        T_it_raveled = np.asarray(T_it).ravel()
        # print(f'T_it_raveled.shape: {T_it_raveled.shape}')
        # print(f'T_WC_list_adjusted_vectorized.shape: {T_WC_list_full_vectorized.shape}')
        T_adjusted = T_CW_list_full_vectorized[:, T_it_raveled.astype(int)]
        S["T"] = T_adjusted

    
    # update with adjusted frames
    S_list_full[-BA_window:] = S_list_adjusted
    
    end_timer = time.perf_counter() 
    print(f'Time spent in Bundle Adjustment: {end_timer-start_timer}s')
    print(f'BA parameters: BA_window: {BA_window}, max_nfev (higher = more time): {max_nfev}')
    
    return S_list_full, T_CW_list_full 