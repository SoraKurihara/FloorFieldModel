# Mapの一番下（Map[-1,:]に相当する場所）に歩行者がいる場合，
# 下への移動方向が入るため，Out of boundsになる．
# ↑これは対処していない

import numpy as np
import skfmm

def p_ij(
    Map,
    positions,
    SFF,
    DFF,
    params=[3, 1],
    d="Neumann",
):
    k_S = params[0]
    k_D = params[1]

    if d == "Neumann":
        move = [(0, 1), (-1, 0), (0, -1), (1, 0), (0, 0)]  # 右上左下止
        M_vals, S_vals, D_vals = (
            np.empty((5, len(positions))),
            np.empty((5, len(positions))),
            np.empty((5, len(positions))),
        )
        coordinates = np.empty((5, len(positions), 2)) # New!

    else:
        move = [
            (0, 1),
            (1, 1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (-1, -1),
            (1, 0),
            (1, -1),
            (0, 0),
        ]  # 右右下上右上左左上下左下止
        M_vals, S_vals, D_vals = (
            np.empty((9, len(positions))),
            np.empty((9, len(positions))),
            np.empty((9, len(positions))),
        )
        coordinates = np.empty((9, len(positions), 2), dtype=int) # New!


    shift_axis = [(-1, 1), (1, 0), (1, 1), (-1, 0), (0, 0)]  # 右上左下止
        
    for idx, mo in enumerate(move):
        tmp = positions + mo
        tmp[tmp[:,0]>=Map.shape[0],0] = -1
        tmp[tmp[:,1]>=Map.shape[1],1] = -1
        coordinates[idx] = tmp

    M_vals = Map[coordinates[:, :, 0], coordinates[:, :, 1]]
    S_vals = SFF[coordinates[:, :, 0], coordinates[:, :, 1]]
    D_vals = DFF[coordinates[:, :, 0], coordinates[:, :, 1]]

    S_max = S_vals.max(axis=0)
    S_vals = np.where(S_vals == -1, S_max, S_vals)
    S_vals = S_vals - S_vals.min(axis=0)

    # 各方向の移動確率を計算
    probs = np.exp(-k_S * S_vals)
    probs *= np.exp(k_D * D_vals)

    probs *= M_vals != 2
    probs[:-1] *= M_vals[:-1] != 1

    move = np.array(move)
    p = positions[np.newaxis, :, :] + move[:, np.newaxis, :]
    Map_exit = Map[p[..., 0], p[..., 1]]
    is_three = Map_exit == 3
    has_three = np.any(is_three, axis=0)
    probs_updated = np.where(is_three, 1, 0)
    probs_updated *= np.cumsum(probs_updated, axis=0) == 1
    probs = np.where(has_three, probs_updated, probs)

    sums = np.sum(probs, axis=0)
    sums[sums == 0] = 1

    probs = probs / sums
    probs_tmp = 1 - probs[:-1, :].sum(axis=0)
    probs_tmp[probs_tmp < 0] = 0
    probs[-1, :] = probs_tmp
    return probs
