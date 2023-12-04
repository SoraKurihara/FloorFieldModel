import numpy as np


def calc_direction(k_Dir, directions):
    Direction_vals = []
    Direction_vals.append(
        np.where(directions == 0, 1, np.where(directions == 1, -1, 0))
    )
    Direction_vals.append(
        np.where(directions == 1, 1, np.where(directions == 0, -1, 0))
    )
    Direction_vals.append(
        np.where(directions == 2, 1, np.where(directions == 3, -1, 0))
    )
    Direction_vals.append(
        np.where(directions == 3, 1, np.where(directions == 2, -1, 0))
    )
    Direction_vals.append(np.zeros_like(directions))
    return np.exp(k_Dir * Direction_vals)


def calc_stress(k_Str, stresses):
    return np.exp(-k_Str * (stresses / 20))


def p_ij(
    Map,
    positions,
    SFF,
    DFF,
    k_S=3,
    k_D=1,
    k_Dir=None,
    directions=None,
    k_Str=None,
    stresses=None,
):
    shift_axis = [(1, 0), (-1, 0), (1, 1), (-1, 1), (0, 0)]  # 上下左右止
    M_vals, S_vals, D_vals = [], [], []

    for shift, axis in shift_axis:
        M_vals.append(
            np.roll(Map, shift=shift, axis=axis)[positions[:, 0], positions[:, 1]]
        )
        S_vals.append(
            np.roll(SFF, shift=shift, axis=axis)[positions[:, 0], positions[:, 1]]
        )
        D_vals.append(
            np.roll(DFF, shift=shift, axis=axis)[positions[:, 0], positions[:, 1]]
        )

    S_vals = np.array(S_vals)
    S_max = S_vals.max(axis=0)
    S_vals = np.where(S_vals == -1, S_max, S_vals)
    S_vals = S_vals - S_vals.min(axis=0)

    # 各方向の移動確率を計算
    probs = np.exp(-k_S * S_vals)
    probs *= np.exp(k_D * D_vals)
    if (k_Dir != None) and (directions != None):
        probs *= calc_direction(k_Dir, directions)
    if (k_Str != None) and (stresses != None):
        probs[4] *= calc_stress(k_Str, stresses)

    probs *= M_vals != 2
    probs[:4] *= M_vals[:4] != 1

    move = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # 上下左右止
    for idx, pos in enumerate(positions):
        for i, dir in enumerate(move):
            next_pos = tuple(np.array(pos) + np.array(dir))
            if Map[next_pos] == 3:
                probs[:, idx] = 0
                probs[i, idx] = 1
                break

    sums = np.sum(probs, axis=0)
    sums[sums == 0] = 1

    probs = probs / sums
    probs_tmp = 1 - probs[:-1, :].sum(axis=0)
    probs_tmp[probs_tmp < 0] = 0
    probs[-1, :] = probs_tmp
    return probs
