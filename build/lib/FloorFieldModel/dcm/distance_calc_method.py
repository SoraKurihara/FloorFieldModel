import numpy as np
import skfmm

def get_outer(arr, Diagonal):
    shift_up = np.roll(arr, -1, axis=0)
    shift_down = np.roll(arr, 1, axis=0)
    shift_left = np.roll(arr, -1, axis=1)
    shift_right = np.roll(arr, 1, axis=1)

    outer_contour_mask = shift_up | shift_down | shift_left | shift_right

    if Diagonal:
        shift_up_left = np.roll(shift_left, -1, axis=0)
        shift_up_right = np.roll(shift_right, -1, axis=0)
        shift_down_left = np.roll(shift_left, 1, axis=0)
        shift_down_right = np.roll(shift_right, 1, axis=0)
        outer_contour_mask = (
            outer_contour_mask
            | shift_up_left
            | shift_up_right
            | shift_down_left
            | shift_down_right
        )

    return outer_contour_mask


def L1norm(Map):
    SFF = np.zeros_like(Map, dtype=np.float64)
    SFF[Map == 2] = -2
    SFF[Map == 0] = -1
    SFF[Map == 4] = -1

    count = 0
    while -1 in SFF:
        print(f"\r{count}", end="")
        starting_points = SFF == count
        increment = get_outer(starting_points, False)
        SFF[increment & (SFF == -1)] = count + 1
        count += 1

    print()
    SFF[SFF == -2] = -1
    return SFF


def L2norm(Map):
    phi = np.ones_like(Map)
    phi[Map == 3] = 0
    mask = Map == 2
    phi = np.ma.MaskedArray(phi, mask)
    SFF = skfmm.distance(phi, dx=1)
    SFF = np.where(SFF.mask, -1, SFF.data)
    return SFF


def Linfnorm(Map):
    SFF = np.zeros_like(Map, dtype=np.float64)
    SFF[Map == 2] = -2
    SFF[Map == 0] = -1
    SFF[Map == 4] = -1

    count = 0
    while -1 in SFF:
        print(f"\r{count}", end="")
        starting_points = SFF == count
        increment = get_outer(starting_points, True)
        if len(SFF[increment & (SFF == -1)]) == 0:
            break
        SFF[increment & (SFF == -1)] = count + 1
        count += 1
    SFF[SFF == 0] = -1

    print()
    SFF[SFF == -2] = -1
    return SFF
