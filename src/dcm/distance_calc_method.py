import numpy as np
import skfmm


def L1norm(Map):
    SFF = np.zeros_like(Map, dtype=np.int64)
    SFF[Map == 2] = -2
    SFF[Map == 0] = -1
    SFF[Map == 4] = -1

    count = 0
    while -1 in SFF:
        print(f"\r{count}", end="")
        starting_points = SFF == count
        for shift, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1)]:
            increment = np.roll(starting_points, shift=shift, axis=axis) & (SFF == -1)
            SFF[increment] = count + 1
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
