import numpy as np
import skfmm

# def L1norm(Map):
#     SFF = np.zeros_like(Map, dtype=np.int64)
#     SFF[Map == 2] = -2
#     SFF[Map == 0] = -1
#     SFF[Map == 4] = -1

#     count = 0
#     while -1 in SFF:
#         print(f"\r{count}", end="")
#         starting_points = SFF == count
#         for shift, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1)]:
#             increment = np.roll(starting_points, shift=shift, axis=axis) & (SFF == -1)
#             SFF[increment] = count + 1
#         count += 1
#     print()
#     SFF[SFF == -2] = -1
#     return SFF


def L2norm(Map):
    phi = np.ones_like(Map)
    phi[Map == 3] = 0
    mask = Map == 2
    phi = np.ma.MaskedArray(phi, mask)
    SFF = skfmm.distance(phi, dx=1)
    SFF = np.where(SFF.mask, -1, SFF.data)
    return SFF


def obstacle_outer(obstacle_mask):
    # 各方向にシフトしたマスクを生成
    shift_up = np.roll(obstacle_mask, -1, axis=0)
    shift_down = np.roll(obstacle_mask, 1, axis=0)
    shift_left = np.roll(obstacle_mask, -1, axis=1)
    shift_right = np.roll(obstacle_mask, 1, axis=1)

    # 角を含むシフト
    shift_up_left = np.roll(shift_left, -1, axis=0)
    shift_up_right = np.roll(shift_right, -1, axis=0)
    shift_down_left = np.roll(shift_left, 1, axis=0)
    shift_down_right = np.roll(shift_right, 1, axis=0)

    # 障害物の外側の輪郭を見つける
    outer_contour_mask = (
        shift_up
        | shift_down
        | shift_left
        | shift_right
        | shift_up_left
        | shift_up_right
        | shift_down_left
        | shift_down_right
    )
    return outer_contour_mask


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
            increment = np.roll(starting_points, shift=shift, axis=axis) & (
                SFF == -1
            )
            SFF[increment] += count + 2
        count += 1

    obstacle_mask = Map == 2
    not_obstacle_mask = Map == 0
    first_layer = obstacle_outer(obstacle_mask) & not_obstacle_mask
    second_layer = obstacle_outer(first_layer) & not_obstacle_mask
    SFF[first_layer] += 1
    SFF[second_layer] += 1

    print()
    SFF[SFF == -2] = -1
    return SFF
