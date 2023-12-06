import numpy as np


def handle_collisions(positions, candidates, unique_positions, counts):
    duplicates = unique_positions[counts > 1]

    # 重複している位置ごとの処理
    for dup in duplicates:
        # 位置が重複しているすべてのインデックスを取得
        indices = np.where((candidates == dup).all(axis=1))[0]

        # 1/2 の確率でアクションを選択
        if np.random.random() < 0.5:
            # 一人のみをその位置に残し、他のすべての人を元の位置に戻す
            chosen_one = np.random.choice(indices)
            for idx in indices:
                if idx != chosen_one:
                    candidates[idx] = positions[idx]
        else:
            # すべての人を元の位置に戻す
            for idx in indices:
                candidates[idx] = positions[idx]

    return candidates
