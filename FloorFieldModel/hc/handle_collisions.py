import numpy as np


# def handle_collisions(positions, candidates, unique_positions, counts):
#     duplicates = unique_positions[counts > 1]

#     # 重複している位置ごとの処理
#     for dup in duplicates:
#         # 位置が重複しているすべてのインデックスを取得
#         # indices = np.where((candidates == dup).all(axis=1))[0]
#         indices = np.where(np.sum(candidates == dup, axis=1) == candidates.shape[1])[0]

#         # 1/2 の確率でアクションを選択
#         if np.random.random() < 0.5:
#             # 一人のみをその位置に残し、他のすべての人を元の位置に戻す
#             chosen_one = np.random.choice(indices)
#             for idx in indices:
#                 if idx != chosen_one:
#                     candidates[idx] = positions[idx]
#         else:
#             # すべての人を元の位置に戻す
#             for idx in indices:
#                 candidates[idx] = positions[idx]

#     return candidates

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 複数の重複を一度に処理する関数
def process_duplicates(duplicates, candidates_local, positions):
    for dup in duplicates:
        mask = np.logical_and.reduce(candidates_local == dup, axis=1)
        indices = np.flatnonzero(mask)
        
        if np.random.random() < 0.5:
            chosen_one = np.random.choice(indices)
            mask = np.ones(len(indices), dtype=bool)
            mask[indices == chosen_one] = False
            candidates_local[indices[mask]] = positions[indices[mask]]
        else:
            candidates_local[indices] = positions[indices]
    return candidates_local

def handle_collisions(positions, candidates, unique_positions, counts, chunk_size=10):
    # 重複している位置を抽出
    duplicates = unique_positions[counts > 1]
    
    # 重複リストをチャンクに分ける
    duplicate_chunks = [duplicates[i:i + chunk_size] for i in range(0, len(duplicates), chunk_size)]

    # 並列処理で重複リストのチャンクごとに処理
    candidates_copy = candidates.copy()  # 共有データをコピーして並列処理
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_duplicates, chunk, candidates_copy, positions) for chunk in duplicate_chunks]
        
        for future in as_completed(futures):
            candidates = future.result()

    return candidates

