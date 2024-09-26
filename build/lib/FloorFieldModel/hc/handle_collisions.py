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

