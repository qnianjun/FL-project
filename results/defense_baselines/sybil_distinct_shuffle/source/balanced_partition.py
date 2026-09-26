"""Capacity-constrained Dirichlet partitioning for experiment C.

Draw one client preference vector per label. Visit every sample in shuffled order,
choose a client using its label preference among clients with free slots, and stop
assigning to a client once full. This is a constrained variant, not the original
unconstrained Dirichlet partition. Realized label distributions must be inspected.
"""

import numpy as np


# 建立每端資料量相同的分配。回傳各 client 的圖片索引，不複製圖片。
# 容量限制會改變原始 Dirichlet 分布，因此必須另外檢查實際數字比例。
def balanced_partition(targets, num_clients, alpha, seed):
    labels = np.asarray(targets)

    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("targets must be a nonempty one-dimensional array")
    
    if num_clients < 1 or len(labels) % num_clients:
        raise ValueError("Sample count must be divisible by num_clients")
    
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    
    # 使用區域亂數產生器，不改變其他訓練步驟的亂數狀態。
    rng = np.random.default_rng(seed)

    # encoded 將原始類別轉成連續索引，對應下方每類的偏好表。
    classes, encoded = np.unique(labels, return_inverse=True)

    # 每列對應一種數字，每欄代表該數字分給某 client 的偏好。

    preferences = rng.dirichlet(np.full(num_clients, alpha), size=len(classes))
    # capacity 記錄剩餘名額；分配後每次減一，因此不需丟掉或重複補資料。

    capacity = np.full(num_clients, len(labels) // num_clients, dtype=int)
    partitions = [[] for _ in range(num_clients)]

    # 打亂所有圖片的處理順序，避免固定先分某個數字。
    for index in rng.permutation(len(labels)):

        weights = preferences[encoded[index]].copy()
        # 滿額 client 的機率歸零，剩下的機率重新加總為 1。

        weights[capacity == 0] = 0

        # Tiny-alpha draws can give zero probability to all remaining clients.
        # 極小 Alpha 可能讓仍有空位的 client 偏好全為零，此時均勻選擇剩餘者。
        if weights.sum() == 0:
            
            weights = (capacity > 0).astype(float)
        weights /= weights.sum()
        cid = int(rng.choice(num_clients, p=weights))
        partitions[cid].append(int(index))
        capacity[cid] -= 1
    # 完成時所有名額必須用完；每張圖片在上方只走訪一次。
    assert np.all(capacity == 0)
    return partitions
