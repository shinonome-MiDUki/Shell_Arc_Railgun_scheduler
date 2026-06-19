from enum import StrEnum

import numpy as np
import pandas as pd

FINENESS = 10

class NpData(StrEnum):
    POS = "position"
    COMP = "component"
    OFFSET = "offset"

point_dtype = np.dtype([
    (NpData.POS, 'i4', (3)),  
    (NpData.COMP, 'U10')           
])

z_offset_dtype = np.dtype([
    (NpData.OFFSET, 'i4', (1)),
    (NpData.POS, 'i4', (3)),  
    (NpData.COMP, 'U10')  
])

def proj_context_constant(total_cut_num: int,
                          proj_length: int
                          ) -> float:
    ctx_constant = proj_length / ((total_cut_num ** 2) + 100) ** 0.5
    return ctx_constant

def sort_array_by_cut_progress(M: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(M, columns=['cut', 'component', 'day'])
    df['component_count'] = df.groupby('cut')['component'].transform('nunique')
    df_sorted = df.sort_values(
        by=['component_count', 'cut'], 
        ascending=[False, True]
        )
    sorted_M = df_sorted[['cut', 'component', 'day']].to_numpy()
    return sorted_M

def complement_array(M: np.ndarray,
                    proj_length: int,
                    total_cut_num: int
                    ) -> np.ndarray:
    M_length = M.shape[0]
    theoretical_length = proj_length * (FINENESS + 1)
    missing_val_num = theoretical_length - M_length
    pseudo_val_array = np.zeros(missing_val_num, dtype=point_dtype)
    existing_pairs = set(map(tuple, M[:, :2]))
    pseudo_item_count = 0
    for cut_i in (1,total_cut_num + 1):
        for component_i in (0, 11):
            if (cut_i, component_i) not in existing_pairs:
                if pseudo_item_count < missing_val_num:
                    pseudo_val_array[pseudo_item_count][NpData.POS] = [cut_i, component_i, proj_length]
                    pseudo_val_array[pseudo_item_count][NpData.COMP] = "None"
                    pseudo_item_count += 1
    complemented_M = np.vstack(M, pseudo_val_array)
    return complemented_M



def compute_z_offset(cut: np.ndarray,
                     component: np.ndarray,
                     day: np.ndarray,
                     proj_ctx_constant: float
                     ) -> np.ndarray:
    t = proj_ctx_constant
    ideal_day = (1 - t) * ((cut ** 2) + (component ** 2)) ** 0.5
    offset_array = ideal_day - day
    return offset_array
    

def get_z_offset_array(M: np.ndarray,
                       total_cut_num: int,
                       proj_length: int,
                       today: int
                       ) -> np.ndarray:
    proj_ctx_constant = proj_context_constant(
        total_cut_num=total_cut_num,
        proj_length=proj_length
    )
    complemented_M = complement_array(
        M=M,
        proj_length=proj_length,
        total_cut_num=total_cut_num
    )
    z_offset_array = np.zeros(
        proj_length * (FINENESS + 1),
        dtype=z_offset_dtype
        )
    z_offset_array[NpData.OFFSET] = compute_z_offset(
        cut=complemented_M[:, 0],
        component=complemented_M[:, 1],
        day=complemented_M[:, 2],
        proj_ctx_constant=proj_ctx_constant
    )
    z_offset_array[NpData.COMP] = complemented_M[NpData.COMP]
    z_offset_array[NpData.POS] = complemented_M[NpData.POS]
    return z_offset_array


# ==================== 動作確認 ====================
rng = np.random.default_rng(42)

# N1: x in {1,2,3}, 各xのy要素数を変えて右下がりになるか確認
#   x=1 -> y: 4点, x=2 -> y: 2点, x=3 -> y: 1点
N1 = np.array([
    [1, 10, 5.0], [1, 20, 6.0], [1, 30, 7.0], [1, 40, 8.0],  # x=1: 4点
    [2, 10, 3.0], [2, 20, 4.0],                                # x=2: 2点
    [3, 10, 9.0],                                              # x=3: 1点
], dtype=float)

# N2: N1の部分集合（x=3は欠損させて c が使われるか確認）
N2 = np.array([
    [1, 10, 1.0], [1, 20, 2.0], [1, 30, 3.0], [1, 40, 4.0],
    [2, 10, 1.0],
    # (2,20), (3,10) は欠損
], dtype=float)

c = -999.0

result = compute_z_offset_sorted(N1, N2, c)

print("result (x, y, z_offset):")
print(result)
print()

# x別のy要素数確認
unique_x, counts = np.unique(result[:, 0], return_counts=True)
print("x別 y要素数（降順確認）:")
for x, cnt in zip(unique_x, counts):
    print(f"  x={x:.0f}: {cnt}点")