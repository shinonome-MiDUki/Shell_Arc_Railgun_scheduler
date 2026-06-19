from enum import StrEnum

import numpy as np
import pandas as pd

import cone_equation
import data_struct

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
    theoretical_length = proj_length * (data_struct.FINENESS + 1)
    missing_val_num = theoretical_length - M_length
    pseudo_val_array = np.zeros(missing_val_num, dtype=data_struct.point_dtype)
    existing_pairs = set(map(tuple, M[:, :2]))
    pseudo_item_count = 0
    for cut_i in (1,total_cut_num + 1):
        for component_i in (0, 11):
            if (cut_i, component_i) not in existing_pairs:
                if pseudo_item_count < missing_val_num:
                    pseudo_val_array[pseudo_item_count][data_struct.NpData.POS] = [cut_i, component_i, proj_length]
                    pseudo_val_array[pseudo_item_count][data_struct.NpData.COMP] = "None"
                    pseudo_item_count += 1
    complemented_M = np.vstack(M, pseudo_val_array)
    return complemented_M
    

def get_z_offset_array(M: np.ndarray,
                       total_cut_num: int,
                       proj_length: int,
                       today: int
                       ) -> np.ndarray:
    proj_ctx_constant = cone_equation.proj_context_constant(
        total_cut_num=total_cut_num,
        proj_length=proj_length
    )
    complemented_M = complement_array(
        M=M,
        proj_length=proj_length,
        total_cut_num=total_cut_num
    )
    z_offset_array = np.zeros(
        proj_length * (data_struct.FINENESS + 1),
        dtype=data_struct.z_offset_dtype
        )
    z_offset_array[data_struct.NpData.OFFSET] = cone_equation.compute_z_offset(
        cut=complemented_M[:, 0],
        component=complemented_M[:, 1],
        day=complemented_M[:, 2],
        proj_ctx_constant=proj_ctx_constant
    )
    z_offset_array[data_struct.NpData.COMP] = complemented_M[data_struct.NpData.COMP]
    z_offset_array[data_struct.NpData.POS] = complemented_M[data_struct.NpData.POS]
    return z_offset_array
