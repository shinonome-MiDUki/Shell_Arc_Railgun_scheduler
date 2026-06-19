from enum import StrEnum

import numpy as np

from shellarc_railgun.computation_core.z_offset import cone_equation, reallocate_pts
from shellarc_railgun.data_model import data_struct

def get_z_offset_array(M: np.ndarray,
                       total_cut_num: int,
                       proj_length: int,
                       today: int
                       ) -> np.ndarray:
    proj_ctx_constant = cone_equation.proj_context_constant(
        total_cut_num=total_cut_num,
        proj_length=proj_length
    )
    complemented_M = reallocate_pts.relocate_points(
        M=M,
        total_cut_num=total_cut_num,
        proj_length=proj_length,
        today=today
    )
    z_offset_array = np.zeros(
        proj_length * (data_struct.FINENESS + 1),
        dtype=data_struct.z_offset_dtype
        )
    z_offset_array[data_struct.NpData.OFFSET] = cone_equation.compute_z_offset(
        cut=complemented_M[:, 0],
        component=complemented_M[:, 1],
        day=complemented_M[:, 2],
        proj_ctx_constant=proj_ctx_constant,
        today=today
    )
    z_offset_array[data_struct.NpData.COMP] = complemented_M[data_struct.NpData.COMP]
    z_offset_array[data_struct.NpData.POS] = complemented_M[data_struct.NpData.POS]
    return z_offset_array
