import numpy as np

from shellarc_railgun.data_model import data_struct
from shellarc_railgun.computation_core.intercepts import component_day_plane

def get_diff_array(M: np.ndarray,
                   total_cut_num: int,
                   today: int
                   ) -> np.ndarray:
    combined_diff_array = np.concatenate(
        [
            component_day_plane.get_cut_intercept_diff(
                M=M,
                cut_num=cut_num_i,
                today=today
            ) for cut_num_i in range(1, total_cut_num + 1)
        ]
    )
    return combined_diff_array

def get_frontend_array(M: np.ndarray,
                       total_cut_num: int
                       ) -> np.ndarray:
    frontend_ls = []
    for cut_num_i in total_cut_num:
        cut_intercept = M[M[data_struct.NpData.POS][:, 0] == cut_num_i]
        if len(cut_intercept) > 0:
            frontend_ls.append(cut_intercept[data_struct.NpData.POS][:, 1].max())
        else:
            frontend_ls.append(0.0)
    return np.array(frontend_ls)