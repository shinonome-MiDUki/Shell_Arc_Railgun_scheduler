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
                       total_cut_num: int,
                       today: int
                       ) -> np.ndarray:
    frontend_diff_list = []
    for cut_num_i in range(1, total_cut_num + 1):
        cut_diff_array = component_day_plane.get_cut_intercept_diff(
            M=M,
            cut_num=cut_num_i,
            today=today
            )
        if len(cut_diff_array[data_struct.NpData.POS][:, 1]) < 1:
            continue
        frontend_component = cut_diff_array[data_struct.NpData.POS][:, 1].max()
        frontend_diff_array = cut_diff_array[cut_diff_array[data_struct.NpData.POS][:, 1] == frontend_component]
        frontend_diff_list.append(frontend_diff_array)
    combined_frontend_diff_array = np.concatenate(frontend_diff_list)
    return combined_frontend_diff_array