import numpy as np

from shellarc_railgun.data_model import data_struct
from shellarc_railgun.computation_core.intercepts import cut_component_plane

def get_frontend_momentum(M: np.ndarray,
                          total_cut_num: int,
                          today: int
                          ) -> np.ndarray:
    frontend_diff_array = cut_component_plane.get_frontend_array(
        M=M,
        total_cut_num=total_cut_num,
        today=today
    )
    return frontend_diff_array