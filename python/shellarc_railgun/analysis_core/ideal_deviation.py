import numpy as np

from shellarc_railgun.computation_core.z_offset import z_offset
from shellarc_railgun.data_model import data_struct

def get_deviation_mean(M: np.ndarray,
                       total_cut_num: int,
                       proj_length: int,
                       today: int
                       ) -> float:
    offset_array = z_offset.get_z_offset_array(
        M=M,
        total_cut_num=total_cut_num,
        proj_length=proj_length,
        today=today
    )
    mean = np.mean(offset_array[data_struct.NpData.OFFSET])
    return mean

def get_deviation_sd(M: np.ndarray,
                     total_cut_num: int,
                     proj_length: int,
                     today: int
                     ) -> float:
    offset_array = z_offset.get_z_offset_array(
        M=M,
        total_cut_num=total_cut_num,
        proj_length=proj_length,
        today=today
    )
    sd = np.std(offset_array[data_struct.NpData.OFFSET])
    return sd
