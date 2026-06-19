import numpy as np
from scipy.interpolate import CubicSpline

from shellarc_railgun.data_model import data_struct

def get_cut_intercept(M: np.ndarray,
                      cut_num: int,
                      today: int
                      ) -> np.ndarray: 
    intercept_array = M[M[data_struct.NpData.POS][:, 0] == cut_num]
    sorted_intercept_array = np.argsort(intercept_array[:, 2])
    if today > sorted_intercept_array[data_struct.NpData.POS][-1, 2]:
        if sorted_intercept_array[data_struct.NpData.POS][-1, 1] < data_struct.FINENESS:
            next_component = sorted_intercept_array[data_struct.NpData.POS][-1, 1] + 1
        else:
            next_component = data_struct.FINENESS
        predicted_today = np.array([(cut_num, next_component, today), "Nil"], dtype=data_struct.point_dtype)
        sorted_intercept_array = np.append(sorted_intercept_array, predicted_today)
    return sorted_intercept_array

def get_cut_intercept_spline(sorted_intercept_array: np.ndarray,) -> CubicSpline:
    sorted_intercept_array = sorted_intercept_array[data_struct.NpData.POS]
    time_data = sorted_intercept_array[:, 2]
    component_data = sorted_intercept_array[:, 1]
    spline = CubicSpline(component_data, time_data)
    return spline

def get_cut_intercept_diff(M: np.ndarray,
                           cut_num: int,
                           today: int
                           ) -> np.ndarray:
    sorted_intercept_array = get_cut_intercept(
        M=M,
        cut_num=cut_num,
        today=today
    )
    spline = get_cut_intercept_spline(sorted_intercept_array=sorted_intercept_array)
    diff_array = np.zeros(sorted_intercept_array.shape[0], dtype=data_struct.diff_point_dtype)
    diff_array[data_struct.NpData.POS] = sorted_intercept_array[data_struct.NpData.POS]
    diff_array[data_struct.NpData.COMP] = sorted_intercept_array[data_struct.NpData.COMP]
    diff_array[data_struct.NpData.DIFF] =  1.0 / spline(sorted_intercept_array[data_struct.NpData.POS][:, 1], nu=1)
    return diff_array