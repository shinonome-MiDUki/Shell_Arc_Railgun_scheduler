import numpy as np
from scipy.interpolate import CubicSpline

from shellarc_railgun.data_model import data_struct

def get_cut_intercept(M: np.ndarray,
                      cut_num: int,
                      today: int
                      ) -> np.ndarray: 
    intercept_array = M[M[data_struct.NpData.POS][:, 0] == cut_num]
    sort_indices = np.argsort(intercept_array[data_struct.NpData.POS][:, 2])
    sorted_intercept_array = intercept_array[sort_indices]
    if len(sorted_intercept_array[data_struct.NpData.POS]) < 1:
        return np.array([], dtype=data_struct.point_dtype)
    if today > sorted_intercept_array[data_struct.NpData.POS][-1, 2]:
        if sorted_intercept_array[data_struct.NpData.POS][-1, 1] < data_struct.FINENESS:
            next_component = sorted_intercept_array[data_struct.NpData.POS][-1, 1] + 1
        else:
            next_component = data_struct.FINENESS
        predicted_today = np.array([((cut_num, next_component, today), "Nil")], dtype=data_struct.point_dtype)
        sorted_intercept_array = np.append(sorted_intercept_array, predicted_today)
    return sorted_intercept_array

def get_cut_intercept_spline(sorted_intercept_array: np.ndarray,) -> CubicSpline | None:
    # POS データの切り出し (y, z) ではなく (x, y, z) の想定
    pos_data = sorted_intercept_array[data_struct.NpData.POS]

    # 列の割り当て (2列目が time, 1列目が component)
    time_data = pos_data[:, 2]
    component_data = pos_data[:, 1]

    # --- エラー対策1: 重複を排除し、x軸(component)が「厳密に増加」するようにソート ---
    # 同一の component_data があると CubicSpline はエラーになるため、平均値等で一意にするか、
    # 簡易的には一意な x に対して y をマッピングします。
    unique_components, indices = np.unique(component_data, return_index=True)

    # --- エラー対策2: ユニークな点数が2点以上あるかチェック ---
    if len(unique_components) < 2:
        # 呼び出し元で None チェックを行ってください
        return None

    # unique_components は自動的に昇順（小さい順）ソートされているので、
    # time_data もそれに対応するインデックスで抽出します
    unique_time_data = time_data[indices]

    # 安全にスプラインを作成
    spline = CubicSpline(unique_components, unique_time_data)
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
    try:
        deriv = spline(sorted_intercept_array[data_struct.NpData.POS][:, 1], nu=1)
        diff_array[data_struct.NpData.DIFF] = np.where(
            np.abs(deriv) > 1e-9,
            1.0 / deriv,
            0.0  
        ).reshape(-1,1)
    except:
        diff_array[data_struct.NpData.DIFF] = 0.0
    print("@@@@@@@@@@@@@@@@")
    print(diff_array)
    print("@@@@@@@@@@@@@@@@")
    return diff_array