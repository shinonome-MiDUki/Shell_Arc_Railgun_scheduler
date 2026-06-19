import numpy as np
from scipy.spatial import distance_matrix

from shellarc_railgun.data_model import data_struct
from shellarc_railgun.computation_core.z_offset import cone_equation

def relocate_points(M: np.ndarray, 
                    total_cut_num: int,
                    proj_length: int,
                    today: int,
                    ) -> np.ndarray:
    inverse_1_minus_t = 1 / (1 - cone_equation.proj_context_constant(
        total_cut_num=total_cut_num,
        proj_length=proj_length
    ))
    
    cut_axis_range = np.arange(1, total_cut_num + 1)
    component_axis_range = np.arange(0, cone_equation.FITNESS + 1)
    xx, yy = np.meshgrid(cut_axis_range, component_axis_range)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T 
    grid_r = np.sqrt(grid_coords[:, 0]**2 + grid_coords[:, 1]**2)
    
    grid_available = np.ones(len(grid_coords), dtype=bool)
    
    total_grid_size = len(grid_coords)
    result_array = np.empty(total_grid_size, dtype=data_struct.point_dtype)
    
    result_array[data_struct.NpData.POS][:, 0] = grid_coords[:, 0]
    result_array[data_struct.NpData.POS][:, 1] = grid_coords[:, 1]
    result_array[data_struct.NpData.POS][:, 2] = np.full(total_grid_size, today)
    result_array[data_struct.NpData.COMP] = "Nil"
    
    sorted_indices = np.argsort(M[data_struct.NpData.POS][2])
    sorted_points = M[sorted_indices]
    
    allowed_indices = set()
    
    start_idx = np.argmin(grid_r)
    allowed_indices.add(start_idx)
    
    neighbors_offset = np.array([[1,0], [-1,0], [0,1], [0,-1]])
    
    for pt in sorted_points:
        z_val = pt[data_struct.NpData.POS][2]
        comp_val = pt['component']
        
        target_r = z_val * inverse_1_minus_t
        
        # 現在配置可能な候補の中から、最も target_r に近いグリッドを選択
        candidate_idxs = list(allowed_indices)
        if not candidate_idxs:
            break # 配置可能場所が尽きた場合
            
        best_candidate_idx = candidate_idxs[np.argmin(np.abs(grid_r[candidate_idxs] - target_r))]
        
        # 決定した座標を取得
        chosen_x, chosen_y = grid_coords[best_candidate_idx]
        
        # 結果配列の該当グリッド位置にデータを書き込む
        result_array[best_candidate_idx] = (chosen_x, chosen_y, z_val, comp_val)
        
        # 使用済みフラグを立て、候補から外す
        grid_available[best_candidate_idx] = False
        allowed_indices.remove(best_candidate_idx)
        
        # 新しく配置した点の4近傍を調べ、有効な空きグリッドがあれば候補に加える
        current_coord = np.array([chosen_x, chosen_y])
        for offset in neighbors_offset:
            nb_coord = current_coord + offset
            # 範囲内かチェック
            if (1 <= nb_coord[0] <= total_cut_num) and (0 <= nb_coord[1] <= data_struct.FINENESS):
                # 対応するグリッドインデックスを取得
                nb_idx = np.where((grid_coords[:, 0] == nb_coord[0]) & (grid_coords[:, 1] == nb_coord[1]))[0][0]
                if grid_available[nb_idx]:
                    allowed_indices.add(nb_idx)
                    
    return result_array