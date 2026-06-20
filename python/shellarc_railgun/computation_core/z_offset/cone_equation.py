import numpy as np

def proj_context_constant(total_cut_num: int,
                          proj_length: int
                          ) -> float:
    ctx_constant = proj_length / ((total_cut_num ** 2) + 100) ** 0.5
    return ctx_constant

def compute_z_offset(cut: np.ndarray,
                     component: np.ndarray,
                     day: np.ndarray,
                     proj_ctx_constant: float,
                     today: int
                     ) -> np.ndarray:
    t = proj_ctx_constant
    ideal_day = (1 - t) * ((cut ** 2) + (component ** 2)) ** 0.5
    offset_array = np.where(ideal_day <= today, ideal_day - day, 0.0)
    return offset_array