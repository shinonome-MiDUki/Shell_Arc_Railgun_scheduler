from enum import StrEnum

import numpy as np

FINENESS = 10

class NpData(StrEnum):
    POS = "position"
    COMP = "component"
    OFFSET = "offset"
    DIFF = "differentiation"

point_dtype = np.dtype([
    (NpData.POS, 'i4', (3,)),  
    (NpData.COMP, 'U10')           
])

diff_point_dtype = np.dtype([
    (NpData.POS, 'i4', (3,)),
    (NpData.DIFF, 'f4', (1,)),
    (NpData.COMP, 'U10')  
])

z_offset_dtype = np.dtype([
    (NpData.OFFSET, 'i4', (1)),
    (NpData.POS, 'i4', (3,)),  
    (NpData.COMP, 'U10')  
])