from enum import StrEnum

from numpy import np

FINENESS = 10

class NpData(StrEnum):
    POS = "position"
    COMP = "component"
    OFFSET = "offset"

point_dtype = np.dtype([
    (NpData.POS, 'i4', (3)),  
    (NpData.COMP, 'U10')           
])

z_offset_dtype = np.dtype([
    (NpData.OFFSET, 'i4', (1)),
    (NpData.POS, 'i4', (3)),  
    (NpData.COMP, 'U10')  
])