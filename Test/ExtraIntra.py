from filterpy.kalman import KalmanFilter
import  numpy as np

class ExtraIntraKalman:

    def __init__(self):

        self.kalman = KalmanFilter(dim_z=3, dim_x=3)


    def interExtrapolate(self, data):
        # copy data
        dup_data = np.copy(data)
        # nan masking
        bool_mask = np.isnan(data)

        for i in range(0, len(data), n):
            mask = bool_mask[i:i + n]

