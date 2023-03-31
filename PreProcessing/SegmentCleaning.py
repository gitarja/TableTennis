import numpy as np
from Utils.DataReader import ViconReader
from Utils.Lib import interExtrapolate
import matplotlib.pyplot as plt
class SegmentsCleaner:

    def cleanData(self, segment:np.array, th:float=2):
        nan_mask = ~np.isnan(np.sum(segment, -1))
        d2_segment = np.abs(np.diff(segment, 2, axis=0)) # compute acceleration
        d2_segment = np.concatenate([np.zeros((2, 3)), d2_segment])
        condition = np.sum(d2_segment > th, axis=-1)
        mask = np.nonzero(nan_mask & (condition> 1))

        segment[mask, :] = np.nan

        new_segment =  np.array([interExtrapolate(segment[:, i]) for i in range(3)]).transpose()

        return new_segment



result_path = "F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.21\\"
reader = ViconReader()
obj, sub = reader.extractData(result_path + "T02_Nexus.csv", cleaning=True)
cleaner = SegmentsCleaner()
for s in sub:
    for i in range(19):
        start_idx = (i*6)+ 3
        segment = s["segments"][:, start_idx:start_idx+3]
        ts = np.copy(segment)
        new_tobii_segment = cleaner.cleanData(ts, th=2)

        s["segments"][:, start_idx:start_idx+3] = new_tobii_segment

        # plt.plot(np.sum(np.abs(np.diff(ts, 2, axis=0)) > 3, axis=-1))
        # plt.show()
        # print(segment.shape)
    # ts = np.copy(tobii_segment)
    # new_tobii_segment = cleaner.cleanData(tobii_segment, th=2)

    # s["segments"][:, -3:] = new_tobii_segment

import pickle
data = [obj, sub]

with open(result_path + 'T02_Nexus.pkl', 'wb') as f:
    pickle.dump(data, f)
