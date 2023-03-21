import numpy as np
from Utils.DataReader import ViconReader
from Utils.Lib import interExtrapolate
class SegmentsCleaner:

    def cleanData(self, segment:np.array, th:float=2):
        nan_mask = ~np.isnan(np.sum(segment, -1))
        d2_segment = np.abs(np.diff(segment, 2, axis=0))
        d2_segment = np.concatenate([np.zeros((2, 3)), d2_segment])
        condition = np.sum(d2_segment > th, axis=-1)
        mask = np.nonzero(nan_mask & (condition> 1))

        segment[mask, :] = np.nan

        new_segment =  np.array([interExtrapolate(segment[:, i]) for i in range(3)]).transpose()

        return new_segment



result_path = "F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.08\\Compiled\\"
reader = ViconReader()
obj, sub = reader.extractData(result_path + "T02_Nexus.csv")
cleaner = SegmentsCleaner()
for s in sub:
    tobii_segment = s["segments"][:, -3:]
    ts = np.copy(tobii_segment)
    new_tobii_segment = cleaner.cleanData(tobii_segment)

    s["segments"][:, -3:] = new_tobii_segment

import pickle
data = [obj, sub]

with open(result_path + 'T02_Nexus.pkl', 'wb') as f:
    pickle.dump(data, f)
