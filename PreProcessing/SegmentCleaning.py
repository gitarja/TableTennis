import numpy as np
from Utils.DataReader import ViconReader
from Utils.Lib import interExtrapolate
import matplotlib.pyplot as plt
from scipy.ndimage import label
class SegmentsCleaner:

    def cleanData(self, segment:np.array, condition:np.array, th:float=2):
        def groupNoisySegment(v: np.array, within_th: int, len_condition:int):
            valley_groups, num_groups = label(np.diff(v) < within_th)
            index = []
            for i in np.unique(valley_groups)[1:]:
                valley_group = v[np.where(valley_groups == i)]
                start = np.min(valley_group)
                stop = np.max(valley_group)
                index.append(np.arange(start, stop+1))

            conditions_def = np.zeros((len_condition)).astype(bool)
            conditions_def[np.concatenate(index)] = True
            return conditions_def
        # segment condition
        d2_segment = np.abs(np.diff(segment, 2, axis=0))  # compute acceleration
        d2_segment = np.concatenate([np.zeros((2, 3)), d2_segment])


        segment_condition = np.sum(d2_segment > th, axis=-1) >= 1
        segment_condition = groupNoisySegment(np.nonzero(segment_condition)[0], within_th=25, len_condition=len(d2_segment))
        nan_mask = ~np.isnan(np.sum(segment, -1))
        mask = np.nonzero(nan_mask & (condition | segment_condition))[0]
        segment[mask, :] = np.nan
        print(len(mask) / len(segment))
        new_segment =  np.array([interExtrapolate(segment[:, i]) for i in range(3)]).transpose()

        return new_segment


    def tobiiCleanData(self, segment:np.array, trajectories: np.array, th:float=2):
        LA = trajectories[:, 0:3]
        RA = trajectories[:,3:6]
        LB = trajectories[:,6:9]
        RB = trajectories[:,9:12]
        LC = trajectories[:,12:15]
        RC = trajectories[:,15:]

        # distance L
        LAB_dist = np.linalg.norm(LA - LB, axis=-1)
        LBC_dist = np.linalg.norm(LB - LC, axis=-1)
        LAC_dist = np.linalg.norm(LA - LC, axis=-1)
        L_dist = LAB_dist + LBC_dist + LAC_dist


        # distance R
        RAB_dist = np.linalg.norm(RA - RB, axis=-1)
        RBC_dist = np.linalg.norm(RB - RC, axis=-1)
        RAC_dist = np.linalg.norm(RA - RC, axis=-1)
        R_dist = RAB_dist + RBC_dist + RAC_dist

        dist = L_dist + R_dist

        nan_mask = ~np.isnan(np.sum(trajectories, -1))

        th_l = th - 7
        th_u = th + 7
        condition = (~((dist >= th_l) & (dist <= th_u)) & nan_mask)

        return self.cleanData(segment, condition)


    def wristCleaning(self, segment:np.array, trajectories: np.array):

        RWA = trajectories[:, 0:3]
        RWB = trajectories[:, 3:]

        dist = np.linalg.norm(RWA - RWB, axis=-1)

        mean_dist = np.nanmean(dist)

        nan_mask = ~np.isnan(np.sum(trajectories, -1))

        th_l = mean_dist - 10
        th_u = mean_dist + 10
        condition = (~((dist >= th_l) & (dist <= th_u)) & nan_mask)

        return self.cleanData(segment, condition)



result_path = "F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.08\\Compiled\\"
reader = ViconReader()
obj, sub = reader.extractData(result_path + "T02_Nexus.csv", cleaning=True)
cleaner = SegmentsCleaner()
for s in sub:

    print(s["name"])

    # tobii cleaning
    start_idx = (18 * 6) + 3
    tobii_segment = s["segments"][:, start_idx:start_idx + 3]
    tobii_trajectories = s["trajectories"][:, -18:]
    tobii_cs = np.copy(tobii_segment)
    tobii_ct = np.copy(tobii_trajectories)
    new_tobii_segment = cleaner.tobiiCleanData(tobii_cs, tobii_ct, th=291)
    s["segments"][:, start_idx:start_idx + 3] = new_tobii_segment
    # right wirst cleaning
    start_idx = (15 * 6) + 3
    rwrist_segment = s["segments"][:, start_idx:start_idx + 3]
    rwrist_trajectories = s["trajectories"][:, (16 * 3): (18 * 3)]
    rwrist_cs = np.copy(rwrist_segment)
    rwrist_ct = np.copy(rwrist_trajectories)
    new_rwrist_segment = cleaner.wristCleaning(rwrist_cs, rwrist_ct)
    s["segments"][:, start_idx:start_idx + 3] = new_rwrist_segment



#
# import pickle
# data = [obj, sub]
#
# with open(result_path + 'T02_Nexus.pkl', 'wb') as f:
#     pickle.dump(data, f)
