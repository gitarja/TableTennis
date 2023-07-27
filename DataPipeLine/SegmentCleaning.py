import numpy as np
from Utils.DataReader import ViconReader
from Utils.Lib import interExtrapolate
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pandas as pd
from Utils.Lib import checkMakeDir

pd.options.mode.chained_assignment = None  # default='warn'


class SegmentsCleaner:

    def cleanData(self, segment: np.array, condition: np.array, th: float = 2):
        def groupNoisySegment(v: np.array, within_th: int, len_condition: int):
            valley_groups, num_groups = label(np.diff(v) < within_th)
            index = []
            for i in np.unique(valley_groups)[1:]:
                valley_group = v[np.where(valley_groups == i)]
                start = np.min(valley_group)
                stop = np.max(valley_group)
                index.append(np.arange(start, stop + 1))

            conditions_def = np.zeros((len_condition)).astype(bool)
            if len(index) != 0:
                conditions_def[np.concatenate(index)] = True
                return conditions_def
            return np.array([])

        # segment condition
        d2_segment = np.abs(np.diff(segment, 2, axis=0))  # compute acceleration
        d2_segment = np.concatenate([np.zeros((2, 3)), d2_segment])

        segment_condition = np.sum(d2_segment > th, axis=-1) >= 1
        cleaned_percentage = 0
        if np.sum(segment_condition) != 0:
            segment_condition = groupNoisySegment(np.nonzero(segment_condition)[0], within_th=25,
                                                  len_condition=len(d2_segment))
            nan_mask = ~np.isnan(np.sum(segment, -1))
            mask = np.nonzero(nan_mask & (condition))[0]
            if len(segment_condition) != 0:
                mask = np.nonzero(nan_mask & (condition | segment_condition))[0]
            segment[mask, :] = np.nan
            cleaned_percentage = 100 * (len(mask) / len(segment))

        new_segment = np.array([interExtrapolate(segment[:, i]) for i in range(3)]).transpose()

        return new_segment, cleaned_percentage

    def tobiiCleanData(self, segment: np.array, trajectories: np.array, th: float = 2):
        LA = trajectories[:, 0:3]
        RA = trajectories[:, 3:6]
        LB = trajectories[:, 6:9]
        RB = trajectories[:, 9:12]
        LC = trajectories[:, 12:15]
        RC = trajectories[:, 15:]

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

    def wristCleaning(self, segment: np.array, trajectories: np.array):

        RWA = trajectories[:, 0:3]
        RWB = trajectories[:, 3:]

        dist = np.linalg.norm(RWA - RWB, axis=-1)

        mean_dist = np.nanmean(dist)

        nan_mask = ~np.isnan(np.sum(trajectories, -1))

        th_l = mean_dist - 10
        th_u = mean_dist + 10
        condition = (~((dist >= th_l) & (dist <= th_u)) & nan_mask)

        return self.cleanData(segment, condition)


if __name__ == '__main__':

    ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
    file_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\"
    result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
    ref_df = pd.read_csv(ref_file)
    single_df = ref_df.loc[ref_df.Trial_Type == "S"]
    double_df = ref_df.loc[ref_df.Trial_Type == "P"]
    double_df_unique = double_df.loc[double_df.Session_Code.drop_duplicates().index]

    for i, d in double_df_unique.iterrows():
        dates = d["Date"].replace(".", "-")
        session = d["Session"]
        trial = d["Trial"]

        folder_name = dates + "_" + session
        file_name = folder_name + "_" + trial

        if file_name == "2022-12-01_A_T06":
            file_session_path = file_path + folder_name + "\\"
            result_session_path = result_path + folder_name + "\\"

            checkMakeDir(result_session_path)
            reader = ViconReader()
            obj, sub, n = reader.extractData(file_session_path + file_name + ".csv", cleaning=True)
            cleaner = SegmentsCleaner()

            # try:
            for s in sub:
                # tobii cleaning
                tobii_segment = s["segments"].filter(regex='TobiiGlass_T').values
                tobii_trajectories = s["trajectories"].filter(regex='Tobii').values
                tobii_cs = np.copy(tobii_segment)
                tobii_ct = np.copy(tobii_trajectories)
                new_tobii_segment, tobii_cleaned_p = cleaner.tobiiCleanData(tobii_cs, tobii_ct, th=291)
                s["segments"].filter(regex='TobiiGlass_T').loc[:] = new_tobii_segment
                # right wirst cleaning
                rwrist_segment = s["segments"].filter(regex='R_Wrist_T').values
                rwrist_trajectories = s["trajectories"].filter(regex='(RWRA|RWRB)').values
                rwrist_cs = np.copy(rwrist_segment)
                rwrist_ct = np.copy(rwrist_trajectories)
                new_rwrist_segment, rwirst_cleaned_p = cleaner.wristCleaning(rwrist_cs, rwrist_ct)
                s["segments"].filter(regex='R_Wrist_T').loc[:] = new_rwrist_segment
                # left wirst cleaning
                start_idx = (6 * 6) + 3
                lwrist_segment = s["segments"].filter(regex='L_Wrist_T').values
                lwrist_trajectories = s["trajectories"].filter(regex='(LWRA|LWRB)').values
                lwrist_cs = np.copy(lwrist_segment)
                lwrist_ct = np.copy(lwrist_trajectories)
                new_lwrist_segment, lwirst_cleaned_p = cleaner.wristCleaning(lwrist_cs, lwrist_ct)
                s["segments"].filter(regex='L_Wrist_T').loc[:] = new_lwrist_segment

                print(
                    "%s, %s, %f, %f, %f" % (file_name, s["name"], tobii_cleaned_p, rwirst_cleaned_p, lwirst_cleaned_p))

            import pickle

            data = [obj, sub]

            with open(result_session_path + "\\" + file_name + ".pkl", 'wb') as f:
                pickle.dump(data, f)

        # except:
        #     print("Error: " + file_name)
