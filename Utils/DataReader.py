import csv
import numpy as np
import pandas as pd
import math
import time
import datetime
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
import transforms3d
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline


from numpy.lib import stride_tricks


import ezc3d

class Object:
    def __init__(self):
        self.joints = []
        self.segments = []
        self.trajectories = []



class TobiiReader:

    def local2GlobalGaze(self, gaze, tobii_seg, tobii_rot):

        rads = np.deg2rad(tobii_rot)
        R_m = np.array([transforms3d.axangles.axangle2mat(r, np.sqrt(np.sum(r ** 2))) for r in rads])

        gaze_global = np.squeeze(np.matmul(R_m, np.expand_dims(gaze, 2)))


        return gaze_global + tobii_seg

    def interPolate(self, g):
        mask = np.isnan(g)
        if np.sum(mask) > 0:
            g[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), g[~mask])
        return g

    def cutData(self, data:pd.DataFrame = None):
        # data = data[(data.Sensor != data.Sensor) | (data.Sensor == "Eye Tracker") | (data.Sensor == "Gyroscope")]
        data = data[(data.Sensor != data.Sensor) | (data.Sensor == "Eye Tracker")]

        #Get start point
        low_start = np.argwhere(data["Event"].values == "SyncPortInLow")
        high_start = np.argwhere(data["Event"].values == "SyncPortInHigh")

        def minPositive(a, b):
            c = a - b
            d = c[c>=0]
            e = np.min(d)
            return  (e > 0) & (e < 2)

        low_high_idx =  np.array([low_start[i] for i in range(len(low_start)) if  minPositive(high_start, low_start[i])])[:, 0]
        all_sync_in = np.diff(data.iloc[low_high_idx]["Recording timestamp"].values)
        start_stop_idx = np.argwhere((all_sync_in >= 200000) & (all_sync_in <= 200500))


        start_events_idx = low_high_idx[start_stop_idx[0]][0]
        stop_events_idx = low_high_idx[start_stop_idx[-1]][0]

        data_cut = data.iloc[start_events_idx:stop_events_idx] # take the data from the start event
        start_time = datetime.datetime.strptime(data_cut.iloc[0]["Recording date"] + " " + data_cut.iloc[0]["Recording start time"], "%m/%d/%Y %H:%M:%S.%f").timestamp()
        data_cut.loc[:]["Timestamp"] = start_time + (data_cut["Recording timestamp"].values / 1e+6)# timestamp is in microsecond

        eye_data = data_cut[(data_cut.Sensor != data_cut.Sensor) | (data_cut.Sensor == "Eye Tracker")]
        # gyro_data = data[(data.Sensor == "Gyroscope")]
        return eye_data


    def normalizeData(self, eye_data:list):
        # low_index = [np.argwhere(d["Event"].values == "SyncPortInLow") for d in eye_data]
        #
        # end_point_idx = np.max([lw[np.argwhere(lw > 10000)[0][0]] for lw in low_index])
        # end_point_sub_idx = np.argmax([lw[np.argwhere(lw > 10000)[0][0]] for lw in low_index])




        clean_ref_data = eye_data[end_point_sub_idx][:end_point_idx]

        eye_data = [e[e.Sensor == "Eye Tracker"]for e in eye_data]
        # selected columns
        # selected_columns = [
        #                     "Gaze point X", "Gaze point Y",
        #                     "Gaze point 3D X", "Gaze point 3D Y",
        #                     "Gaze point 3D Z", "Gaze direction left X",
        #                     "Gaze direction left Y", "Gaze direction left Z",
        #                     "Gaze direction right X","Gaze direction right Y",
        #                     "Gaze direction right Z"]

        selected_columns = [
            "Gaze point X", "Gaze point Y",
            "Gaze point 3D X", "Gaze point 3D Y",   "Gaze point 3D Z",
            "Pupil position left X","Pupil position left Y", "Pupil position left Z",
            "Pupil position right X", "Pupil position right Y", "Pupil position right Z"]

        # gyro_selected_columns = ["Gyro X",
        #     "Gyro Z", "Gyro Y"]

        time_stamp = []
        gaze_info = []
        gyro_info = []
        for idx in range(len(clean_ref_data) - 1):
            start_time = clean_ref_data.iloc[idx]["Timestamp"]
            end_time =  clean_ref_data.iloc[idx]["Timestamp"] + .2
            time_stamp.append(start_time)
            gaze_info.append([d[(d["Timestamp"] >= start_time) & (d["Timestamp"] <= start_time + .1)][selected_columns].iloc[0].values for d in eye_data])

        gaze_info = np.array(gaze_info).transpose((1, 2, 0))

        for i in range(len(eye_data)):
            for j in range(len(selected_columns)):
                gaze_info[i][j] = self.interPolate(gaze_info[i][j])


        clean_gaze = gaze_info.transpose((2, 0, 1)).tolist()

        clean_data = pd.DataFrame({"Timestamp": time_stamp, "Gaze": clean_gaze})

        return clean_data


    def extractData(self, file_paths:list = None):
        data = []

        for f in file_paths:
            d = self.cutData(pd.read_csv(f, delimiter="\t"))
            data.append(d)


        data = self.normalizeData(data)
        return data


class ECGReader:

    def interPolate(self, rr):
        rr[rr>1000] = np.nan
        mask = np.isnan(rr)
        if np.sum(mask)>0:
            rr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), rr[~mask])
        return rr


    def normalizeData(self, data):

        # get the start point
        start_points = [d.loc[0]["Timestamp"] for d in data]
        earliest_start_idx = np.argmin(start_points)


        # get ref data and remove duplicate data
        ref_data = data[earliest_start_idx]
        time_stamp_diff = np.diff(ref_data["Timestamp"])
        clean_ref_data = ref_data.iloc[:-1][time_stamp_diff>=0.5]

        time_stamp = []
        rr = []
        for idx in range(len(clean_ref_data)-1):
            start_time = math.floor(clean_ref_data.iloc[idx]["Timestamp"])
            end_time = start_time + 1.5 # plus 1.5 seconds
            time_stamp.append(start_time)
            rr.append([np.max(d[(d["Timestamp"] >= start_time) & (d["Timestamp"] <= end_time)]["RR"]) for d in data])

        rr = np.array(rr).T
        clean_rr = np.array([self.interPolate(rr_item).tolist() for rr_item in rr]).T.tolist()
        clean_data = pd.DataFrame({"Timestamp":time_stamp, "RR": clean_rr})

        return clean_data

    def extractData(self, file_paths:list = None):
        data = []
        for f in file_paths:
            data.append(pd.read_csv(f))

        data = self.normalizeData(data)
        return data

class ViconReader:
    N_HEADER = 3
    N_TITLE = 2
    N_SPACE = 1
    NON_SUBJECTS = ["Racket1a", "Racket1", "Racket2", "Table", "Wall"]

    SEGMENTS_IDX = [
        "L_Collar",
        "L_Elbow",
        "L_Femur",
        "L_Foot",
        "L_Humerus",
        "L_Tibia",
        "L_Wrist",
        "L_Wrist_End",
        "LowerBack",
        "R_Collar",
        "R_Elbow",
        "R_Femur",
        "R_Foot",
        "R_Humerus",
        "R_Tibia",
        "R_Wrist",
        "R_Wrist_End",
        "Root",
        "TobiiGlass"
    ]

    SEGMENTS_PAIR =[
        # Left: upper part body
        [SEGMENTS_IDX.index("L_Collar"), SEGMENTS_IDX.index("L_Humerus")],
        [SEGMENTS_IDX.index("L_Humerus"), SEGMENTS_IDX.index("L_Elbow")],
        [SEGMENTS_IDX.index("L_Elbow"), SEGMENTS_IDX.index("L_Wrist")],
        [SEGMENTS_IDX.index("L_Wrist"), SEGMENTS_IDX.index("L_Wrist_End")],

        # Left: lower part body
        [SEGMENTS_IDX.index("Root"), SEGMENTS_IDX.index("L_Femur")],
        [SEGMENTS_IDX.index("L_Femur"), SEGMENTS_IDX.index("L_Tibia")],
        [SEGMENTS_IDX.index("L_Tibia"), SEGMENTS_IDX.index("L_Foot")],

        # center
        [SEGMENTS_IDX.index("LowerBack"), SEGMENTS_IDX.index("TobiiGlass")],
        [SEGMENTS_IDX.index("LowerBack"), SEGMENTS_IDX.index("Root")],

        # Right: upper part body
        [SEGMENTS_IDX.index("R_Collar"), SEGMENTS_IDX.index("R_Humerus")],
        [SEGMENTS_IDX.index("R_Humerus"), SEGMENTS_IDX.index("R_Elbow")],
        [SEGMENTS_IDX.index("R_Elbow"), SEGMENTS_IDX.index("R_Wrist")],
        [SEGMENTS_IDX.index("R_Wrist"), SEGMENTS_IDX.index("R_Wrist_End")],

        # Right: lower part body
        [SEGMENTS_IDX.index("Root"), SEGMENTS_IDX.index("R_Femur")],
        [SEGMENTS_IDX.index("R_Femur"), SEGMENTS_IDX.index("R_Tibia")],
        [SEGMENTS_IDX.index("R_Tibia"), SEGMENTS_IDX.index("R_Foot")],

    ]

    def constructSegments(self, segments, segment_pair, segment_idx):
        '''
        :param segments: (N, 6 * 19 segments)
        :return:
        '''


        position_idx = [(np.array([3, 4, 5]) + (3 * i*2)).tolist() for i in range(len(segment_idx))]

        position_data = segments[:, position_idx]

        human_segments = position_data[:, segment_pair, :]
        N, n_pair, _, _ = human_segments.shape



        return human_segments.reshape((N, n_pair*2, 3))

    def getSegmentsRot(self, segments):
        degree_idx = [(np.array([0, 1, 2]) + (3 * i * 2)).tolist() for i in range(19)]

        rotation_data = segments[:, degree_idx]

        return rotation_data

    def createArray(self, arr:list):
        '''
        :param arr: a list containing float and space values
        :return: a float array
        '''
        arr = np.array(arr)
        arr[arr == ''] = np.nan
        return arr.astype(float)


    def getObjectsName(self, header:list):
        a = list(filter(None, header)) # remove empty value
        b = [x.split(":")[0] for x in a ]

        return list(set(b))

    def augmentHeader(self, header1:list, header2:list):
        header1_c = []
        v = ""
        for h in header1:
            if h != "":
                v = h
            header1_c.append(v)

        a = ["_".join(x) for x in zip(header1_c, header2)]

        return a

    def getIndexOf(self, arr:list,obj_name:str):
        return [i for i, x in enumerate(arr) if obj_name in x]

    def extractData(self, file_path: str = None):
        with open(file_path, mode='r', encoding='utf-8-sig') as file:

            csv_reader = list(csv.reader(file))

            '''
            data structure
                - title
                - header
                - data
                - space (blank)
                - title
                - ..
            
            n_data      = total_data / 3
            total data  = N - (n_header * 3 + n_title * 3 + n_space * 3)
            n_header    = 3 (joints, segments, trajectories)
            n_title     = 2 (joints, segments, trajectories)
            n_space     = 1
            '''
            csv_reader = csv_reader[1:]
            n_data = int((len(csv_reader) - (self.N_HEADER * 3 + self.N_TITLE * 3 + self.N_SPACE * 3)) / 3)
            # joints
            i_joint = self.N_HEADER + self.N_TITLE
            joints_header = csv_reader[self.N_HEADER- 1:self.N_HEADER+1]
            joints_data = self.createArray(csv_reader[i_joint:i_joint+n_data])
            aug_header_joints = self.augmentHeader(joints_header[0], joints_header[1])

            # get objects name
            obj_names = self.getObjectsName(joints_header[0])

            # segments
            i_segment = 2 * (self.N_HEADER + self.N_TITLE) + self.N_SPACE + n_data
            segments_header = csv_reader[i_segment-self.N_HEADER:i_segment-self.N_HEADER+2]
            segments_data = self.createArray(csv_reader[i_segment:i_segment+n_data])
            aug_header_segments = self.augmentHeader(segments_header[0], segments_header[1])

            # trajectories
            i_trajectories = 3 * (self.N_HEADER + self.N_TITLE) + 2 *  (self.N_SPACE +  n_data)
            trajectories_header =  csv_reader[i_trajectories-self.N_HEADER:i_trajectories-self.N_HEADER+2]
            trajectories_data = self.createArray(csv_reader[i_trajectories:i_trajectories+n_data])
            aug_header_trajectories = self.augmentHeader(trajectories_header[0], trajectories_header[1])


            results_sub = []
            results_ob = []
            for obj in obj_names:
                # get joints
                idx_joints = self.getIndexOf(aug_header_joints, obj)
                obj_joints = joints_data[:, idx_joints]

                # get segments
                idx_segments = self.getIndexOf(aug_header_segments, obj)
                obj_segments = segments_data[:, idx_segments]

                # get trajectories
                idx_trajectories = self.getIndexOf(aug_header_trajectories, obj)
                obj_trajectories = trajectories_data[:, idx_trajectories]
                # if len(obj_segments[0]) > 50:
                #     self.constructSegments(obj_segments)
                new_inpt = {"name": obj, "joints":obj_joints, "segments": obj_segments, "trajectories": obj_trajectories}
                if obj in self.NON_SUBJECTS:
                    results_ob.append(new_inpt)
                else:
                    results_sub.append(new_inpt)


            return results_ob, results_sub




class C3dReader(object):

    def __init__(self, obj, subject):
        # relocated table
        self.ball_area = np.array([
            [-749.966797, 117.712341, 726.281189],  # table pt1_x - 60, table pt1_y - 400, table pt1_z
            [817.196533, 104.012634, 746.193665],  # table pt4_x  - 60, table pt4_y - 400, table pt4_z
            [-749.386292, 1860.348145, 739.174377],  # table pt3_x, table pt3_y + 600, table pt3_z
            [814.946838, 1860.348145, 739.174377],  # table pt2_x, table pt2_y + 600, table pt2_z

            [-749.966797, 217.712341, 2036.201416],  # table pt1_x  - 60, table pt1_y, table pt1_z * 2
            [817.196533, 204.012634, 2036.201416],  # table pt4_x  - 60, table pt4_y, table pt4_z * 2
            [-690.061218, 1947.592773, 2036.201416],  # wall pt4_x, wall pt4_y, wall pt4_z + 400
            [877.275452, 1930.623779, 2036.201416],  # wall pt1_x, wall pt1_y, wall pt1_z + 400

        ])









    def extractData(self, file_path: str = None):
        data = ezc3d.c3d(file_path)
        labels = data['parameters']['POINT']['LABELS']['value']
        tobii_idx_L = [i for i in range(len(labels)) if "LTobiiA" in labels[i]]
        tobii_idx_R = [i for i in range(len(labels)) if "RTobiiA" in labels[i]]
        unlabeled_idx =  [ i for i in range(len(labels)) if "*" in labels[i]] # the column label of the unlabelled marker starts with *
        data_points = np.array(data['data']['points'])

        unlabelled_data = data_points[0:3, unlabeled_idx, :]
        tobii_data = 0.5 * (data_points[0:3, tobii_idx_L, :] + data_points[0:3, tobii_idx_R, :])
        normalized_data = self.normalizeData(unlabelled_data, tobii_data)
        normalized_data = np.array([self.avgSmoothing(normalized_data[:, i], n=11) for i in range(3)]).transpose()

        # ori_data = np.copy(normalized_data)


        # interpolate_data = np.array([self.interPolate(normalized_data[:, i], n=100, th=0.83) for i in range(3)]).transpose()
        # sanity_mask = self.maskInterpolateSanity(normalized_data, n=30, th=0.2)
        # normalized_data[sanity_mask, :] = interpolate_data[sanity_mask, :]

        # normalized_data = np.array([self.avgSmoothing(normalized_data[:, i], n=11) for i in range(3)]).transpose()
        return normalized_data
        # normalized_data = np.array([self.interPolate(normalized_data[:, i]) for i in range(3)])
        # return normalized_data.transpose((1, 0))

if __name__ == '__main__':



    # reader = ECGReader()
    # ecg_files = ["F:\\users\\prasetia\\data\\TableTennis\\Pilot\\27-07-2022\\ECG\\rp-0\\20220727_095702516448.csv",
    #              "F:\\users\\prasetia\\data\\TableTennis\\Pilot\\27-07-2022\\ECG\\rp-1\\20220727_095702413999.csv"]
    # data = reader.extractData(ecg_files)
    # print(data)

    # reader = TobiiReader()
    # ecg_files = ["F:\\users\\prasetia\\data\\TableTennis\\Experiments\\08.11.2022-afternoon\\Tobii\\Data Export - 08-11-2022-afternoo\\08-11-2022-afternoo Recording 1 (2).tsv",
    #              "F:\\users\\prasetia\\data\\TableTennis\\Experiments\\08.11.2022-afternoon\\Tobii\\Data Export - 08-11-2022-afternoo\\08-11-2022-afternoo Recording 1 (3).tsv"]
    # data = reader.extractData(ecg_files)
    # print(data)


    reader = ViconReader()
    obj, sub = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Test\\T01.csv")


    reader = C3dReader(obj, sub)
    data = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Test\\T01.c3d")
    # data = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Pilot\\Ball\\BallTest08.c3d")
