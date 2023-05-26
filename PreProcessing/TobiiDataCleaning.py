import pandas as pd
import math
import numpy as np
from Utils.DataReader import SubjectObjectReader


class TobiiCleaner:

    def __init__(self, ref_path):
        self.ref_file = pd.read_csv(ref_path)

    def matchData(self, tobii_path, vicon=None):
        data = pd.read_csv(tobii_path, delimiter="\t")

        file_name = tobii_path.split("\\")[-1].split(" ")
        date = file_name[0].split("-")[0]
        trial = file_name[1].split("_")[0]
        participant = file_name[1].split("_")[-1].split(".tsv")[0]

        ref = self.ref_file[(self.ref_file["Date"] == date) & (self.ref_file["Trial"] == trial) & (
                self.ref_file["Participants"] == participant)]

        start = ref.Start_TTL.values[0] * 1000
        stop = ref.Stop_TTL.values[0] * 1000
        vicon_frame = ref.Vicon_Frame.values[0]

        cut_data = data[
            (data["Recording timestamp"] >= start) & (data["Recording timestamp"] <= stop) & (
                    data["Sensor"] == "Eye Tracker")]
        print(100 * len(cut_data) / vicon_frame)
        # print(stop - start)
        # print(len(cut_data))
        # print(vicon_frame)
        selected_columns = ["Recording timestamp",
                            "Gaze point X", "Gaze point Y",
                            "Gaze point 3D X", "Gaze point 3D Y", "Gaze point 3D Z",
                            "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
                            "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
                            "Pupil position left X", "Pupil position left Y", "Pupil position left Z",
                            "Pupil position right X", "Pupil position right Y", "Pupil position right Z"]

        # cut_data = cut_data.dropna(subset=["Gaze direction left X"])
        # print( 100 * np.average(np.isnan(cut_data["Gaze direction left X"].values)))

        columns = ["Timestamp",
                   "Gaze_point_X", "Gaze_point_Y",
                   "Gaze_point_3D_X", "Gaze_point_3D_Y", "Gaze_point_3D_Z",
                   "Gaze_direction_left_X", "Gaze_direction_left_Y", "Gaze_direction_left_Z",
                   "Gaze_direction_right X", "Gaze_direction_right_Y", "Gaze_direction_right_Z",
                   "Pupil_position_left_X", "Pupil_position_left_Y", "Pupil_position_left_Z",
                   "Pupil_position_right_X", "Pupil_position_right_Y", "Pupil_position_right_Z"]



        temp_data = np.empty((vicon_frame, len(columns)))
        tobii_df = pd.DataFrame(data=temp_data, columns=columns)

        sp = cut_data.iloc[1]["Recording timestamp"] - cut_data.iloc[0]["Recording timestamp"]
        if sp < 15:
            cut_data = cut_data.iloc[::2]
        start = start + np.arange(vicon_frame) * 10
        valid = (cut_data["Validity left"].values == "Valid") | (cut_data["Validity right"].values == "Valid")
        dist_mat = np.abs(np.expand_dims(cut_data["Recording timestamp"], 1) - np.expand_dims(start, axis=0))
        closest_points = np.nanmin(dist_mat, axis=-1)
        closest_idx = np.nanargmin(dist_mat, axis=-1)
        closest_idx = closest_idx - np.min(closest_idx)
        selected_idx = np.argwhere((closest_points <= 15) & (valid))[:, 0]

        tobii_df.iloc[int(ref.Offset.values) + closest_idx[selected_idx]] = cut_data.iloc[selected_idx][selected_columns]

        print(100 * (np.sum(tobii_df.Timestamp.values != 0) / vicon_frame))
        return participant, tobii_df


if __name__ == '__main__':
    import glob
    import pickle

    folder_name = "2022-11-08_A"
    file_name = "2022-11-08_A_T03"

    # folder_name = "2023-02-08_A"
    # file_name = "2023-02-08_A_T04"
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
    reader = SubjectObjectReader()
    obj, sub, ball = reader.extractData(
        path + folder_name + "\\" + file_name + "_wb.pkl")

    ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"

    reader = TobiiCleaner(ref_file)
    tobii_filename = file_name.split("_")[-1] + "_" + sub[0]["name"] + ".tsv"
    tobii_files = glob.glob(path + folder_name + "\\Tobii\\*" + tobii_filename + "")

    if len(tobii_files) == 1:
        participant, tobii_df = reader.matchData(tobii_files[0])

        data = [obj, sub, ball, [{"name": participant, "trajectories": tobii_df}]]

        with open(
                path + folder_name + "\\" + file_name + "_complete.pkl",
                'wb') as f:
            pickle.dump(data, f)
