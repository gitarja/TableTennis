import pandas as pd
import math
import numpy as np
from Utils.DataReader import SubjectObjectReader


def eventToNumeric(x_0, x):
    if x == 'Fixation':
        return 2
    elif x == 'Saccade':
        return 1
    elif x == 0:
        return x_0
    else:
        return 3


class TobiiCleaner:
    # eye event labels
    eye_labels = {"Fixation": 1, "Saccade": 2}

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

        # print(start)
        # start = start - (int(ref.Offset.values[0]) * 10)
        # stop  = stop + (int(ref.Offset.values[0]) * 10)
        cut_data = data[
            (data["Recording timestamp"] >= start) & (data["Recording timestamp"] <= stop) & (
                    data["Sensor"] == "Eye Tracker")]
        # print(100 * len(cut_data) / vicon_frame)
        # print(stop - start)
        # print(len(cut_data))
        # print(vicon_frame)
        selected_columns = ["Recording timestamp",
                            "Gaze point X", "Gaze point Y",
                            "Gaze point 3D X", "Gaze point 3D Y", "Gaze point 3D Z",
                            "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
                            "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
                            "Pupil position left X", "Pupil position left Y", "Pupil position left Z",
                            "Pupil position right X", "Pupil position right Y", "Pupil position right Z",
                            "Eye movement type"]

        # cut_data = cut_data.dropna(subset=["Gaze direction left X"])
        # print( 100 * np.average(np.isnan(cut_data["Gaze direction left X"].values)))

        columns = ["Timestamp",
                   "Gaze_point_X", "Gaze_point_Y",
                   "Gaze_point_3D_X", "Gaze_point_3D_Y", "Gaze_point_3D_Z",
                   "Gaze_direction_left_X", "Gaze_direction_left_Y", "Gaze_direction_left_Z",
                   "Gaze_direction_right X", "Gaze_direction_right_Y", "Gaze_direction_right_Z",
                   "Pupil_position_left_X", "Pupil_position_left_Y", "Pupil_position_left_Z",
                   "Pupil_position_right_X", "Pupil_position_right_Y", "Pupil_position_right_Z", "Eye_movement_type"]

        temp_data = np.empty((vicon_frame, len(columns)))
        tobii_df = pd.DataFrame(data=temp_data, columns=columns)

        sp = cut_data.iloc[1]["Recording timestamp"] - cut_data.iloc[0]["Recording timestamp"]
        if sp < 15:
            cut_data = cut_data.iloc[::2]
        start = start + np.arange(vicon_frame) * 10
        valid = (cut_data["Validity left"].values == "Valid") | (cut_data["Validity right"].values == "Valid")
        tobii_time = cut_data["Recording timestamp"].values
        dist_mat = np.abs(np.expand_dims(start, axis=0) - np.expand_dims(tobii_time, 1))
        closest_points = np.nanmin(dist_mat, axis=-1)
        closest_idx = np.nanargmin(dist_mat, axis=-1)
        selected_idx = np.argwhere((closest_points <= 10) & (valid))[:, 0]
        tobii_df.iloc[10+closest_idx[selected_idx]] = cut_data.iloc[selected_idx][
            selected_columns]
        for i in range(1, len(tobii_df)):
            tobii_df.loc[i, 'Eye_movement_type'] = eventToNumeric(tobii_df.loc[i - 1, 'Eye_movement_type'],
                                                                  tobii_df.loc[i, 'Eye_movement_type'])
        # print(int(ref.Offset.values[0]))
        # tobii_df.iloc[ int(ref.Offset.values[0]) + closest_idx[selected_idx]] = cut_data.iloc[selected_idx][selected_columns]

        percentage_fill = 100 * (np.sum(tobii_df.Timestamp.values != 0) / vicon_frame)
        return participant, tobii_df, percentage_fill


if __name__ == '__main__':
    import glob
    import pickle

    ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
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

        if "2022-11-24_M_T04" in file_name:

            try:
                reader = SubjectObjectReader()
                obj, sub, ball = reader.extractData(
                    result_path + folder_name + "\\" + file_name + "_wb.pkl")

                tobii_results = []
                for s in sub:
                    reader = TobiiCleaner(ref_file)
                    tobii_filename = file_name.split("_")[-1] + "_" + s["name"] + ".tsv"
                    # print(result_path + folder_name + "\\Tobii\\*" + tobii_filename + "")
                    tobii_files = glob.glob(result_path + folder_name + "\\Tobii\\*" + tobii_filename + "")
                    participant, tobii_df, percentage_fill = reader.matchData(tobii_files[0])
                    tobii_results.append({"name": participant, "trajectories": tobii_df})

                    # check ball bounce
                    import random

                    success_idx = ball[0]["success_idx"]
                    for se in success_idx[20:]:
                        if (tobii_df.loc[se[2]]["Timestamp"] != 0):
                            break
                    print("%s, %s, %f,  %f, %f" % (file_name, participant, percentage_fill, tobii_df.loc[se[2]]["Timestamp"], se[2]))

                data = [obj, sub, ball, tobii_results]



                with open(
                        result_path + folder_name + "\\" + file_name + "_complete.pkl",
                        'wb') as f:
                    pickle.dump(data, f)
            except:

                print("Error: " + file_name + ", ")
