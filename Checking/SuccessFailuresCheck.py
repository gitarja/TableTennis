import pandas as pd
import numpy as np
from Utils.DataReader import SubjectObjectReader
import random

ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
file_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\"
result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
ref_df = pd.read_csv(ref_file)
single_df = ref_df.loc[ref_df.Trial_Type == "S"]

for i, d in single_df.iterrows():
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]

    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial
    if file_name =="2023-03-15_A_T01":

        file_session_path = file_path + folder_name + "\\"
        result_session_path = result_path + folder_name + "\\"

        # if file_name == "2023-03-15_A_T01":
        reader = SubjectObjectReader()
        obj, sub, ball = reader.extractData(
            result_path + folder_name + "\\" + file_name + "_wb.pkl")
        trajectory = ball[0]["trajectories"]
        success_episode = ball[0]["success_idx"]
        failures_episode = ball[0]["failures_idx"]

        start = 25173
        stop = start + 1000

        num_success = np.sum((success_episode[:, 0] >= start) & (success_episode[:, 0] <= stop))
        if (len(failures_episode)) > 0:
            num_failure = np.sum((failures_episode[:, 0] >= start) & (failures_episode[:, 0] <= stop))
        else:
            num_failure = 0
        file_info = file_name.split("_")

        print("%s, %s, %s, %d, %d, %f, %f" % (file_info[0], file_info[1], file_info[2], start, stop, num_success, num_failure))
