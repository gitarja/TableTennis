import pandas as pd
import numpy as np
from Utils.DataReader import ViconReader, SubjectObjectReader
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

    file_session_path = file_path + folder_name + "\\"
    result_session_path = result_path + folder_name + "\\"

    reader = SubjectObjectReader()
    obj, sub, ball = reader.extractData(
        result_path + folder_name + "\\" + file_name + "_wb.pkl")

    failures_episode = ball[0]["failures_idx"]
    if len(failures_episode) > 0:
        bounce_table = failures_episode[:, -1]
        bounce_wall = failures_episode[:, 2]
        if np.sum(bounce_table < bounce_wall) > 0:
            print(file_name)
    # print(ball)
