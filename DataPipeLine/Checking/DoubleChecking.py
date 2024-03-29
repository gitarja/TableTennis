import pandas as pd
import numpy as np
from Utils.DataReader import SubjectObjectReader
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

    if file_name == "2023-03-15_A_T01":
        reader = SubjectObjectReader()
        obj, sub, ball = reader.extractData(
            result_path + folder_name + "\\" + file_name + "_wb.pkl")