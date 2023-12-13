import pandas as pd
import glob
import pickle
from Utils.DataReader import SubjectObjectReader
import numpy as np

ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
ref_df = pd.read_csv(ref_file)
ref_new = ref_df.copy()
ref_new["racket"] = np.nan
ref_new["hand"] = np.nan
double_df = ref_df.loc[ref_df.Trial_Type == "P"]
double_df_unique = double_df.loc[double_df.Session_Code.drop_duplicates().index]

for i, d in double_df_unique.iterrows():
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]

    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial

    reader = SubjectObjectReader()
    obj, sub, ball_list, tobii = reader.extractData(
        result_path + folder_name + "\\" + file_name + "_complete.pkl")
    racket_list = []
    racket_list_idx = []
    ball = ball_list[0]
    success = ball["success_idx"]
    j = 0
    for o in obj:
        if "racket" in o["name"].lower():
            racket_list.append(o)
            racket_list_idx.append(j)
        j+=1

    # clean the racket

    if racket_list[0]["segments"].shape[1] != racket_list[1]["segments"].shape[1]:
        true_idx = np.argmin([racket_list[0]["segments"].shape[1], racket_list[1]["segments"].shape[1]])
        false_idx = np.argmax([racket_list[0]["segments"].shape[1], racket_list[1]["segments"].shape[1]])
        true_name = racket_list[true_idx]["name"]

        # drop data from false

        racket_list[false_idx]["segments"] = racket_list[false_idx]["segments"][
            racket_list[false_idx]["segments"].columns.drop(
                (list(racket_list[false_idx]["segments"].filter(regex=true_name))))]
        racket_list[false_idx]["joints"] = racket_list[false_idx]["joints"][
            racket_list[false_idx]["joints"].columns.drop(
                (list(racket_list[false_idx]["joints"].filter(regex=true_name))))]
        racket_list[false_idx]["trajectories"] = racket_list[false_idx]["trajectories"][
            racket_list[false_idx]["trajectories"].columns.drop(
                (list(racket_list[false_idx]["trajectories"].filter(regex=true_name))))]

        obj[racket_list_idx[0]] = racket_list[0]
        obj[racket_list_idx[1]] = racket_list[1]


        data = [obj, sub, ball_list, tobii]
        print(file_name)
        with open(
                result_path + folder_name + "\\" + file_name + "_complete_corrected.pkl",
                'wb') as f:
            pickle.dump(data, f)
    s_idx = success[3, 2]
    # change racket values

    # if d["Session_Code"] == "SE006_T05":
    #     print("error")
    for s in sub:
        try:
            hand = "L"
            r_idx = 0
            r_wirst = s["segments"].filter(regex='R_Wrist_T').values[s_idx]
            l_wirst = s["segments"].filter(regex='L_Wrist_T').values[s_idx]
            r1 = racket_list[0]["segments"].filter(regex='pt_T').values[s_idx]
            r2 = racket_list[1]["segments"].filter(regex='pt_T').values[s_idx]
            l_r1 = np.linalg.norm(r1 - l_wirst)
            l_r2 = np.linalg.norm(r2 - l_wirst)
            r_r1 = np.linalg.norm(r1 - r_wirst)
            r_r2 = np.linalg.norm(r2 - r_wirst)

            closest_r = np.argmin([l_r1, l_r2, r_r1, r_r2])
            if closest_r == 1:
                r_idx = 1
            elif closest_r == 2:
                hand = "R"
            elif closest_r == 3:
                hand = "R"
                r_idx = 1

            df_idx = (ref_new["Session_Code"] == d["Session_Code"]) & (ref_new["Participants"] == s["name"])
            ref_new.loc[df_idx, "racket"] = racket_list[r_idx]["name"]
            ref_new.loc[df_idx, "hand"] = hand
            # print(s["name"] + " " + hand + " " + racket_list[r_idx]["name"])
        except:
            print("Error " + file_name)


ref_new.to_csv("F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref_with_racket.csv")