from Utils.DataReader import SubjectObjectReader
import pickle
import pandas as pd

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
    # if file_name == "2022-11-08_A_T02":

    file_session_path = result_path + folder_name + "\\"

    # clean data
    reader = SubjectObjectReader()
    obj, sub = reader.extractData(file_session_path + file_name + ".pkl")

    # complete data

    reader_comp = SubjectObjectReader()
    obj_comp, sub_comp, ball_comp, tobii_comp = reader_comp.extractData(file_session_path + file_name + "_complete.pkl")

    reader_ball = SubjectObjectReader()
    _, _, ball_clean = reader_ball.extractData(file_session_path + file_name + "_wb.pkl")

    data = [obj, sub, ball_comp, tobii_comp]

    with open(
            file_session_path + file_name + "_complete_final.pkl",
            'wb') as f:
        pickle.dump(data, f)
    print(file_session_path + file_name + ".pkl")

# result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-12-08_A\\"
#
# file_name = "2022-12-08_A_T06"

# # clean data
# reader = SubjectObjectReader()
# obj, sub  = reader.extractData(result_path+ file_name+ ".pkl")
#
# # complete data
#
# reader_comp = SubjectObjectReader()
# obj_comp, sub_comp, ball_comp, tobii_comp  = reader_comp.extractData(result_path+ file_name+"_complete.pkl")
#
# print(len(sub[0]["trajectories"].values) == len(tobii_comp[0]["trajectories"].values))
#
# data = [obj, sub, ball_comp, tobii_comp]
#
# with open(
#         result_path  + file_name + "_complete_final.pkl",
#         'wb') as f:
#     pickle.dump(data, f)
