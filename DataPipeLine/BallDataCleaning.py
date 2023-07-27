import pandas as pd
from Utils.DataReader import SubjectObjectReader
import pickle
from Double.DoubleBallProcessing import DoulbeBallProcessing

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
        # try:
            file_session_path = file_path + folder_name + "\\"
            result_session_path = result_path + folder_name + "\\"

            reader = SubjectObjectReader()
            obj, sub = reader.extractData(
                result_path + folder_name + "\\" + file_name + ".pkl")

            reader = DoulbeBallProcessing(obj, sub, file_name)
            data, success, failures = reader.cleanDoubleData(file_path + folder_name + "\\" + file_name + ".c3d")

            df = pd.DataFrame(data, columns=["ball_x", "ball_y", "ball_z"])
            data = [obj, sub, [{"name": "ball", "trajectories": df, "success_idx": success, "failures_idx": failures}]]

            with open(result_session_path + file_name + "_wb.pkl",
                      'wb') as f:
                pickle.dump(data, f)

        # except:
        #     print("Error: " + file_name)
