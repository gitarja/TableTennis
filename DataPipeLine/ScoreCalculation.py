import pandas as pd
from Utils.DataReader import SubjectObjectReader
from FeaturesEngineering.FeaturesLib import computeScore
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
if __name__ == '__main__':

    ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
    file_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\"
    result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
    ref_df = pd.read_csv(ref_file)
    single_df = ref_df.loc[ref_df.Trial_Type == "S"]
    double_df = ref_df.loc[ref_df.Trial_Type == "P"]
    double_df_unique = double_df.loc[double_df.Session_Code.drop_duplicates().index]

    summary_data = []

    for i, d in double_df_unique.iterrows():
        dates = d["Date"].replace(".", "-")
        session = d["Session"]
        trial = d["Trial"]

        folder_name = dates + "_" + session
        file_name = folder_name + "_" + trial


        result_session_path = result_path + folder_name + "\\"

        reader = SubjectObjectReader()
        obj, sub, ball, tobii = reader.extractData(result_session_path + file_name + "_complete.pkl")

        #wall
        wall = None
        for o in obj:
            if o["name"] == "Wall":
                wall=o
        ball = ball[0]
        success = ball["success_idx"]
        failures = ball["failures_idx"]

        n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq, bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt,  mov_var1, mov_var2, mov_var3 = computeScore(success, failures, ball_trajetories=ball["trajectories"].values, wall_trajectories= wall["trajectories"])

        if len(sub) == 1:
            summary_data.append([file_name, sub[0]["name"], "", n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq, bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt,  mov_var1, mov_var2, mov_var3])
        else:
            summary_data.append([file_name, sub[0]["name"], sub[1]["name"], n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq, bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt,  mov_var1, mov_var2, mov_var3])

    columns = ["file_name", "Subject1", "Subject2", "n_success", "n_failures", "norm_score", "skill", "task_score", "max_seq", "avg_seq", "bounce_hull", "bounce_std",  "bounce_sp_entropy", "bounce_sc_entropy",  "rt_lyp", "samp_en", "std_rt", "var_p1", "var_p2", "var_p3"]
    summary_df = pd.DataFrame(summary_data, columns=columns)

    scaler = StandardScaler()

    # summary_df["norm_score"] = scaler.fit_transform(np.expand_dims(summary_df["norm_score"].values, 1) ).flatten()

    X = summary_df.loc[:, ["norm_score"]].values

    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=20).fit(X_scaled)

    session_class = kmeans.labels_
    summary_df['session_class'] = session_class
    summary_df.to_csv(result_path + "summary\\" + "double_summary.csv")




