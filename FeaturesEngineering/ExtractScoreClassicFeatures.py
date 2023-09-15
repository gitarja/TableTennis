import pandas as pd
from Utils.DataReader import SubjectObjectReader
from FeaturesEngineering.FeaturesLib import computeScore
from FeaturesEngineering.ClassicFeatures import Classic
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

    for i, d in single_df.iterrows():
        dates = d["Date"].replace(".", "-")
        session = d["Session"]
        trial = d["Trial"]

        folder_name = dates + "_" + session
        file_name = folder_name + "_" + trial
        print(file_name)

        result_session_path = result_path + folder_name + "\\"

        reader = SubjectObjectReader()
        obj, sub, ball, tobii = reader.extractData(result_session_path + file_name + "_complete.pkl")

        # wall
        racket = None
        table = None
        wall = None
        for o in obj:
            if "racket" in o["name"].lower():
                racket = o
            if "table" in o["name"].lower():
                table = o
            if "wall" in o["name"].lower():
                wall = o
        ball = ball[0]
        success = ball["success_idx"]
        failures = ball["failures_idx"]

        n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq, bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt, mov_avg1, mov_avg2, mov_avg3, mov_var1, mov_var2, mov_var3 = computeScore(
            success, failures, ball_trajetories=ball["trajectories"].values, wall_trajectories=wall["trajectories"])

        for sub_i, tobii_i in zip(sub, tobii):
            features_extractor = Classic(sub_i, racket, ball, tobii_i, table, wall)
            s_fsc, _ = features_extractor.extractForwardswing(prev=False)
        # extract anticipatory features

        if len(sub) == 1:
            summary_data.append(
                [file_name, sub[0]["name"], "", n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq, bounce_hull,
                 bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt, mov_avg1, mov_avg2, mov_avg3, mov_var1, mov_var2, mov_var3,
                 s_fsc["avg_start_fs"], s_fsc["std_start_fs"]])
        else:
            summary_data.append(
                [file_name, sub[0]["name"], sub[1]["name"], n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq,
                 bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt, mov_avg1, mov_avg2, mov_avg3, mov_var1, mov_var2, mov_var3,
                 s_fsc["avg_start_fs"], s_fsc["std_start_fs"]])

columns = ["file_name", "Subject1", "Subject2", "n_success", "n_failures", "norm_score", "skill", "task_score",
           "max_seq", "avg_seq", "bounce_hull", "bounce_std", "bounce_sp_entropy", "bounce_sc_entropy", "rt_lyp",
           "samp_en", "std_rt", "avg_p1", "avg_p2", "avg_p3","var_p1","var_p2","var_p3", "avg_start_fs", "std_start_fs"]
summary_df = pd.DataFrame(summary_data, columns=columns)

# summarize the results
scaler = StandardScaler()

X = summary_df.loc[:, ["norm_score"]].values

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=20).fit(X_scaled)

session_class = kmeans.labels_
summary_df['session_class'] = session_class
summary_df.to_csv(result_path + "summary\\" + "single_summary.csv")
