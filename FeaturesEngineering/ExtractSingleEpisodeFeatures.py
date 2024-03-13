from FeaturesEngineering.SingleDoubleFeaturesExtractor import SingleFeaturesExtractor
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np
import pandas as pd
from FeaturesEngineering.FeaturesLib import computeSkill
from SubjectObject import Subject, Ball, TableWall
import matplotlib.pyplot as plt
from Conf import single_features_col, single_features_col_num

# define readers
reader = SubjectObjectReader()
vis = Visualization()
ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
ref_df = pd.read_csv(ref_file)
single_df = ref_df.loc[ref_df.Trial_Type == "S"]

double_df = ref_df.loc[ref_df.Trial_Type == "P"]
double_df_unique = double_df.loc[double_df.Session_Code.drop_duplicates().index]
saccade_summary = []

df = pd.DataFrame()

for i, d in single_df.iterrows():
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]
    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial

    print(file_name)
    # if file_name == "2022-11-09_A_T07":
    obj, sub, ball, tobii = reader.extractData(
        result_path + folder_name + "\\" + file_name + "_complete_final.pkl")
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

    j = 0
    for s, t in zip(sub, tobii):
        ball = ball[0]
        success = ball["success_idx"]
        failures = ball["failures_idx"]
        b_ball = Ball(ball=ball)
        p_subject = Subject(sub=s, tobii=t, racket=racket, ball=b_ball)
        table_wall = TableWall(table=table, wall=wall)
        features_extractor = SingleFeaturesExtractor(sub=p_subject, ball=b_ball, table_wall=table_wall)
        skill = computeSkill(success, failures)
        subject_episode_features = features_extractor.extractEpisodeFeatures(saccade_normalize=False)
        subject_skill = np.ones((len(subject_episode_features), 1)) * skill
        subject_name = np.array([[s["name"]] * len(subject_episode_features)]).transpose()
        complete_features = np.hstack([subject_name, subject_skill, subject_episode_features])
        row_to_append = pd.DataFrame(complete_features, columns=single_features_col)



        # print(len(complete_features))
        df = pd.concat([df, row_to_append], ignore_index=True)
        # print(len(df))

df[single_features_col_num] = df[single_features_col_num].apply(
    pd.to_numeric, errors='coerce')
df.to_pickle(
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl",
    protocol=4)
df.to_csv(
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.csv")
