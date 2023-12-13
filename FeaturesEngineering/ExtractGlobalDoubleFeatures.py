from FeaturesEngineering.SingleDoubleFeaturesExtractor import SingleFeaturesExtractor, DoubleFeaturesExtractor
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np
import pandas as pd
from FeaturesEngineering.FeaturesLib import computeSkill
from SubjectObject import Subject, Ball, TableWall
import matplotlib.pyplot as plt

# define readers
reader = SubjectObjectReader()
vis = Visualization()
ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref_with_racket.csv"
result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
ref_df = pd.read_csv(ref_file)
single_df = ref_df.loc[ref_df.Trial_Type == "S"]

double_df = ref_df.loc[ref_df.Trial_Type == "P"]
double_df_unique = double_df.loc[double_df.Session_Code.drop_duplicates().index]


for i, d in double_df_unique.iterrows():

    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]
    racket = d["racket"]
    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial

    # racket pairs
    session_info = double_df[double_df["Session_Code"] == d["Session_Code"] ]
    obj, sub, ball, tobii = reader.extractData(
        result_path + folder_name + "\\" + file_name + "_complete.pkl")
    rackets = {}
    table = None
    wall = None
    for o in obj:
        if "racket" in o["name"].lower():
            rackets[o["name"]] = o
        if "table" in o["name"].lower():
            table = o
        if "wall" in o["name"].lower():
            wall = o

    j = 0


    subjects_list = []

    # extract ball information
    ball = ball[0]
    success = ball["success_idx"]
    failures = ball["failures_idx"]
    b_ball = Ball(ball=ball)
    table_wall = TableWall(table=table, wall=wall)
    for s, t in zip(sub, tobii):
        racket_sub = rackets[session_info[session_info["Participants"] == s["name"]]["racket"].values[0]]
        hand = session_info[session_info["Participants"] == s["name"]]["hand"].values[0]
        p_subject = Subject(sub=s, tobii=t, racket=racket_sub, ball=b_ball, hand=hand)
        features_extractor = SingleFeaturesExtractor(sub=p_subject, ball=b_ball, table_wall=table_wall)
        subjects_list.append(features_extractor)

    double_feature_extractor = DoubleFeaturesExtractor(subjects_list[0], subjects_list[1], ball=b_ball, table_wall=table_wall)
    double_feature_extractor.extractGlobalFeatures()

    print(file_name)
    # if file_name == "2022-11-21_A_T04":
    # obj, sub, ball, tobii = reader.extractData(
    #     result_path + folder_name + "\\" + file_name + "_complete.pkl")
    # racket = None
    # table = None
    # wall = None
    # for o in obj:
    #     if "racket" in o["name"].lower():
    #         racket = o
    #     if "table" in o["name"].lower():
    #         table = o
    #     if "wall" in o["name"].lower():
    #         wall = o
