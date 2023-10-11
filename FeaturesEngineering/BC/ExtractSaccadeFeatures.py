from FeaturesEngineering.ClassicFeatures import Classic
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np
import pandas as pd
from FeaturesEngineering.FeaturesLib import computeSampEn, computeLypanovMax

import matplotlib.pyplot as plt

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

for i, d in single_df.iterrows():
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]
    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial

    print(file_name)
    # if file_name == "2022-11-17_M_T04":
    obj, sub, ball, tobii = reader.extractData(
        result_path + folder_name + "\\" + file_name + "_complete.pkl")
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
        features_extractor = Classic(s, racket, ball[0], t, table, wall)

        s_sacc, _ = features_extractor.extractSaccadePursuit(normalize=False)

        # summarize features for saccade 1
        al_p1 = s_sacc["saccade_p1"]
        avg_alp1 = np.average(al_p1[al_p1[:, 1] != 0], axis=0)
        occ_1 = np.average(al_p1[:, 1] != 0)
        # sampen_alp1 = computeLypanovMax(al_p1[al_p1[:, 1] != 0][:, 1], emb_dim=7)
        sampen_alp1 = computeSampEn(al_p1[al_p1[:, 1] != 0][:, 0], emb_dim=2, r=0.55)
        std_alp1 = np.std(al_p1[al_p1[:, 1] != 0][:, 0])

        # summarize features for saccade 2
        al_p2 = s_sacc["saccade_p2"]
        avg_alp2 = np.average(al_p2[al_p2[:, 1] != 0], axis=0)
        occ_2 = np.average(al_p2[:, 1] != 0)
        # sampen_alp2 = computeLypanovMax(al_p2[al_p2[:, 1] != 0][:, 1], emb_dim=7)
        sampen_alp2 = computeSampEn(al_p2[al_p2[:, 1] != 0][:, 0], emb_dim=2, r=0.55)
        std_alp2 = np.std(al_p2[al_p2[:, 1] != 0][:, 0])

        # summarize features for fsp 3
        fsp_p3 = s_sacc["fsp_phase3"]
        avg_fsp3 = np.average(fsp_p3[fsp_p3[:, 1] != 0], axis=0)
        occ_fsp3 = np.average(fsp_p3[:, 1] != 0)
        # sampen_fsp3 = computeLypanovMax(fsp_p3[fsp_p3[:, 1] != 0][:, 1], emb_dim=7)
        sampen_fsp3 = computeSampEn(fsp_p3[fsp_p3[:, 1] != 0][:, 1], emb_dim=2, r=0.55)
        std_fsp3 = np.std(fsp_p3[fsp_p3[:, 1] != 0][:, 1])

        saccade_summary.append([file_name, s["name"]] + np.hstack(
            [avg_alp1, occ_1, sampen_alp1, std_alp1, avg_alp2, occ_2, sampen_alp2, std_alp2, avg_fsp3, occ_fsp3, sampen_fsp3, std_fsp3]).tolist())

    # # adding saccade of phase 3
    # saccade_phase3_s.append(s_sacc["saccade_p3"])

columns = ["file_name", "Subject",
           "p1_on", "p1_off", "p1_dn", "p1_mda", "p1_md", "p1_mid", "p1_mad", "p1_mm", "p1_sm", "p1_gm", "p1_occ",
           "p1_sampen","p1_std",
           "p2_on", "p2_off", "p2_dn", "p2_mda", "p2_md", "p2_mid", "p2_mad", "p2_mm", "p2_sm", "p2_gm", "p2_occ",
           "p2_sampen","p2_std",
           "fsp3_on", "fsp3_off", "fsp3_dn", "fsp3_mfs", "fsp3_mifs", "fsp3_occ", "fsp3_sampen","fsp3_std"
           ]
summary_df = pd.DataFrame(saccade_summary, columns=columns)
summary_df.to_csv(result_path + "summary\\" + "single_saccade_summary.csv")
