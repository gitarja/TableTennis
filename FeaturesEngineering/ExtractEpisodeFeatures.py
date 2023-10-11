from FeaturesEngineering.FeaturesExtractor import F
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np
import pandas as pd
from FeaturesEngineering.FeaturesLib import computeSkill

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

columns = ['id_subject',
           "skill_subject",
           "pr_p1_al",
           "pr_p2_al",
           "pr_p3_fx",
           "pr_p1_sf",
           "pr_p2_sf",
           "pr_p1_sc_on",
           "pr_p1_sc_off",
           "pr_p1_sc_du",
           "pr_p1_sc_meDa",
           "pr_p1_sc_meDo",
           "pr_p1_sc_miDo",
           "pr_p1_sc_maDo",
           "pr_p1_sc_meM",
           "pr_p1_sc_sumM",
           "pr_p1_sc_gM",
           "pr_p1_sc_pv",
           "pr_p1_sc_bgd",
           "pr_p1_sc_sdr",
           "pr_p2_sc_on",
           "pr_p2_sc_off",
           "pr_p2_sc_du",
           "pr_p2_sc_meDa",
           "pr_p2_sc_meDo",
           "pr_p2_sc_miDo",
           "pr_p2_sc_maDo",
           "pr_p2_sc_meM",
           "pr_p2_sc_sumM",
           "pr_p2_sc_gM",
           "pr_p2_sc_pv",
           "pr_p2_sc_bgd",
           "pr_p2_sc_sdr",
           "pr_p3_fx_on",
           "pr_p3_fx_off",
           "pr_p3_fx_du",
           "pr_p3_meDa",
           "pr_p3_miDa",
           "pr_p3_stdDA",
           "pr_p3_phaseDA",
           "pr_p3_avgGain",
           "pr_p3_stdGain",
           "ec_start_fs",
           "ec_e1_angle",
           "ec_e2_angle",
           "ec_e3_angle",
           "ec_e4_angle",
           "ec_fs_racket_speed",
           "ec_fs_racket_acc",
           "ec_fs_ball_speed",
           "ec_fs_ball_acc",
           "ec_fs_ball_racket_ratio",
           "ec_fs_ball_racket_dir",
           "im_racket_force",
           "im_ball_force",
           "im_rb_ang_collision",
           "im_ball_fimp",
           "im_rb_dist",
           "im_to_wrist_dist",
           "im_rack_wrist_dist",
           'episode_label', 'observation_label', 'success']
df = pd.DataFrame()

for i, d in single_df.iterrows():
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]
    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial

    print(file_name)
    # if file_name == "2022-11-21_A_T04":
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
        ball = ball[0]
        success = ball["success_idx"]
        failures = ball["failures_idx"]
        features_extractor = Classic(s, racket, ball, t, table, wall)
        skill = computeSkill(success, failures)
        subject_episode_features = features_extractor.extractEpisodeFeatures(saccade_normalize=False)
        subject_skill = np.ones((len(subject_episode_features), 1)) * skill
        subject_name = np.array([[s["name"]] * len(subject_episode_features)]).transpose()
        complete_features = np.hstack([subject_name, subject_skill, subject_episode_features])
        row_to_append = pd.DataFrame(complete_features, columns=columns)
        # print(len(complete_features))
        df = pd.concat([df, row_to_append], ignore_index=True)
        # print(len(df))

df[[
    "skill_subject",
    "pr_p1_al",
    "pr_p2_al",
    "pr_p3_fx",
    "pr_p1_sf",
    "pr_p2_sf",
    "pr_p1_sc_on",
    "pr_p1_sc_off",
    "pr_p1_sc_du",
    "pr_p1_sc_meDa",
    "pr_p1_sc_meDo",
    "pr_p1_sc_miDo",
    "pr_p1_sc_maDo",
    "pr_p1_sc_meM",
    "pr_p1_sc_sumM",
    "pr_p1_sc_gM",
    "pr_p1_sc_pv",
    "pr_p1_sc_bgd",
    "pr_p1_sc_sdr",
    "pr_p2_sc_on",
    "pr_p2_sc_off",
    "pr_p2_sc_du",
    "pr_p2_sc_meDa",
    "pr_p2_sc_meDo",
    "pr_p2_sc_miDo",
    "pr_p2_sc_maDo",
    "pr_p2_sc_meM",
    "pr_p2_sc_sumM",
    "pr_p2_sc_gM",
    "pr_p2_sc_pv",
    "pr_p2_sc_bgd",
    "pr_p2_sc_sdr",
    "pr_p3_fx_on",
    "pr_p3_fx_off",
    "pr_p3_fx_du",
    "pr_p3_meDa",
    "pr_p3_miDa",
    "pr_p3_stdDA",
    "pr_p3_phaseDA",
    "pr_p3_avgGain",
    "pr_p3_stdGain",
    "ec_start_fs",
    "ec_e1_angle",
    "ec_e2_angle",
    "ec_e3_angle",
    "ec_e4_angle",
    "ec_fs_racket_speed",
    "ec_fs_racket_acc",
    "ec_fs_ball_speed",
    "ec_fs_ball_acc",
    "ec_fs_ball_racket_ratio",
    "ec_fs_ball_racket_dir",
    "im_racket_force",
    "im_ball_force",
    "im_rb_ang_collision",
    "im_ball_fimp",
    "im_rb_dist",
    "im_to_wrist_dist",
    "im_rack_wrist_dist",
    'episode_label', 'observation_label', 'success']] = df[["skill_subject",
                                                            "pr_p1_al",
                                                            "pr_p2_al",
                                                            "pr_p3_fx",
                                                            "pr_p1_sf",
                                                            "pr_p2_sf",
                                                            "pr_p1_sc_on",
                                                            "pr_p1_sc_off",
                                                            "pr_p1_sc_du",
                                                            "pr_p1_sc_meDa",
                                                            "pr_p1_sc_meDo",
                                                            "pr_p1_sc_miDo",
                                                            "pr_p1_sc_maDo",
                                                            "pr_p1_sc_meM",
                                                            "pr_p1_sc_sumM",
                                                            "pr_p1_sc_gM",
                                                            "pr_p1_sc_pv",
                                                            "pr_p1_sc_bgd",
                                                            "pr_p1_sc_sdr",
                                                            "pr_p2_sc_on",
                                                            "pr_p2_sc_off",
                                                            "pr_p2_sc_du",
                                                            "pr_p2_sc_meDa",
                                                            "pr_p2_sc_meDo",
                                                            "pr_p2_sc_miDo",
                                                            "pr_p2_sc_maDo",
                                                            "pr_p2_sc_meM",
                                                            "pr_p2_sc_sumM",
                                                            "pr_p2_sc_gM",
                                                            "pr_p2_sc_pv",
                                                            "pr_p2_sc_bgd",
                                                            "pr_p2_sc_sdr",
                                                            "pr_p3_fx_on",
                                                            "pr_p3_fx_off",
                                                            "pr_p3_fx_du",
                                                            "pr_p3_meDa",
                                                            "pr_p3_miDa",
                                                            "pr_p3_stdDA",
                                                            "pr_p3_phaseDA",
                                                            "pr_p3_avgGain",
                                                            "pr_p3_stdGain",
                                                            "ec_start_fs",
                                                            "ec_e1_angle",
                                                            "ec_e2_angle",
                                                            "ec_e3_angle",
                                                            "ec_e4_angle",
                                                            "ec_fs_racket_speed",
                                                            "ec_fs_racket_acc",
                                                            "ec_fs_ball_speed",
                                                            "ec_fs_ball_acc",
                                                            "ec_fs_ball_racket_ratio",
                                                            "ec_fs_ball_racket_dir",
                                                            "im_racket_force",
                                                            "im_ball_force",
                                                            "im_rb_ang_collision",
                                                            "im_ball_fimp",
                                                            "im_rb_dist",
                                                            "im_to_wrist_dist",
                                                            "im_rack_wrist_dist",
                                                            'episode_label', 'observation_label', 'success']].apply(
    pd.to_numeric, errors='coerce')
df.to_pickle(
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl",
    protocol=4)
df.to_csv(
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.csv")
