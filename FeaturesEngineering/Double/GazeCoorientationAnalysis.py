from FeaturesEngineering.SingleDoubleFeaturesExtractor import SingleFeaturesExtractor, DoubleFeaturesExtractor
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np
import pandas as pd
from FeaturesEngineering.FeaturesLib import computeSkill, computeSequenceFeatures
from FeaturesEngineering.SubjectObject import Subject, Ball, TableWall
import matplotlib.pyplot as plt
from FeaturesEngineering.Conf import double_features_col, double_features_col_num



def getIndividualInfo(df):
    reader = SubjectObjectReader()
    dates = df["Date"].replace(".", "-")
    session = df["Session"]
    trial = df["Trial"]

    ind_folder_name = dates + "_" + session
    ind_file_name = folder_name + "_" + trial

    obj, sub, ball, tobii = reader.extractData(
        result_path + ind_folder_name + "\\" + ind_file_name + "_complete_final.pkl")
    ball = ball[0]
    skill = computeSkill(ball["success_idx"], ball["failures_idx"])

    return skill
# define readers
reader = SubjectObjectReader()
vis = Visualization()
ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref_with_racket.csv"
result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
ref_df = pd.read_csv(ref_file)
single_df = ref_df.loc[ref_df.Trial_Type == "S"]

double_df = ref_df.loc[ref_df.Trial_Type == "P"]
double_df_unique = double_df.loc[double_df.Session_Code.drop_duplicates().index]


df = pd.DataFrame()
avg_phase_all = []
surr_avg_phase_all = []
for i, d in double_df_unique.iterrows():
    print(d["Session_Code"])
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]
    racket = d["racket"]
    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial

    # racket pairs
    session_info = double_df[double_df["Session_Code"] == d["Session_Code"] ]
    obj, sub, ball, tobii = reader.extractData(
        result_path + folder_name + "\\" + file_name + "_complete_final.pkl")
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
    subjects_skill = []
    subjects_name = []

    # extract ball information
    ball = ball[0]
    success = ball["success_idx"]
    failures = ball["failures_idx"]
    b_ball = Ball(ball=ball)
    table_wall = TableWall(table=table, wall=wall)
    for s, t in zip(sub, tobii):
        racket_sub = rackets[session_info[session_info["Participants"] == s["name"]]["racket"].values[0]]
        hand = session_info[session_info["Participants"] == s["name"]]["hand"].values[0]

        # get individual skill
        ind_df = single_df[single_df["Participants"]==s["name"]]
        ind_skill = getIndividualInfo(ind_df.iloc[0])
        subjects_skill.append(ind_skill)
        subjects_name.append(s["name"])

        p_subject = Subject(sub=s, tobii=t, racket=racket_sub, ball=b_ball, hand=hand)
        features_extractor = SingleFeaturesExtractor(sub=p_subject, ball=b_ball, table_wall=table_wall)
        subjects_list.append(features_extractor)

    team_skill = computeSkill(success, failures)
    team_max_seq, team_avg_seq = computeSequenceFeatures(success, failures)
    double_feature_extractor = DoubleFeaturesExtractor(subjects_list[0], subjects_list[1], ball=b_ball, table_wall=table_wall)

    avg_phase, surr_avg = double_feature_extractor.s1.detectGazeCoorientation(double_feature_extractor.b.success_idx, double_feature_extractor.s2, b=double_feature_extractor.b)
    avg_phase_all.append(avg_phase)
    surr_avg_phase_all.append(surr_avg)
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
mask = ~np.isnan(np.concatenate(avg_phase_all))
avg_phase_all = np.concatenate(avg_phase_all)[mask]
surr_avg_phase_all = np.concatenate(surr_avg_phase_all)[mask]
label_ori = ["ori" for i in range(len(avg_phase_all))]
label_surr = ["surr" for i in range(len(surr_avg_phase_all))]

df = pd.DataFrame({"data": np.hstack([avg_phase_all, surr_avg_phase_all]), "label":np.hstack([label_ori, label_surr])})
sns.catplot(data=df, x="label", y="data", kind="box")
sns.stripplot(df, x="label", y="data", size=3, color=".3", alpha=.1)
# sns.despine(trim=True, left=True)
plt.show()
print(stats.ttest_ind(avg_phase_all, surr_avg_phase_all, equal_var=False))