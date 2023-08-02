from FeaturesEngineering.ClassicFeatures import Classic
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# define histoplot

def plotHist(data_s, data_f):
    '''
    Onset
    Offet
    Duration
    Mean distance angle
    Mean difference
    Min difference
    Max difference
    Mean magnitude
    Sum magnitude
    Global magnitude
    :param data:
    :return:
    '''
    n_bins = 50
    labels = ["Onset", "Offset", "Duration", "MDA", "MD", "MID", "MAD", "MM", "SM", "GM"]
    xlims = [[0, 350], [0, 350] , [0, 150], [0, 750], [0, 50], [0, 100], [0, 100],  [0, 3000], [0, 3000]]
    fig, axs = plt.subplots(4, 10, tight_layout=True)
    for i in range(4):
        ds = data_s[((data_s[:, i, 0] != 0) & (data_s[:, i, 1] != 0)), i, :]
        df = data_f[((data_f[:, i, 0] != 0) & (data_f[:, i, 1] != 0)), i, :]
        # print("---------------Phase "+str(i)+" --------------------------")
        for j in range(10):
            # hist, bin_edges = np.histogram(d[:, j], bins=n_bins)
            # print("%s: %.2f,  %.2f, %.2f" % (labels[j], bin_edges[np.argmax(hist)], np.average(d[:, j]), np.std(d[:, j])))
            axs[i, j].hist(ds[:, j], bins=n_bins, histtype='step',  fill=False)
            axs[i, j].hist(df[:, j], bins=n_bins, histtype='step',  fill=False)

            # axs[i, j].boxplot([ds[:, j], df[:, j]])
            axs[i, j].set_title(labels[j])
            # axs[i, j].set_xlim(xlims[j])

    plt.show()


# define readers
reader = SubjectObjectReader()
vis = Visualization()
ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
ref_df = pd.read_csv(ref_file)
single_df = ref_df.loc[ref_df.Trial_Type == "S"]

double_df = ref_df.loc[ref_df.Trial_Type == "P"]
double_df_unique = double_df.loc[double_df.Session_Code.drop_duplicates().index]

# single_df = single_df.iloc[0:10]
saccade_phase1_s = []
saccade_phase2_s = []
saccade_phase3_s = []


saccade_phase1_f = []
saccade_phase2_f = []
saccade_phase3_f = []
for i, d in double_df_unique.iterrows():
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]
    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial


    print(file_name)
    # if file_name == "2022-11-08_A_T02":

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

    j=0
    for s, t in zip(sub, tobii):
        if j == 0:
            j+=1
            continue
        print(s["name"])
        features_extractor = Classic(s, racket, ball[0], t, table, wall)

        s_sacc, f_sacc = features_extractor.extractSaccadePursuit(normalize=False)

        # adding saccade of phase 1
        saccade_phase1_s.append(s_sacc["saccade_p1"])

        # adding saccade of phase 2
        saccade_phase2_s.append(s_sacc["saccade_p2"])

        # adding saccade of phase 3
        saccade_phase3_s.append(s_sacc["saccade_p3"])

        # adding saccade of failures episode
        if len(f_sacc) > 0:
            saccade_phase1_f.append(f_sacc["saccade_p1"])
            saccade_phase2_f.append(f_sacc["saccade_p2"])
            saccade_phase3_f.append(f_sacc["saccade_p3"])

saccade_phase1_s = np.concatenate(saccade_phase1_s, 0)
saccade_phase2_s = np.concatenate(saccade_phase2_s, 0)
saccade_phase3_s = np.concatenate(saccade_phase3_s, 0)

saccade_phase1_f = np.concatenate(saccade_phase1_f, 0)
saccade_phase2_f = np.concatenate(saccade_phase2_f, 0)
saccade_phase3_f = np.concatenate(saccade_phase3_f, 0)

plotHist(saccade_phase1_s, saccade_phase1_f)
plotHist(saccade_phase2_s, saccade_phase2_f)
plotHist(saccade_phase3_s, saccade_phase3_f)
