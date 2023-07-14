from FeaturesExtractor.ClassicFeatures import Classic
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# define histoplot

def plotHist(data):
    n_bins = 100
    labels = ["Onset", "Offset", "Duration", "MeanVel", "MeanDistAn", "Mid-diff", "Mean-diff"]
    xlims = [[0, 350], [0, 350] , [0, 150], [0, 750], [0, 50], [0, 100], [0, 100]]
    fig, axs = plt.subplots(4, 7, tight_layout=True)
    for i in range(4):
        d = data[((data[:, i, 0] != 0) & (data[:, i, 1] != 0)), i, :]
        print("---------------Phase "+str(i)+" --------------------------")
        for j in range(7):
            hist, bin_edges = np.histogram(d[:, j], bins=n_bins)
            print("%s: %.2f,  %.2f, %.2f" % (labels[j], bin_edges[np.argmax(hist)], np.average(d[:, j]), np.std(d[:, j])))
            axs[i, j].hist(d[:, j], bins=n_bins)
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
# single_df = single_df.iloc[0:10]
saccade_phase1 = []
saccade_phase2 = []
saccade_phase3 = []
for i, d in single_df.iterrows():
    dates = d["Date"].replace(".", "-")
    session = d["Session"]
    trial = d["Trial"]
    folder_name = dates + "_" + session
    file_name = folder_name + "_" + trial


    print(file_name)
    if file_name == "2022-11-08_A_T04":

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
        features_extractor = Classic(sub[0], racket, ball[0], tobii[0], table, wall)

        features_extractor.extractSaccadePursuit(normalize=False)
    # s_sacc_pursuit_c, f_sacc_pursuit_c = features_extractor.extractSaccadePursuit(normalize=False)

#     # adding saccade of phase 1
#     saccade_phase1.append(s_sacc_pursuit_c["saccade_p1"])
#
#     # adding saccade of phase 2
#     saccade_phase2.append(s_sacc_pursuit_c["saccade_p2"])
#
#     # adding saccade of phase 3
#     saccade_phase3.append(s_sacc_pursuit_c["saccade_p3"])
#
#     # adding saccade of failures episode
#     if len(f_sacc_pursuit_c) > 0:
#         saccade_phase1.append(f_sacc_pursuit_c["saccade_p1"])
#         saccade_phase2.append(f_sacc_pursuit_c["saccade_p2"])
#         saccade_phase3.append(f_sacc_pursuit_c["saccade_p3"])
#
# saccade_phase1 = np.concatenate(saccade_phase1, 0)
#
# saccade_phase2 = np.concatenate(saccade_phase2, 0)
#
# saccade_phase3 = np.concatenate(saccade_phase3, 0)
#
# plotHist(saccade_phase1)
# plotHist(saccade_phase2)
# plotHist(saccade_phase3)
