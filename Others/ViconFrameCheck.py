import os
import shutil
import glob
import numpy as np
import pandas as pd

from Utils.DataReader import ViconReader
results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\*"
reader = ViconReader()

df = pd.DataFrame(columns=["Date", "Session", "Trial", "N_frame", "Participants", "Racket", "Trial_Type"])
i=0
for ss in glob.glob(results_path):
    # files = glob.glob(ss+"\\*.csv")
    # print(files)
    for f in glob.glob(ss+"\\*.csv"):
        filename = f.split("\\")[-1].split("_")
        participant_p = ""
        racket_p = ""
        obj, sub, n_frame = reader.extractData(f,
                                      cleaning=True)
        trial_type = "P" if (len(sub) == 2)  else "S"
        for o in obj:
            if "racket" in o["name"].lower():
                racket_p = racket_p + o["name"] + "=" + str(np.average((~np.isnan(o["segments"])) & (o["segments"] != 0)) * 100) + ","

        for s in sub:
            participant_p = participant_p + s["name"] + "=" + str(np.average((~np.isnan(s["segments"])) & (s["segments"] != 0)) * 100) + ","

        df.loc[i] = [filename[0].replace("-", "."), filename[1], filename[2].replace(".csv", ""), n_frame, participant_p, racket_p, trial_type]
        df.to_csv("Vicon_Ref.csv")
        i+=1