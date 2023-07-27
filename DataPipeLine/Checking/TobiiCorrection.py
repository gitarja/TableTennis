import pandas as pd
import numpy as np

ori_ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref-bc-2023-07-26.csv"
ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
correction_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_sync_double_correction.csv"


correction_df = pd.read_csv(correction_file)

ref_df = pd.read_csv(ori_ref_file)

# only select the ones with offset greater > 0
correction_df = correction_df[np.abs(correction_df["Offset"].values)>0]

for i, r in correction_df.iterrows():
    date, subject, trial = r["Session_name"].split("_")
    date = date.replace("-", ".")
    participant = r["Participant"]
    offset = r["Offset"]

    condition = (ref_df["Date"] == date) & (ref_df["Trial"] == trial) & (ref_df["Participants"] == participant)

    # select_ref = ref_df.loc[condition]

    ref_df.loc[condition, "Start_TTL"] = ref_df.loc[condition,"Start_TTL"] + (offset / 1000)
    ref_df.loc[condition,"Stop_TTL"] = ref_df.loc[condition,"Stop_TTL"] + (offset / 1000)


ref_df.to_csv(ref_file)