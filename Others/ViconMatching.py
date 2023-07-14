import pandas as pd



sessions_df = pd.read_csv("Vicon_Session.csv")

ref_df = pd.read_csv("Vicon_Ref.csv")



for index, row in sessions_df.iterrows():
    date = row["Date"]

    session = row["Session"]
    trial = row["Trial"]
    subject = row["Subject"]

    condition = (ref_df["Date"] == date) & (ref_df["Session"] == session) & (ref_df["Trial"]. apply(str. lower) == trial.lower())
    current_ref = ref_df.loc[condition]
    if len(current_ref) > 0:
        sessions_df.loc[index,  "Vicon_Frame"] = current_ref["N_frame"].values[0]
        sessions_df.loc[index, "Trial_Type"] = current_ref["Trial_Type"].values[0]
    else:
        print(date)



sessions_df.to_csv("Vicon_Session_filled.csv")



