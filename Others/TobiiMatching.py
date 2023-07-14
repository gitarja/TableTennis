import pandas as pd



sessions_df = pd.read_csv("Tobii_Sessions.csv")

ref_df = pd.read_csv("Tobii_Ref.csv")



for index, row in sessions_df.iterrows():
    date = row["Date"]

    session = row["Session"]
    trial = row["Trial"]
    subject = row["Subject"]

    condition = (ref_df["Date"] == date) & (ref_df["Session"] == session) & (ref_df["Trial"] == trial) & (ref_df["Subject"] == subject)
    current_ref = ref_df.loc[condition]
    if len(current_ref) > 0:
        sessions_df.loc[index,  "Start_TTL"] = current_ref["Start"].values[0]
        sessions_df.loc[index, "Stop_TTL"] = current_ref["Stop"].values[0]
    else:
        print(date)



sessions_df.to_csv("Tobii_Sessions_filled.csv")



