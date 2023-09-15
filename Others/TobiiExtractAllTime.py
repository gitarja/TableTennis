import pandas as pd
from Utils.DataReader import SubjectObjectReader

if __name__ == '__main__':
    import glob
    import pickle

    ref_file = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\Tobii_ref.csv"
    result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\"
    ref_df = pd.read_csv(ref_file)

    experiment_summary = []
    for i, d in ref_df.iterrows():
        dates = d["Date"].replace(".", "-")
        session = d["Session"]
        trial = d["Trial"]

        folder_name = dates + "_" + session
        file_name = folder_name + "_" + trial

        reader = SubjectObjectReader()
        obj, sub, ball = reader.extractData(
            result_path + folder_name + "\\" + file_name + "_wb.pkl")

        tobii_results = []
        for s in sub:
            try:
                tobii_filename = file_name.split("_")[-1] + "_" + s["name"] + ".tsv"
                tobii_files = glob.glob(result_path + folder_name + "\\Tobii\\*" + tobii_filename + "")
                data = pd.read_csv(tobii_files[0], delimiter="\t")
                date_str = data.loc[1, ["Recording date"]].values[0]
                participant_str = data.loc[1, ["Participant name"]].values[0].split("_")[-1]
                trial_str = data.loc[1, ["Participant name"]].values[0].split("_")[0]
                start_time = data.loc[1, ["Recording start time UTC"]].values[0]

                experiment_summary.append([date_str, start_time, trial_str, participant_str])
            except:
                print(tobii_files[0])


    df_experiment = pd.DataFrame(experiment_summary,
                                              columns =['Date', "Time", "Trial_name", "Participant_name"])


    df_experiment.to_csv("F:\\users\\prasetia\\OneDrive - bwstaff\\CollectiveSport+SocialLearning\\TableTennis\\Experiment1\\experiment_date_time_summary.csv")
