import pandas as pd
from Conf import x_episode_columns, y_episode_column, excluded_subject

class SingleFeaturesReader:

    def __init__(self, file_path=""):
        df = pd.read_pickle(file_path)

        # exclude subjects
        df = df.loc[~df["id_subject"].isin(excluded_subject), :]
        # exclude -1 data
        df = df.loc[df["success"] != -1]

        self.df = df

    def splitEpisode(self, v, th=15, augment=3, min_seq = 2):
        X_sequence = []
        y_sequence = []
        if len(v) > th:
            n = len(v) // augment

            for i in range(augment - 1):
                # if len(v.iloc[:((i+1)*n)][x_episode_columns].values) < 3:
                #     print("Error")
                X_sequence.append(v.iloc[(i*n):][x_episode_columns].values)
                X_sequence.append(v.iloc[:((i+1)*n)][x_episode_columns].values)

                y_sequence.append(v.iloc[(i*n):][ y_episode_column].values)
                y_sequence.append(v.iloc[:((i+1)*n)][y_episode_column].values)

        else:
            if len(v) > 2:
                X_sequence.append(v.iloc[:][ x_episode_columns].values)
                y_sequence.append(v.iloc[:][ y_episode_column].values)
        return X_sequence, y_sequence

    def constructEpisodes(self, df):
        subjects_group = df.groupby(['id_subject'])
        X_all = []
        y_all = []
        for s in subjects_group:
            for e in s[1].groupby(['episode_label']):
                X_seq, y_seq = self.splitEpisode(e[1])

                X_all = X_all + X_seq
                y_all = y_all + y_seq

        return X_all, y_all

    def getAllData(self):

        self.constructEpisodes(self.df)



if __name__ == '__main__':

    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

    features_reader = SingleFeaturesReader(path)

    features_reader.getAllData()
