import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoLarsIC
from Analysis.Conf import x_column, y_column

results_path = "F:\\users\\prasetia\\results\\TableTennis\\"

df = pd.read_csv(results_path + "single_anticipation_scores.csv")

df = df[(df["norm_score"] > 0.5) & (df["percentage"] > 65)]


skf = KFold(n_splits=5, random_state=None, shuffle=True)



X = df.values
y = df["skill"].values

# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     train_data = df.iloc[train_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)
#     test_data = df.iloc[test_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)
#
#     train_data.to_csv("Experiment\\train_"+str(i)+".csv")
#     test_data.to_csv("Experiment\\test_"+str(i)+".csv")