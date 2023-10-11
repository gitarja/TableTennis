import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoLarsIC

def skillClassification(skill):
    skill_class =  np.zeros_like(skill)
    skill_class[(skill> 0) & (skill <= 0.25)] = 0
    skill_class[(skill > 0.25) & (skill <= 0.5)] = 1
    skill_class[(skill > 0.5) & (skill <= 0.75)] = 2
    skill_class[skill > 0.75] = 3

    return skill_class

results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\"

df = pd.read_csv(results_path + "single_summary.csv")

df = df[(df["norm_score"] > 0.5) & (df["Tobii_percentage"] > 65)]


df.to_csv("Experiment\\included_data.csv")

# skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
#
#
#
# X = df.values
# y = skillClassification(df["skill"].values)
#
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     train_data = df.iloc[train_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)
#     test_data = df.iloc[test_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)
#
#     train_data.to_csv("Experiment\\single_train_"+str(i)+".csv")
#     test_data.to_csv("Experiment\\single_test_"+str(i)+".csv")