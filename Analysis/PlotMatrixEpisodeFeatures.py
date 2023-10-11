import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Conf import x_episode_columns
from scipy import stats
import numpy as np

sns.set()
results_path =  "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\"


df_features = pd.read_pickle(results_path + "single_episode_features.pkl")
df_subjects = pd.read_csv("Experiment\\included_data.csv")

included_subjects = df_subjects["Subject1"].values

filtered_features = df_features.loc[df_features["id_subject"].isin(included_subjects), :]
filtered_features = filtered_features.loc[filtered_features["success"] != -1, :]

row = 4
col = 10
fig, axes = plt.subplots(row, col, figsize=(20, 10))
k= 0
for i in range(row):
    for j in range(col):
        if k < len(x_episode_columns):
            x = filtered_features[x_episode_columns[k]].values
            y = filtered_features["success"].values
            xx = x[~np.isnan(x)]
            yx = y[~np.isnan(x)]
            sns.violinplot(ax=axes[i, j], data=filtered_features, x='success', y=x_episode_columns[k], split=True, cut=0, bw=0.15)
            # axes[i, j].violinplot([xx[yx==0], xx[yx==1]])
            print(x_episode_columns[k])
            print(stats.pearsonr(xx, yx))
        # sns.stripplot(ax=axes[i, j], data=filtered_features, x='success', y=x_episode_columns[k+15],
        #               size=2, color=".3", linewidth=0)
        k+=1


plt.tight_layout()
plt.show()
print(len(df_features))
print(len(filtered_features))