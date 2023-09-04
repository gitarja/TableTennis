import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

results_path = "F:\\users\\prasetia\\results\\TableTennis\\"

df = pd.read_csv(results_path + "single_anticipation_scores.csv")

# scaler = StandardScaler()
# df["task_score"] = scaler.fit_transform(np.expand_dims(df["task_score"].values, -1))
# df["skill"] = scaler.fit_transform(np.expand_dims(df["skill"].values, -1))
# df["n_success"] = scaler.fit_transform(np.expand_dims(df["n_success"].values, -1))
df = df[(df["norm_score"] > 0.5) & (df["percentage"] > 65)]


df[["norm_score",
    "skill",
    "task_score",
    "max_seq",
    "avg_seq"]] = df[["norm_score",
    "skill",
    "task_score",
    "max_seq",
    "avg_seq"]].apply(np.log)

columns = [
    # "n_success",
    # "n_failures",
    # "norm_score",
    # "skill",
    # "task_score",
    # "max_seq",
    # "avg_seq",

    # "bounce_hull",
    # "bounce_std",
    # "bounce_sp_entropy",
    # "bounce_sc_entropy",

    # "rt_lyp",
    # "samp_en",
    # "std_rt",
    # "var_p1",
    # "var_p2",
    # "var_p3",

    "avg_start_fs",
    "std_start_fs",


    "p1_on",
    "p1_off",
    "p1_dn",
    "p1_mda",
    "p1_md",
    "p1_mid",
    "p1_mad",
    "p1_mm",
    "p1_sm",
    "p1_gm",
    "p1_occ",


    # "p2_on",
    # "p2_off",
    # "p2_dn",
    # "p2_mda",
    # "p2_md",
    # "p2_mid",
    # "p2_mad",
    # "p2_mm",
    # "p2_sm",
    # "p2_gm",
    # "p2_occ"
]


scaled_subset_df = df[columns]

# perform standarization
# ss = StandardScaler()
# scaled_subset_df = pd.DataFrame(ss.fit_transform(subset_df),columns = subset_df.columns)






g = sns.PairGrid(scaled_subset_df)
g.map_upper(sns.regplot, line_kws={"color": "#e41a1c", 'ls':'--'}, scatter_kws={'s':5}, x_jitter=0.01, y_jitter=0.01)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)

path = "F:\\users\\prasetia\\results\\TableTennis\\"
plt.savefig(path + 'fs_al_p1.png', bbox_inches='tight')