import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from Analysis.Lib import transformScore
from sklearn.feature_selection import r_regression

results_path = "F:\\users\\prasetia\\results\\TableTennis\\"

df = pd.read_csv(results_path + "single_anticipation_scores.csv")

df = df[(df["norm_score"] > 0.5) & (df["percentage"] > 65)]

df = transformScore(df)

columns = [
"avg_start_fs",
"var_p1",
"fsp3_sampen",
"avg_p2",
"p1_std",
"std_rt",
"p1_mda",
"p1_occ",
"p1_sampen",
"p1_md",
"p1_off",
"var_p2",
"p1_gm",
"std_start_fs",
"p1_on",
"fsp3_mfs",
"skill",
]

scaled_subset_df = df[columns]

results_path = "F:\\users\\prasetia\\results\\TableTennis\\"

# g = sns.PairGrid(scaled_subset_df)
# g.map_upper(sns.regplot, line_kws={"color": "#e41a1c", 'ls': '--'}, scatter_kws={'s': 5}, x_jitter=0.01, y_jitter=0.01)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.histplot, kde=True)

# sns.pairplot(scaled_subset_df, x_vars=["avg_start_fs",
# "var_p1",
# "fsp3_sampen",
# "avg_p2",
# "p1_std",
# "std_rt",
# "p1_mda",
# "p1_occ",
# "p1_sampen",
# "p1_md",
# "p1_off",
# "var_p2",
# "p1_gm",
# "std_start_fs",
# "p1_on",
# "fsp3_mfs",], y_vars=["skill"],
#              height=5, aspect=.5, kind="reg")


fig, axs = plt.subplots(3, 5)

for ax, i in zip(axs.flat, range(15)):
    sns.regplot(x=y_column[0], y=x_lasso_column[i], data=scaled_subset_df,line_kws={"color": "#e41a1c", 'ls': '--'}, scatter_kws={'s': 5}, ax=ax)



plt.show()
# plt.savefig(results_path + 'important_features_lasso.png', bbox_inches='tight')


# ss = StandardScaler()
# coefficients_folds = []
# for i in range(5):
#     train_data = pd.read_csv("Experiment\\train_" + str(i) + ".csv")
#
#     # transform score
#     train_data = transformScore(train_data)
#     # load and prepare the data
#     X_train = train_data[x_lasso_column].values
#     y_train = train_data[y_column].values
#
#
#     ss.fit(X_train)
#     X_train = ss.transform(X_train)
#     coeff = r_regression(X_train, y_train.flatten())
#     coefficients_folds.append(coeff)
#
# results_path = "F:\\users\\prasetia\\results\\TableTennis\\"
# # save alpha score
# results_coeff = pd.DataFrame({'features': x_lasso_column, 'pearson_coeff': np.average(coefficients_folds, 0)})
#
# results_coeff.to_csv(results_path + "pearson_coeff.csv")
