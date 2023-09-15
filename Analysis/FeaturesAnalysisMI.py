import numpy as np
import pandas as pd
from Analysis.Lib import transformScore
from Analysis.Conf import x_column, y_column, x_lasso_column
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
results_path = "F:\\users\\prasetia\\results\\TableTennis\\"


ss = StandardScaler()
mi_folds = []
for i in range(5):
    train_data = pd.read_csv("Experiment\\train_" + str(i) + ".csv")

    # transform score
    train_data = transformScore(train_data)
    # load and prepare the data
    X_train = train_data[x_lasso_column].values
    y_train = train_data[y_column].values


    ss.fit(X_train)
    X_train = ss.transform(X_train)
    mi= mutual_info_regression(X_train, y_train.flatten())
    mi_folds.append(mi)

results_path = "F:\\users\\prasetia\\results\\TableTennis\\"
# save alpha score
results_coeff = pd.DataFrame({'features': x_lasso_column, 'MI': np.average(mi_folds, 0)})

results_coeff.to_csv(results_path + "MI_coeff.csv")

