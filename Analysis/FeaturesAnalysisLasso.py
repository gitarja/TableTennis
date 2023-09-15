import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

import matplotlib.pyplot as plt
from sklearn import linear_model
from Analysis.Lib import transformScore
from sklearn.linear_model import LassoLarsIC
from Analysis.Conf import x_column, y_column
import statsmodels.api as sm


ss = StandardScaler()
r2_score = []
aic_score = []
bic_score = []
alphas = np.arange(0.001, 0.003, 0.0005)
for a in alphas:
    r2_score_folds = []
    for i in range(5):
        train_data = pd.read_csv("Experiment\\train_" + str(i) + ".csv")
        test_data = pd.read_csv("Experiment\\test_" + str(i) + ".csv")

        # transform score
        train_data = transformScore(train_data)
        test_data = transformScore(test_data)
        # load and prepare the data
        X_train = train_data[x_column].values
        y_train = train_data[y_column].values

        X_test = test_data[x_column].values
        y_test = test_data[y_column].values

        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)

        #r2
        clf = linear_model.Lasso(alpha=a, max_iter=10000)
        clf.fit(X_train, y_train)

        r2_score_folds.append(clf.score(X_test, y_test))

    r2_score.append(np.average(r2_score_folds))




# save lasso coefficients
a = alphas[np.argmax(r2_score)]
coefficients_folds = []
for i in range(5):
    train_data = pd.read_csv("Experiment\\train_" + str(i) + ".csv")
    test_data = pd.read_csv("Experiment\\test_" + str(i) + ".csv")

    # transform score
    train_data = transformScore(train_data)
    test_data = transformScore(test_data)
    # load and prepare the data
    X_train = train_data[x_column].values
    y_train = train_data[y_column].values

    X_test = test_data[x_column].values
    y_test = test_data[y_column].values

    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    clf = linear_model.Lasso(alpha=a, max_iter=10000)
    clf.fit(X_train, y_train)

    coefficients_folds.append(np.abs(clf.coef_))


#
results_path = "F:\\users\\prasetia\\results\\TableTennis\\"
# save alpha score
results_coeff = pd.DataFrame({'features': x_column, 'coeff': np.average(coefficients_folds, 0)})

results_coeff.to_csv(results_path + "r2_coeff.csv")


# save alpha score

results_alpha = pd.DataFrame({'alphas': alphas, 'r2_score': np.array(r2_score)})


results_alpha.to_csv(results_path + "r2_score.csv")

