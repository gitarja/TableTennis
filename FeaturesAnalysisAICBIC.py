import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoLarsIC
from Analysis.Conf import x_column, y_column
from Analysis.Lib import transformScore

results_path = "F:\\users\\prasetia\\results\\TableTennis\\"

df = pd.read_csv(results_path + "single_anticipation_scores.csv")

df = df[(df["norm_score"] > 0.5) & (df["percentage"] > 65)]
df = transformScore(df)


X = df[x_column].values
y = df[y_column].values

lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic")).fit(X, y.flatten())

results = pd.DataFrame(
    {
        "alphas": lasso_lars_ic[-1].alphas_,
        "AIC criterion": lasso_lars_ic[-1].criterion_,
    }
).set_index("alphas")
alpha_aic = lasso_lars_ic[-1].alpha_

lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(X, y.flatten())
results["BIC criterion"] = lasso_lars_ic[-1].criterion_
alpha_bic = lasso_lars_ic[-1].alpha_

results.to_csv(results_path + "AIC_BIC.csv")