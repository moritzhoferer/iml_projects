#! /usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

np.random.seed(0)
lambda_list = 10. ** np.arange(-2,3)
kfold = KFold(n_splits=10, shuffle=False)

# Load data and separate labels
data = pd.read_csv('./train.csv', index_col=0)

# Separate labels from features
labels = data.pop('y')

# Normalize features
stats = data.describe()
# data = (data - stats.loc['mean']) / stats.loc['std']

output_data = []
rmse = []
for alpha in lambda_list:
    se = []
    ridge_model = Ridge(
        alpha=alpha,
        fit_intercept=True,
    )
    # weights = []
    # Loop for 10-fold cross-validation
    for train, test in kfold.split(data):
        # Perform ridge regression
        ridge_model.fit(data.loc[train], labels.loc[train])
        pred = ridge_model.predict(data.loc[test])
        truth = labels.loc[test]
        residuals = pred - truth
        se += list(residuals**2)

    # Calculate average RMSE scores and store it
    # Weights are used as the fold size varies between 50 and 51 feature sets/labels.
    rmse.append(np.sqrt(np.mean(se)))
    # output_data.append(np.average(rsme, weights=weights))

np.savetxt('./submission.csv', rmse)
