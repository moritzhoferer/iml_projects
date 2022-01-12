#! /usr/bin/python3

# Import necessary packages
import numpy as np
import pandas as pd

# Definition of given feature transformations
lin_f = lambda x: x
quad_f = lambda x: x**2
exp_f = lambda x: np.exp(x)
cos_f = lambda x: np.cos(x)

# Function to get the set of feature transformations
def feature_transformation(df):
    # Execute all feature transformations in the right order
    new_df = pd.concat(
        [lin_f(df), quad_f(df), exp_f(df), cos_f(df)],
        axis=1
    )
    # Rename the column name to fit the task
    new_df.columns = ['x{i:d}'.format(i=i+1) for i, _ in enumerate(new_df.columns)]
    # Add 21st column with constant value
    new_df['x{i:d}'.format(i=new_df.shape[1]+1)] = 1
    return new_df


if __name__ == "__main__":
    from sklearn.linear_model import Lasso

    # Set seed for reproducibility
    np.random.seed(4)

    # Load data and separate features and labels
    features = pd.read_csv('./train.csv', index_col=0)
    labels = features.pop('y')
    # Feature transformation of the five feature for five functions
    f = feature_transformation(features)

    model = Lasso(
        alpha=.05,
        fit_intercept=False,
        max_iter=int(1e6),
        tol=1e-6,
        random_state=None,
    )
    model.fit(f, labels)
    # Save coefficients for submission
    np.savetxt('./submission.csv', model.coef_)
