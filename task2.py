#! /usr/bin/python3

# Data handling
import os
import numpy as np
import pandas as pd
import random

# Learning
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import roc_auc_score

from score_submission import get_score

# For reporducability
random_state = np.random.RandomState(0)

# Store order of column labels for latter
feature_column_names = ['Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3',
                        'BaseExcess', 'RRate', 'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine',
                        'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2',
                        'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate', 'Bilirubin_total',
                        'TroponinI', 'ABPs', 'pH']

# Label as required by the instruction
labels_task_a = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
labels_task_b = 'LABEL_Sepsis'
labels_task_c = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

kfold = KFold(
    n_splits=5,
    shuffle=False,
#     random_state=random_state,
)


# Relevant models
def get_svc_model(alpha=1.):
    return SVC(
        C=alpha,
        probability=True,
        tol=1e-3,
        random_state=random_state,
        cache_size=24*1024,
        max_iter=-1,
    )

batch_size_ = 64
verbose_ = False
validation_fraction_ = .2
tol_ = 1e-3
early_stopping_ = True


def get_nn_regressor_model(layers=2, npl=256, alpha=1e-4):
    return MLPRegressor(
        hidden_layer_sizes=layers * (npl),
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size=batch_size_,
        random_state=random_state,
        max_iter=int(1e3),
        tol=1e-3,
        early_stopping=early_stopping_,
        validation_fraction=validation_fraction_,
        verbose=verbose_,
    )


def get_nn_classifier_model(layers=2, npl=256, alpha=1e-4):
    return MLPClassifier(
        hidden_layer_sizes=layers * (npl),
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size=batch_size_,
        max_iter=1000,
        random_state=random_state,
        tol=1e-3,
        early_stopping=early_stopping_,
        validation_fraction=validation_fraction_,
        verbose=verbose_,
    )


def get_agg_df(df, kind='last') -> pd.DataFrame:
    if kind in ['last', 'mean']:
        feature_column_names = df.columns
        dict_ = {}
        for pid_, p_df in df.groupby('pid'):
            if kind == 'last':
                dict_[pid_] = {
                    label: s.loc[s.last_valid_index()] if s.last_valid_index() else 0
                    for label, s in p_df.iteritems()
                }
            elif kind == 'mean':
                dict_[pid_] = {
                    l: v if not np.isnan(v) else 0
                    for l, v in p_df.mean(axis=0).iteritems()
                }
        _out_df = pd.DataFrame(dict_).T[feature_column_names]
    else:
        # raise IOError('Wrong input parameter "kind"!')
        raise Warning('Not-defined import paramter "kind", using "mean" instead.')
        # _out_df = get_agg_df(df, kind='mean')
    return _out_df


def get_pp_data(df, n_neighbors=5):
    _new_dict = {
        pid: sub_df.median(axis=0)
        for pid, sub_df in df.groupby('pid')
    }
    _new_df = pd.DataFrame(_new_dict).T
    
    _stats = _new_df.describe()
    _new_df_normed = (_new_df - _stats.loc['mean']) / _stats.loc['std']
    
    _imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    _imputed_df = pd.DataFrame(
        _imputer.fit_transform(_new_df_normed),
        index=_new_df.index,
        columns=_new_df.columns
    )
    _imputed_stats = _imputed_df.describe()
    return (_imputed_df - _imputed_stats.loc['mean']) / _imputed_stats.loc['std']


def get_features(df, max_degree=1):
    handle = '{name:s}_{count:d}'
    dict_ = {}
    for pid, sub_df in df.groupby('pid'):
        pid_dict = {}
        for m_name, m_value in sub_df.loc[pid].iteritems():
            m_value.index -= min(m_value.index)
            r_values = m_value.dropna()
            if r_values.shape[0] > 1:
                fit_params = np.append(
                    np.polyfit(r_values.index, r_values, max_degree),
                    m_value.median()
                )
                for count, c in enumerate(fit_params):
                    pid_dict[handle.format(name=m_name, count=len(fit_params)-1-count)] = c
            elif r_values.shape[0] == 1:
                for count in range(max_degree+1):
                    pid_dict[handle.format(name=m_name, count=max_degree+1-count)] = 0
                pid_dict[handle.format(name=m_name, count=0)] = m_value.median()
            else:
                for count in range(max_degree+1):
                    pid_dict[handle.format(name=m_name, count=max_degree+1-count)] = 0
                pid_dict[handle.format(name=m_name, count=0)] = np.nan
        dict_[pid] = pid_dict
    return pd.DataFrame(dict_).T

# Training features
train_features = pd.read_csv('./train_features.csv', index_col=['pid', 'Time'])
train_features = train_features.sort_index(level=['pid', 'Time'])

# Training labels
train_labels = pd.read_csv('./train_labels.csv', index_col='pid')
train_labels = train_labels.sort_index(level='pid')
train_labels_a = train_labels[labels_task_a]
train_labels_b = train_labels[labels_task_b]
train_labels_c = train_labels[labels_task_c]

# Test feauters
test_features = pd.read_csv('./test_features.csv', index_col=['pid', 'Time'])
# test_features = test_features.sort_index(level=['pid', 'Time'])

default_dist = .01
md = 2

new_train_df = get_features(train_features, max_degree=md)
new_train_df = new_train_df.drop(['Age_3', 'Age_2', 'Age_1'], axis=1)
new_test_df = get_features(test_features, max_degree=md)
new_test_df = new_test_df.drop(['Age_3', 'Age_2', 'Age_1'], axis=1)
stats_f = pd.concat([new_test_df, new_train_df], axis=0).describe()

fill_values = (1.+default_dist) * stats_f.loc['min'] - default_dist * stats_f.loc['max']
filled_train_df = new_train_df.fillna(fill_values)
filled_test_df = new_test_df.fillna(fill_values)
# stats_f = pd.concat([filled_train_df, filled_test_df], axis=0).describe()
norm_train_features = (filled_train_df - stats_f.loc['mean']) / stats_f.loc['std']
norm_test_features = (filled_test_df - stats_f.loc['mean']) / stats_f.loc['std']

# Add. for task 2
l_min = train_labels_b[train_labels_b==1]
l_maj = train_labels_b[train_labels_b==0]
f_min = norm_train_features.loc[l_min.index]
f_maj = norm_train_features.loc[l_maj.index]

# Add. for task 3
stats_labels_c = train_labels_c.describe()
norm_train_labels_c = (train_labels_c - stats_labels_c.loc['mean']) / stats_labels_c.loc['std']

if __name__ == '__main__':
    # Training
    ## Task 1
    clf_a = OneVsRestClassifier(get_svc_model(alpha=.1), n_jobs=-1)
    clf_a.fit(norm_train_features, train_labels_a)

    pred_a = clf_a.predict_proba(norm_train_features)
    pred_a_df = pd.DataFrame(pred_a, index=norm_train_features.index, columns=train_labels_a.columns)
    truth_a = train_labels_a.values
    print("For task 1, AU ROC score: ", roc_auc_score(y_true=truth_a, y_score=pred_a))

    ## Task 2
    train_min_index = random.choices(f_min.index, k=f_maj.shape[0])

    train_f = pd.concat([f_maj, f_min.loc[train_min_index]])
    train_l = pd.concat([l_maj, l_min.loc[train_min_index]])

    stats_2 = train_f.describe()
    train_f = (train_f - stats_2.loc['mean']) / stats_2.loc['std']
    test_f = (norm_test_features - stats_2.loc['mean']) / stats_2.loc['std']  

    clf_b = get_svc_model(alpha=.1)
    clf_b.fit(train_f, train_l)

    pred_b = clf_b.predict_proba((norm_train_features-stats_2.loc['mean'])/stats_2.loc['std'])[:,1]
    pred_b_s = pd.Series(pred_b, index=norm_train_features.index)
    pred_b_s.name = 'LABEL_Sepsis'
    true_b = train_labels_b.values
    print("For task 2, AU ROC score: ", roc_auc_score(y_true=true_b, y_score=pred_b))

    ## Task 3
    model_c = MLPRegressor(
        hidden_layer_sizes= 16 * (512),
        activation='tanh',
        solver='adam',
        alpha=1.,
        batch_size=128,
        random_state=random_state,
        max_iter=int(1e4),
        early_stopping=True,
        validation_fraction=.2,
        verbose=False,
    )
    model_c.fit(norm_train_features, norm_train_labels_c)
    pred_c = model_c.predict(norm_train_features)
    pred_c_df = pd.DataFrame(pred_c, index=norm_train_features.index, columns=train_labels_c.columns)
    pred_c_df *= stats_labels_c.loc['std']
    pred_c_df += stats_labels_c.loc['mean']
    print("For task 3, R2 score: ", model_c.score(norm_train_features, norm_train_labels_c))
    
    ## Score training data
    pred_train_df = pd.concat([pred_a_df, pred_b_s, pred_c_df], axis=1)
    pred_train_df.index.name = 'pid'
    print(get_score(train_labels, pred_train_df))

    # Prediction
    ## Task 1
    pred_test_a = clf_a.predict_proba(norm_test_features)
    pred_test_a_df = pd.DataFrame(pred_test_a, index=norm_test_features.index, columns=train_labels_a.columns)

    ## Task 2
    pred_test_b = clf_b.predict_proba(test_f)
    pred_test_b_df= pd.DataFrame(pred_test_b, index=test_f.index, columns=['yes', 'no'])
    pred_test_b_s = pred_test_b_df.no
    pred_test_b_s.name = 'LABEL_Sepsis'

    ## Task 3
    pred_test_c = model_c.predict(norm_test_features)
    pred_test_c_df = pd.DataFrame(pred_test_c, index=norm_test_features.index, columns=train_labels_c.columns)
    pred_test_c_df *= stats_labels_c.loc['std']
    pred_test_c_df += stats_labels_c.loc['mean']

    # Export results
    prediction_df = pd.concat([pred_test_a_df, pred_test_b_s, pred_test_c_df], axis=1)
    prediction_df.index.name = 'pid'
    prediction_df = prediction_df.sort_index()
    sample_df = pd.read_csv('./sample.zip', index_col='pid')
    prediction_df = prediction_df.loc[sample_df.index]
    prediction_df.to_csv('./submission.csv', index=True, float_format='%.5f')
