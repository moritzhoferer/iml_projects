#! /usr/bin/python3

import numpy as np 
import pandas as pd 

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

random_state = np.random.RandomState(0)

amino_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

char2int = {
    v: k
    for k, v in enumerate(amino_list)
}

int2h1 = {}
for integer in char2int.values():
    h = np.zeros(len(char2int), dtype=int)
    h[integer] = 1
    int2h1[integer] = h
    
char2h1 = {}
for character, integer in char2int.items():
    char2h1[character] = list(int2h1[integer])


def prepare_df(df: pd.DataFrame):
    dict_ = np.zeros([df.shape[0], 80])
    for r_name, row in df.iterrows():
        l = []
        for x in row['Sequence']:
            l += char2h1[x]
        dict_[r_name] = l
    return dict_


if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    train_f = prepare_df(train_df)
    test_f = prepare_df(test_df)

    train_l = np.array(train_df['Active'])
    upsample_factor = (len(train_l) - sum(train_l))//sum(train_l)
    upsample_dict = {0: 1, 1: upsample_factor}
    new_indices = np.array([k for k, v in enumerate(train_l) for _ in range(upsample_dict[v])])
    np.random.shuffle(new_indices)

    # Normalize features (UPSAMPLED)
    mu, rho = .5, .5
    norm_train_f = (train_f - mu) / rho
    norm_test_f = (test_f - mu) / rho

    model_ann = MLPClassifier(
        hidden_layer_sizes=4*(2048),
        activation='tanh',
        solver='adam',
        alpha=1e-3,
        max_iter=100000,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
    )
    train_index = np.array([k for k in range(len(train_l)) for _ in range(upsample_dict[train_l[k]])])
    np.random.shuffle(train_index)
    model_ann.fit(
        norm_train_f[train_index],
        train_l[train_index]
    )
    np.savetxt(
        'submission.csv',
        model_ann.predict(norm_test_f),
        fmt='%d',
    )
    score = f1_score(
                y_true=train_l,
                y_pred=model_ann.predict(norm_train_f)
            )
    print('F1 score: ', score)
