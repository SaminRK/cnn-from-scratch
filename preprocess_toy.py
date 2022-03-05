import numpy as np
import pandas as pd


def get_toy(one_hot=False):
    file_dir = 'datasets/Toy Dataset'
    
    train_ds = pd.read_csv(f'{file_dir}/trainNN.txt', sep=r'\s+', header=None)
    train_data = np.vstack((train_ds[0], train_ds[1], train_ds[2], train_ds[3])).T
    train_labels = train_ds[4].to_numpy()

    test_ds = pd.read_csv(f'{file_dir}/testNN.txt', sep=r'\s+', header=None)
    test_data = np.vstack((test_ds[0], test_ds[1], test_ds[2], test_ds[3])).T
    test_labels = test_ds[4].to_numpy()

    toy = {}
    toy['train_data'] = train_data
    toy['train_labels'] = train_labels
    toy['test_data'] = test_data
    toy['test_labels'] = test_labels

    if one_hot:
        from sklearn.preprocessing import OneHotEncoder
    
        enc = OneHotEncoder()
        one_hot_encoded = enc.fit_transform(toy['train_labels'].reshape(-1, 1)).toarray()
        toy["train_one_hot"] = one_hot_encoded

        one_hot_encoded = enc.fit_transform(toy['test_labels'].reshape(-1, 1)).toarray()
        toy["test_one_hot"] = one_hot_encoded
    

    return toy

if __name__ == '__main__':
    toy_ds = get_toy(one_hot=True)
    print(toy_ds['train_one_hot'].shape)