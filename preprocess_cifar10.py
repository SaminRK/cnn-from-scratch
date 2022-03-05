import numpy as np
import matplotlib.pyplot as plt
import pickle

# inspired by: https://github.com/snatch59/load-cifar-10


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def get_cifar10(negatives=False, one_hot=False):
    DATA_DIR = 'datasets/cifar-10-python/cifar-10-batches-py'

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(f"{DATA_DIR}/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(f"{DATA_DIR}/data_batch_{i}")
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(f"{DATA_DIR}/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    cifar = {}
    cifar['train_data'] = cifar_train_data
    cifar['train_labels'] = cifar_train_labels
    cifar['test_data'] = cifar_test_data
    cifar['test_labels'] = cifar_test_labels
    cifar['train_filenames'] = cifar_train_filenames
    cifar['test_filenames'] = cifar_test_filenames
    cifar['cifar_label_names'] = cifar_label_names

    if one_hot:
        from sklearn.preprocessing import OneHotEncoder
    
        enc = OneHotEncoder()
        one_hot_encoded = enc.fit_transform(cifar['train_labels'].reshape(-1, 1)).toarray()
        cifar["train_labels_one_hot"] = one_hot_encoded

        one_hot_encoded = enc.fit_transform(cifar['test_labels'].reshape(-1, 1)).toarray()
        cifar["test_labels_one_hot"] = one_hot_encoded

    return cifar


if __name__ == "__main__":
    cifar = get_cifar10(one_hot=True)

    print("Train data: ", cifar['train_data'].shape)
    print("Train filenames: ", cifar['train_filenames'].shape)
    print("Train labels: ", cifar['train_labels'].shape)
    print("Train one hot: ", cifar['train_labels_one_hot'].shape)
    print("Test data: ", cifar['test_data'].shape)
    print("Test filenames: ", cifar['test_filenames'].shape)
    print("Test labels: ", cifar['test_labels'].shape)
    print("Test one hot: ", cifar['test_labels_one_hot'].shape)
    print("Label names: ", cifar['cifar_label_names'].shape)

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, cifar['train_data'].shape[0])
            ax[m, n].imshow(cifar['train_data'][idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()