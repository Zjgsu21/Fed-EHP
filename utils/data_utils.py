import ujson
import numpy as np
import os
import torch

# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# NUM_CHANNELS = 1

# IMAGE_SIZE_CIFAR = 32
# NUM_CHANNELS_CIFAR = 3


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    # np.random.seed(100)
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset, idx, is_train=True, is_public=False):
    # 训练数据集
    if is_train and not is_public:
        # train_data_dir = os.path.join('../dataset', dataset, 'train/')
        train_data_dir = '/'.join(('../dataset', dataset, 'train/'))
        # print(f"train_data_dir:{train_data_dir}")

        train_file = train_data_dir + 'train' + str(idx) + '_' + '.npz'

        # print(f"train_file:{train_file}")
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data
    # 测试数据集
    elif not is_train and not is_public:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + 'test' + str(idx) + '_' + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return test_data

    # 公共数据集
    elif is_public:
        public_data_dir = os.path.join('../dataset', dataset)

        public_data_file = public_data_dir + '/public_set.npz'
        with open(public_data_file, 'rb') as f:
            public_data = np.load(f, allow_pickle=True)['data'].tolist()
        # print("-" * 50, '\n',
        #       type(public_data), public_data)
        # exit()
        return public_data


def read_client_data(dataset, idx, is_train=True, is_public=False):
    # 对于agnews数据集 单独处理。
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx)
    # print("dataset:", dataset)
    if is_train and not is_public:
        train_data = read_data(dataset, idx, is_train, is_public)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data

    elif not is_train and not is_public:
        test_data = read_data(dataset, idx, is_train, is_public)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

    elif is_public:     # 公共数据集
        public_data = read_data(dataset, 0, is_train, is_public)
        # print(public_data)
        X_public = torch.Tensor(public_data['x']).type(torch.float32)
        # print("-"*50,'\n',
        #     X_public)
        y_public = torch.Tensor(public_data['y']).type(torch.int64)
        public_data = [(x, y) for x, y in zip(X_public, y_public)]
        return public_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        # y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data
