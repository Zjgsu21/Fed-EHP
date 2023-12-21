import random
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from clientbase import Client
from utils.data_utils import read_client_data

"""
1. 坐标
2. 数据分布
3. 与雾节点连接
4. 本地训练
5. 上传模型
6. 更新模型
"""


class client_KTpFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # 随机产生坐标
        self.coords = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
        # self.coords = [1, 1]

        self.distribution = []
        self.train_data = self.load_train_data(batch_size=self.batch_size)



    def train(self):
        # trainloader = self.load_train_data()

        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(self.train_data):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        # print(f"client {self.id} local training:", self.model.state_dict())

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def get_distribution(self, data):
        # 1.概率分布
        list = []
        for i in range(len(data)):
            list.append(data[i][1].numpy().tolist())
        prob = get_prob(list, self.num_classes)
        print(f"probability of client {self.id}: ", prob)
        print(len(prob))
        return prob

        # data.append(lit)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)

        # 求 概率分布
        self.distribution = self.get_distribution(train_data)

        data = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
        # print(type(data))
        return data


def get_prob(data, class_num):
    """求概率"""
    prob = [0 for i in range(class_num)]

    for i in data:
        prob[i] += 1

    for i in range(class_num):
        prob[i] = round(prob[i] / len(data), 4)

    # print(prob)
    return prob