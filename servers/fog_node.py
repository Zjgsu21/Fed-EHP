import copy
import time
from math import sqrt
import random

import torch

from clients.clientamp import clientAMP
from servers.serverbase import Server

def fun(before_model, after_model):

    for before, after in zip(before_model.parameters(), after_model.parameters()):
        ddd = before.data - after.data
        print("*"*100, ddd)


class fogNode(Server, clientAMP):
    def __init__(self, args, id):
        super().__init__(args, id)

        # select slow clients   暂时不用
        # self.set_slow_clients()
        # self.set_clients(args, clientTest)
        self.id = id
        self.clients = []
        # 雾节点聚合器 聚合次数
        self.k_n2c = args.knf

        # 随机产生坐标
        self.coords = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
        # 与客户端的距离
        self.dis_to_clients = []
        # self.clients = self.select_clients()

        # 概率
        self.distribution = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.model = copy.deepcopy(args.model)
        # self.fog_u = copy.deepcopy()

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished linking fog node and clients.")

        # self.load_model()


    def train(self):
        start_time = time.time()
        for i in range(self.k_n2c):
            self.send_models()
            # /
            # before_model = copy.deepcopy(self.model)
            # for para in before_model.parameters():
            #     print("AMP_model_para : {} type : {} size : {}".format(para, type(para), para.size()))
            # print("done!")

            if i % self.eval_gap == 0 and i != 0:
                print(f"\n-------------{self.id} Fog Round number: {i}-------------")
                print("\nEvaluate fog node model")
                self.evaluate_()

            for client in self.clients:
                # print("=client.id"*50, client.id, "="*50)
                client.train()

            self.receive_model()
            self.aggregate_parameters()

            # /
            # after_model = copy.deepcopy(self.model)
            #
            # for para in after_model.parameters():
            #     print("model_AVG_para5 : {} type : {} size : {}".format(para, type(para), para.size()))
            # print("done!")


            # 求是否更新 !有更新
            # fun(before_model, after_model)
            #
            # exit()
        # print("\nBest fog model accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()

    def send_models(self):
        # 断言
        # 等价于 if not expression:
        #     raise AssertionError
        assert (len(self.clients) > 0)

        for client in self.clients:
            # 发送模型
            client.set_parameters(copy.deepcopy(self.model))

    def set_parameters(self, model, coef_self):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = (new_param.data + coef_self * old_param.data).clone()

    def receive_model(self):
        assert (len(self.clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []

        for client in self.clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(copy.deepcopy(client.model))

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = round(w / tot_samples, 4)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # 雾节点模型
        self.model = copy.deepcopy(self.uploaded_models[0])
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):

        for fog_param, client_param in zip(self.model.parameters(), client_model.parameters()):
            fog_param.data += client_param.data.clone() * w

        self.model.state_dict()


    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []

        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def evaluate_(self):
        stats = self.test_metrics()

        # stats_train = self.train_accuracy_and_loss()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])

        self.rs_test_acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        # self.rs_train_loss.append(train_loss)

        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))



