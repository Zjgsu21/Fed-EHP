import copy
import time
import random
import torch
from torch.utils.data import DataLoader
from clients.clientavg import clientAVG
from servers.serverbase import Server
from torch import nn
import torch.nn.functional as F
from utils.data_utils import read_client_data
from torch.optim import SGD
from threading import Thread
from tqdm import tqdm
import numpy as np

def KL_loss(inputs, target, reduction='average'):
    log_likelihood = F.log_softmax(inputs, dim=1)
    # print('log_probs:',log_likelihood)
    # batch = inputs.shape[0]
    if reduction == 'average':
        # loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        loss = F.kl_div(log_likelihood, target, reduction='mean')
    else:
        # loss = torch.sum(torch.mul(log_likelihood, target))
        loss = F.kl_div(log_likelihood, target, reduction='sum')
    return loss


class fogNode(Server, clientAVG):
    def __init__(self, args, id, publicset_samples):
        super().__init__(args, id)

        # select slow clients   暂时不用
        # self.set_slow_clients()
        # self.set_clients(args, clientTest)
        self.id = id
        self.clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # 随机产生坐标
        self.coords = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
        # self.coords = [1, 1]
        # 与客户端的距离
        self.dis_to_clients = []
        # self.clients = self.select_clients()

        # 学习率 wn ← wn − η2∇wnLKL
        self.learning_rate2 = args.fog_learning_rate
        # 雾节点聚合器 聚合次数
        self.k_n2c = args.knf

        # 概率分布
        self.distribution = []

        # TODO 模型异构性
        self.model = args.model
        # self.fog_u = copy.deepcopy()

        self.publicset_samples = publicset_samples
        self.public_data = self.load_public_data()

        # logit
        self.logit = []
        # 温度
        self.Temp = args.Temp
        self.N_logits_matching_round = args.N_logits_matching_round

        self.loss_fun = nn.CrossEntropyLoss()
        self.criterion = nn.KLDivLoss()  # KL散度
        self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate2)

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished linking fog node and clients.")
        # self.load_model()

    def train(self, global_round):
        start_time = time.time()
        print('Distillation_updata before....')
        # self.evaluate_()
        if global_round != 0:
            self.Distillation_updata()

        print('Distillation_updata after....')

        for i in range(self.k_n2c):
            self.send_models()

            # if i % self.eval_gap == 0 and i != 0:
            #     print(f"\n-------------{self.id} Fog Round number: {i}-------------")
            #     print("\nEvaluate fog node model")
            #     self.evaluate_()

            for client in self.clients:
                # print("=client.id"*50, client.id, "="*50)
                client.train()

            self.receive_model()
            self.aggregate_parameters()

        self.knowledge_transfer()

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

    def send_models(self):
        # 断言
        # 等价于 if not expression:
        #     raise AssertionError
        assert (len(self.clients) > 0)

        for client in self.clients:
            # 发送模型
            client.set_parameters(copy.deepcopy(self.model))

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
            # self.uploaded_weights[i] = round(w / tot_samples, 4)
            self.uploaded_weights[i] = w / tot_samples

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

    def knowledge_transfer(self):
        print("model {0} starting alignment with public logits... ".format(self.model))
        print(f"update logits of fog_node {self.id}... ")

        self.model.to(self.device)
        self.model.eval()
        out = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.public_data):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.model.zero_grad()
                logit = self.model(x)
                Tsoftmax = nn.Softmax(dim=1)
                # 加入温度系数Temp#
                output_logit = Tsoftmax(logit.float() / self.Temp)
                output_logit = output_logit.to(torch.float32)
                out.append(output_logit.cpu().numpy())

            out = np.concatenate(out)

            # print(f'fog_node {self.id} logit:', out, out.shape, type(out))
            self.logit.clear()      # 清除logit
            self.logit.append(out)

    def load_public_data(self, batch_size=None):
        """
        载入公共数据集
        """
        if batch_size == None:
            batch_size = self.batch_size
        public_data = read_client_data(self.dataset, self.id, is_train=False, is_public=True)

        data = DataLoader(public_data, batch_size, drop_last=True, shuffle=True)
        # print(type(data))
        return data

    def Distillation_updata(self):
        # TODO
        self.model.to(self.device)
        self.model.train()
        updated_logit = torch.tensor(np.array(self.logit)).to(torch.float32)
        # updated_logit.to(self.device)
        idx = 0
        for iter in range(self.N_logits_matching_round):
            for i, (x, y) in enumerate(self.public_data):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.model.zero_grad()
                # 硬标签
                log_probs = self.model(x)
                loss_hard = self.loss_fun(F.log_softmax(log_probs), y)
                # 软标签
                output_logit = log_probs.float() / self.Temp
                output_logit = output_logit.to(torch.float32)
                # print("1,", output_logit.shape[0], updated_logit[0:output_logit.shape[0]].shape)
                # exit()
                logit = updated_logit[idx:idx+output_logit.shape[0]].to(self.device)
                loss_soft = self.criterion(output_logit, logit)
                loss_soft = loss_soft.to(torch.float32)

                loss = loss_hard*0.1 + 0.9*loss_soft

                # print("loss", loss)

                loss.backward()
                self.optimizer.step()

                idx = idx + output_logit.shape[0]






