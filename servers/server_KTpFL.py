import random

import scipy
import torch
import copy
import time
import numpy as np
import math
from torch import nn
from tqdm import tqdm
from clients.client_test import client_KTpFL
from servers.fog_node_KTpFL import fogNode
from servers.serverbase import Server
from threading import Thread
import torch.nn.functional as F

from utils.data_utils import read_client_data


def KL_loss(inputs, target, reduction='average'):
    log_likelihood = F.log_softmax(inputs, dim=1)

    if reduction == 'average':
        loss = F.kl_div(log_likelihood, target, reduction='mean')
    else:
        loss = F.kl_div(log_likelihood, target, reduction='sum')
    return loss


class Server_KTpFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.global_rounds = args.kfa
        self.candidate_modles = args.model
        # select slow clients
        self.set_slow_clients()
        # 选择用户
        self.clients = []
        self.set_clients(args, client_KTpFL)

        # 选择fog_node
        self.fog_nodes = []
        # fog number
        self.fog_num = args.fog_num
        self.public_data = read_client_data(self.dataset, 0, is_train=False, is_public=True)
        self.set_fogNode(args, fogNode)

        # fog -> node 权衡
        self.lambda_ = args.lambda_
        # self.lambda_ = 1000
        #
        self.alphaK = args.alphaK
        self.sigma = args.sigma

        # 知识系数矩阵
        self.weight_matrix = torch.ones(self.fog_num, self.fog_num, requires_grad=True)
        self.weight_matrix = self.weight_matrix.float() / (self.fog_num)

        self.uploaded_logits = [[] for i in range(self.fog_num)]

        # ρ
        self.penalty_ratio = 0.7
        print("Finished creating server , fog nodes and clients.")

    def train(self):
        # 分组
        self.group_client(self.fog_nodes, self.clients)

        # exit()

        for i in range(self.global_rounds + 1):
            # self.selected_clients = self.select_clients()
            if i != 0:
                self.send_logit()
            # /
            # before_model = copy.deepcopy(self.fog_us[0])

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")

                self.evaluate()


            for fog_node in tqdm(self.fog_nodes):
                print(f'fog node id:{fog_node.id}, fog node model:{fog_node.model}')
                fog_node.train(i)
            self.uploaded_logits, self.weight_matrix = self.receive_logits()

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

    def set_fogNode(self, args, fogNode):
        print("self.num_clients,", self.fog_num)
        for i, train_slow, send_slow in zip(range(self.fog_num), self.train_slow_clients, self.send_slow_clients):
            fog_node = fogNode(args,
                               id=i,
                               publicset_samples=len(self.public_data)
                               )
            self.fog_nodes.append(fog_node)

    def group_client(self, fog_nodes, clients):
        assert (len(fog_nodes) > 0)
        # print(self.num_clients)
        # print("len(fog_nodes):", len(fog_nodes))
        # print("len(clients):", len(clients))
        assert (len(clients) > 0)
        # 求距离
        for fog_id in range(len(fog_nodes)):
            fog_nodes[fog_id].dis_to_clients = get_dis(fog_nodes[fog_id].coords, clients)

            # print(f"{fog_id}:", fog_nodes[fog_id].dis_to_clients)

        # 分组
        client_set = copy.deepcopy(clients)
        # fog_nodes = copy.deepcopy(fog_nodes)

        # 初始化 Pn:
        for fog_id in range(len(fog_nodes)):
            print(fog_nodes[fog_id].dis_to_clients)
            # print("======", min(fog_nodes[fog_id].dis_to_clients))

            min_id = fog_nodes[fog_id].dis_to_clients.index(min(fog_nodes[fog_id].dis_to_clients))
            # print("min_id: ", min_id)

            # while client_set[min_id] == 0:
            #     fog_nodes[fog_id].dis_to_clients[min_id] = float("inf")
            #     min_id = fog_nodes[fog_id].dis_to_clients.index(min(fog_nodes[fog_id].dis_to_clients))

            # 连接fog_node
            fog_nodes[fog_id].clients.append(client_set[min_id])

            # 概率分布赋值
            fog_nodes[fog_id].distribution = client_set[min_id].distribution
            # print(fog_nodes[fog_id].distribution)

            # 去除分组的结点
            # client_set[min_id] = 0
            del client_set[min_id]
            # fog_nodes[fog_id].dis_to_clients[min_id] = float("inf")
            for i in range(len(fog_nodes)):
                del fog_nodes[i].dis_to_clients[min_id]
            print("更改成功！：")
            print(fog_nodes[fog_id].dis_to_clients)

            for i in range(len(client_set)):
                print(f"client_set{fog_id}", client_set[i])

        d_min = []
        while len(client_set) != 0:
            for client_set_ in client_set:
                d_list = []
                J_ne = []
                n = client_set.index(client_set_)
                for f in range(len(fog_nodes)):
                    # Pe+Pn client_set+fog_nodes
                    # P = P_sum(client_set[n].distribution, fog_nodes[f].distribution)
                    P = P_sum(client_set_.distribution, fog_nodes[f].distribution)
                    print("P;", P)
                    # ∆d
                    # d = KL_divergence(P, fog_nodes[f].distribution) - KL_divergence(client_set[n].distribution, fog_nodes[f].distribution)
                    d = KL_divergence(P, fog_nodes[f].distribution)\
                        # - KL_divergence(client_set_.distribution,
                        #                                                             fog_nodes[f].distribution)
                    print("=" * 50)
                    print("d:", d)
                    print("=" * 50)

                    # d1 = KL_divergence(P, fog_nodes[f].distribution)
                    # d2 = KL_divergence(client_set_.distribution, fog_nodes[f].distribution)
                    if math.isnan(d) or d < 0:
                        d = float("inf")
                    print("d:", d)
                    d_list.append(d)
                    # ∆Jne ← κccne + γ*1/|E|*∆d;
                    # J_ne.append(fog_nodes[f].dis_to_clients[n] * fog_nodes[f].k_n2c + self.lambda_ * (1/len(fog_nodes) * d))
                    J_ne.append(
                        fog_nodes[f].dis_to_clients[n] * fog_nodes[f].k_n2c + self.lambda_ * (1 / len(fog_nodes) * d))

                # print("J_ne", J_ne)
                # print("=" * 50)
                d_min.append(min(d_list))
                min_j = J_ne.index(min(J_ne))
                # print("min_j:", min_j)
                fog_nodes[min_j].clients.append(client_set_)

                # print("fog_nodes[J_ne].distribution:", fog_nodes[min_j].distribution)
                fog_nodes[min_j].distribution = P_sum(client_set[n].distribution, fog_nodes[min_j].distribution)
                # print("client_set[n].distribution:", client_set_.distribution)
                client_set.remove(client_set_)

        sum = 0
        for n in fog_nodes:
            print("fog_nodes", n.clients, n.distribution, len(n.clients))
            sum += (len(n.clients))

        print(sum)

        print("="*50)
        print("min_d: ", d_min)
        print("=" * 50)

    def receive_logits(self):
        assert (len(self.fog_nodes) > 0)

        self.uploaded_ids = []
        self.uploaded_models = []
        weight_mean = torch.ones(self.fog_num, self.fog_num, requires_grad=True)
        weight_mean = weight_mean.float() / (self.fog_num)

        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

        for fog in self.fog_nodes:
            self.uploaded_ids.append(fog.id)
            self.uploaded_logits[fog.id] = copy.deepcopy(fog.logit)

        uploaded_logits = np.reshape(np.array(self.uploaded_logits), (self.fog_num, len(self.public_data), -1))
        teacher_logits = torch.zeros(self.fog_num, np.size(uploaded_logits[0], 0),
                                     np.size(uploaded_logits[0], 1),
                                     requires_grad=False)
        models_logits = torch.zeros(self.fog_num, np.size(uploaded_logits[0], 0),
                                    np.size(uploaded_logits[0], 1),
                                    requires_grad=True)
        weight = self.weight_matrix.clone()

        for self_idx in range(self.fog_num):
            teacher_logits_local = teacher_logits[self_idx]
            for teacher_idx in range(self.fog_num):

                teacher_logits_list = uploaded_logits[teacher_idx]

                teacher_logits_local = torch.add(teacher_logits_local,
                                                 weight[self_idx][teacher_idx] * torch.from_numpy(teacher_logits_list))

            loss_input = torch.from_numpy(uploaded_logits[self_idx])

            loss_target = teacher_logits_local

            loss = loss_fn(loss_input, loss_target)

            loss_penalty = loss_fn(weight[self_idx], weight_mean[self_idx])
            loss += loss_penalty * self.penalty_ratio
            weight.retain_grad()
            loss.backward(retain_graph=True)
            print('weight:', weight)

            with torch.no_grad():
                gradabs = torch.abs(weight.grad)
                gradsum = torch.sum(gradabs)
                gradavg = gradsum.item() / (self.fog_num)
                grad_lr = 1.0
                for i in range(5):  # 0.1
                    if gradavg > 0.01:
                        gradavg = gradavg * 1.0 / 5
                        grad_lr = grad_lr / 5
                    if gradavg < 0.01:
                        gradavg = gradavg * 1.0 * 5
                        grad_lr = grad_lr * 5
                weight.sub_(weight.grad * grad_lr)
                weight.grad.zero_()

        # 更新 raw_logits
        for self_idx in range(self.fog_num):
            weight_tmp = torch.zeros(self.fog_num)
            idx_count = 0
            for teacher_idx in range(self.fog_num):
                weight_tmp[idx_count] = weight[self_idx][teacher_idx]
                idx_count += 1
            weight_local = nn.functional.softmax(weight_tmp * 5.0)
            idx_count = 0
            for teacher_idx in range(self.fog_num):
                teacher_logits_list = np.array(uploaded_logits[teacher_idx])
                with torch.no_grad():
                    models_logits[self_idx] = torch.add(models_logits[self_idx],
                                                        weight_local[idx_count] * torch.from_numpy(teacher_logits_list))
                with torch.no_grad():

                    weight[self_idx][teacher_idx] = weight_local[idx_count]
                idx_count += 1
        return models_logits.tolist(), weight

    def evaluate(self):
        # stats=(ids, num_samples, tot_correct, tot_auc)
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
        # self.print_(test_acc, train_acc, train_loss)

        # for x, y in zip(stats[2], stats[1]):
        #     # print("------------------------------")
        #     print("client Accurancy: ", x * 1.0 / y)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []

        for fog in self.fog_nodes:
            for c in fog.clients:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def send_logit(self):
        assert (len(self.fog_nodes) > 0)
        for fog in self.fog_nodes:
            fog.logit = copy.deepcopy(self.uploaded_logits[fog.id])


# clients to fog_node 距离
def get_dis(fog_coords, clients):
    dis = []
    for i in range(len(clients)):
        dis_x = (fog_coords[0] - clients[i].coords[0]) ** 2
        dis_y = (fog_coords[1] - clients[i].coords[1]) ** 2
        res = round(math.sqrt(dis_x + dis_y), 4)
        dis.append(res)
    return dis


def KL_divergence(p, q):
    p_normalized = p
    q_normalized = q
    kl_divergence = 0
    for i in range(len(p)):
        if q_normalized[i] <= 0 or p_normalized[i] <= 0:
            continue
        kl_divergence += p_normalized[i] * math.log(p_normalized[i] / q_normalized[i])
    return kl_divergence


def P_sum(p, q):
    """概率求和"""
    p_sum = []
    for i in range(len(p)):
        l = round((p[i] + q[i]) / 2, 4)
        p_sum.append(l)
    return p_sum
