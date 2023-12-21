#!/usr/bin/env python

import sys
import argparse
import os
import time
import warnings
import datetime
import numpy as np
import torchvision
from flcore.servers.server_KTpFL import Server_KTpFL
from flcore.trainmodel import model_KTpFL
from flcore.trainmodel.models import *
from flcore.trainmodel.bilstm import BiLSTM_TextClassification
from utils.logger import Logger
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32


def run(args):
    log_path = f"./logs/log_{args.algorithm}_{args.dataset}_{args.global_rounds}_{datetime.date.today()}.log"
    sys.stdout = Logger(log_path, sys.stdout)
    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        if type(args.model) == type(''):
            model_str = args.model

        # Generate args.model
        if model_str == "mlr":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":
            if args.dataset == "mnist" or args.dataset == "fmnist" or args.dataset == "Emnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif args.dataset[:13] == "Tiny-imagenet" or args.dataset[:8] == "Imagenet":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)


        elif model_str == "dnn":  # non-convex
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=hidden_dim,
                                                   output_size=args.num_classes,
                                                   num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                                   attention_dropout=0,
                                                   embedding_length=hidden_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=hidden_dim, max_len=max_len, vocab_size=vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "mix":
            # if args.dataset == "mnist" or args.dataset == "fmnist":
            CANDIDATE_MODELS = [
                # model_KTpFL.DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device),
                model_KTpFL.cnn_2layer_fc_model(args.num_classes),
                model_KTpFL.AlexNet(num_classes=args.num_classes),
                model_KTpFL.LeNet(),
                model_KTpFL.shufflenetv2(num_classes=args.num_classes),
                model_KTpFL.mobilenetv2(num_classes=args.num_classes),
                model_KTpFL.ResNet18(num_classes=args.num_classes),
                model_KTpFL.cnn_3layer_fc_model(args.num_classes)]
            args.model = CANDIDATE_MODELS

    server = Server_KTpFL(args, i)

    server.train()

    time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset,
                 algorithm=args.algorithm,
                 goal=args.goal,
                 times=args.times,
                 length=args.global_rounds / args.eval_gap + 1)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-ls', "--local_steps", type=int, default=20)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation ")  # 评估差距

    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument('-fog_num', "--fog_num", type=int, default=10,
                        help="Number of fog nodes")
    parser.add_argument('-knf', "--knf", type=int, default=5,
                        help="Aggregation times of fog node aggregator")
    parser.add_argument('-kfa', "--kfa", type=int, default=50,
                        help="Aggregation times of the aggregator")
    parser.add_argument('-lambda', "--lambda_", type=float, default=0.5,
                        help="The tradeoff between distance and KL divergence")
    parser.add_argument('-Temp', "--Temp", type=float, default=10.0,
                        help="temperature")
    parser.add_argument('-lmr', "--N_logits_matching_round", type=int, default=1,
                        help="Knowledge number of distillation rounds")
    parser.add_argument('-flr', "--fog_learning_rate", type=float, default=0.0001,
                        help="Learning rate of fog node knowledge distillation")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    if args.algorithm == "pFedMe":
        print("Average moving parameter beta: {}".format(args.beta))
        print("Regularization rate: {}".format(args.lamda))
        print("Number of personalized training steps: {}".format(args.K))
        print("personalized learning rate to caculate theta: {}".format(args.p_learning_rate))
    elif args.algorithm == "PerAvg":
        print("Second learning rate beta: {}".format(args.beta))
    elif args.algorithm == "FedProx":
        print("Proximal rate: {}".format(args.mu))
    elif args.algorithm == "FedFomo":
        print("Server sends {} models to one client at each round".format(args.M))
    elif args.algorithm == "FedMTL":
        print("The iterations for solving quadratic subproblems: {}".format(args.itk))
    elif args.algorithm == "FedAMP":
        print("alphaK: {}".format(args.alphaK))
        print("lamda: {}".format(args.lamda))
        print("sigma: {}".format(args.sigma))
    elif args.algorithm == "APFL":
        print("alpha: {}".format(args.alpha))
    elif args.algorithm == "Ditto":
        print("plocal_steps: {}".format(args.plocal_steps))
        print("mu: {}".format(args.mu))
    elif args.algorithm == "FedRep":
        print("plocal_steps: {}".format(args.plocal_steps))
    elif args.algorithm == "FedPHP":
        print("mu: {}".format(args.mu))
        print("lamda: {}".format(args.lamda))
    print("=" * 50)

run(args)
