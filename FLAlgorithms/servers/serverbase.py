import torch
import os
import numpy as np
import copy
from logging_results import server_logging

class Server:
    def __init__(self, device, args, train_loader, test_loader, model, train_data_samples, cluster_total_train_data_sample, server_total_train_data_sample, running_time, hyper_param):

        self.device = device
        self.server_iters = args.server_epochs
        self.user_train_samples = train_data_samples  
        self.cluster_total_train_samples = cluster_total_train_data_sample
        self.server_total_train_samples = server_total_train_data_sample
        self.model = copy.deepcopy(model)
        self.selected_cluster = []          # 暂时没用到
        self.clusters = []
        self.num_users = args.client_num
        self.num_clusters = args.cluster_num
        self.rs_train_loss_per, self.rs_HR_per, self.rs_NDCG_per, self.rs_train_loss, self.rs_HR, self.rs_NDCG = [], [], [], [], [], []     # 暂时不确定  存放簇的总的指标
        self.running_time = running_time
        self.hyper_param = hyper_param
        # DP
        self.if_DP = args.if_DP

    ##################### gradient update #####################(暂时不用)
    def aggregate_grads(self):
        assert (self.clusters is not None and len(self.clusters) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for cluster in self.clusters:
            self.add_grad(cluster, cluster.cluster_total_train_samples / self.server_total_train_samples)

    def add_grad(self, cluster, ratio):
        cluster_grad = cluster.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + cluster_grad[idx].clone() * ratio

    def send_grads(self):
        assert (self.clusters is not None and len(self.clusters) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for cluster in self.clusters:
            cluster.set_grads(grads)
    ###########################################################
        
    ################# weight parameters update #################
    def send_parameters(self):
        assert (self.clusters is not None and len(self.clusters) > 0)
        for cluster in self.clusters:
            cluster.set_parameters(self.model)

    def add_parameters(self, cluster, ratio):
        model = self.model.parameters()
        for server_param, cluster_param in zip(self.model.parameters(), cluster.get_parameters()):
            server_param.data = server_param.data + cluster_param.data.clone() * ratio

    # average aggregate
    def aggregate_parameters(self):
        assert (self.clusters is not None and len(self.clusters) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for cluster in self.selected_cluster:
            total_train += cluster.cluster_total_train_samples
        for cluster in self.selected_cluster:
            self.add_parameters(cluster, cluster.cluster_total_train_samples / total_train)
    ###########################################################

    def select_cluster(self, round, num_cluster):
        return self.clusters

    ################# evaluate #################
    # def test(self):
    #     total_HR = []
    #     total_NDCG = []
    #     for c in self.clusters:
    #         HR, NDCG = c.test()
    #         total_HR.append(HR)
    #         total_NDCG.append(NDCG)
    #         print("testing global client {}  HR: {:.4f}  NDCG: {:.4f}\t".format(c.id, HR, NDCG))
    #     ids = [c.id for c in self.users]

    #     return ids, total_HR, total_NDCG

    def evaluate(self, server_iter):
        total_HR = []
        total_NDCG = []
        total_loss = []
        for c in self.clusters:
            HR, NDCG, train_loss = c.server_evaluate()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            total_loss.append(train_loss)
        
        avg_HR = sum(total_HR) / len(total_HR)
        avg_NDCG = sum(total_NDCG) / len(total_NDCG)
        avg_loss = sum(total_loss) / len(total_loss)

        # print("Average Server HR: ", avg_HR)
        # print("Average Server NDCG: ", avg_NDCG)
        # print("Average Server Trainning Loss: ", avg_loss)
        print("Average server:    loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(avg_loss, avg_HR, avg_NDCG))

        # server_iter, total_train_loss, total_test_HR, total_test_NDCG, running_time
        server_logging(server_iter, avg_loss, avg_HR, avg_NDCG, self.running_time, self.hyper_param)
    ###########################################################



        
        
        
        






    









