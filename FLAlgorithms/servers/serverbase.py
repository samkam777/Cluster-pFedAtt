import torch
import os
import numpy as np
import copy
from logging_results import server_logging
import torch.nn.functional as F

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
        self.beta = args.beta
        self.lamda = args.lamda
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

    def get_att_weight(self, cluster):
        att_weight = torch.tensor(0.).to(self.device)
        for server_param, cluster_param in zip(self.model.parameters(), cluster.get_parameters()):
            att_weight += torch.norm(server_param-cluster_param, p=2)
        return att_weight
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
    
    # attention pFed
    def attention_persionalized_aggregate_parameters(self):
        assert (self.clusters is not None and len(self.clusters) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        att_w = []
        for cluster in self.selected_cluster:
            att_weight = self.get_att_weight(cluster)
            att_w.append(att_weight)
        att_w_ = torch.Tensor(att_w)
        print("att_w: {}\t".format(att_w_))
        min_att_w_ = torch.min(att_w_)
        max_att_w_ = torch.max(att_w_)
        # norm_att_w_ = 1 - ((att_w_ - min_att_w_) / (max_att_w_ - min_att_w_))
        norm_att_w_ = (att_w_ - min_att_w_) / (max_att_w_ - min_att_w_)
        norm_att_w_ = F.softmax(norm_att_w_, dim=0)   # 行和
        print("att_w after softmax: {}\t".format(norm_att_w_))

        for i, cluster in enumerate(self.selected_cluster):
            self.add_parameters(cluster, norm_att_w_[i])  

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data 
    ###########################################################

    def select_cluster(self, round, num_cluster):
        return self.clusters

    ################# evaluate #################
    def evaluate_personalized_model(self, server_iter, total_loss):
        total_HR = []
        total_NDCG = []
        # total_loss = []
        for c in self.clusters:
            HR, NDCG = c.server_evaluate_persionalized_model()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            # total_loss.append(train_loss)
        
        avg_HR = sum(total_HR) / len(total_HR)
        avg_NDCG = sum(total_NDCG) / len(total_NDCG)
        avg_loss = sum(total_loss) / len(total_loss)

        print("Personal server:    loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(avg_loss, avg_HR, avg_NDCG))

        server_logging(server_iter, avg_loss, avg_HR, avg_NDCG, self.running_time, self.hyper_param)

    def evaluate(self, server_iter, total_loss):
        total_HR = []
        total_NDCG = []
        # total_loss = []
        for c in self.clusters:
            HR, NDCG = c.server_evaluate()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            # total_loss.append(train_loss)
        
        avg_HR = sum(total_HR) / len(total_HR)
        avg_NDCG = sum(total_NDCG) / len(total_NDCG)
        avg_loss = sum(total_loss) / len(total_loss)

        print("Average server:    loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(avg_loss, avg_HR, avg_NDCG))

        server_logging(server_iter, avg_loss, avg_HR, avg_NDCG, self.running_time, self.hyper_param)
    ###########################################################



        
        
        
        






    









