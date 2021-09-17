import torch
import os
import numpy as np
import copy
from logging_results import cluster_logging

class Cluster:
    def __init__(self, device, args, train_loader, test_loader, model, train_data_samples, cluster_total_train_data_sample, running_time, cluster_id, hyper_param):
        
        self.device = device
        self.cluster_iters = args.cluster_epochs
        self.cluster_total_train_samples = cluster_total_train_data_sample
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = args.client_num
        self.num_clusters = args.cluster_num
        self.beta = args.beta
        self.lamda = args.lamda
        self.rs_train_loss_per, self.rs_HR_per, self.rs_NDCG_per, self.rs_train_loss, self.rs_HR, self.rs_NDCG = [], [], [], [], [], []
        self.running_time = running_time
        self.cluster_id = cluster_id
        self.hyper_param = hyper_param

        self.local_model = copy.deepcopy(list(self.model.parameters()))

        # DP
        self.if_DP = args.if_DP

    
##################### gradient update #####################(暂时不用)
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)
###########################################################

################# weight parameters update #################
    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
    
    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for cluster_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            cluster_param.data = cluster_param.data + user_param.data.clone() * ratio

    # FedAvg aggregate
    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    # pFedMe
    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)             # 聚合在这

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data      # 论文算法1中第10行
############################################################

############## get and set cluster parameters ##############
    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()   # average param update

    def set_grads(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.grad.data = new_param.grad.data.clone()
            local_param.grad.data = new_param.grad.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
############################################################

    # 暂时没有写选择用户
    def select_users(self, round, num_users):
        return self.users



###################### cluster evaluate ######################
    # FedAvg
    def test(self):
        total_HR = []
        total_NDCG = []
        for c in self.users:
            HR, NDCG = c.test()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            # print("testing global client {}  HR: {:.4f}  NDCG: {:.4f}\t".format(c.user_id, HR, NDCG))
        ids = [c.user_id for c in self.users]

        return ids, total_HR, total_NDCG

    # FedAvg
    def train_error_and_loss(self):
        losses = []
        for c in self.users:
            loss = c.train_error_and_loss() 
            losses.append(loss)

        ids = [c.user_id for c in self.users]
        return ids, losses

    # pFedMe
    def test_persionalized_model(self):
        total_HR = []
        total_NDCG = []
        for c in self.users:
            HR, NDCG = c.test_persionalized_model()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            # print("testing persionalized model client {}  HR: {:.4f}  NDCG: {:.4f}\t".format(c.user_id, HR, NDCG))
        ids = [c.user_id for c in self.users]

        return ids, total_HR, total_NDCG

    # pFedMe
    def train_error_and_loss_persionalized_model(self):
        losses = []
        for c in self.users:
            loss = c.train_error_and_loss_persionalized_model() 
            losses.append(loss)

        ids = [c.user_id for c in self.users]
        return ids, losses

    # FedAvg
    def evaluate(self, server_iter, cluster_iter, cluster_id):
        stats = self.test()
        stats_train = self.train_error_and_loss()

        HR = sum(stats[1]) / len(stats[1])
        NDCG = sum(stats[2]) / len(stats[2])
        train_loss = sum(stats_train[1]) / len(stats_train[1])
        self.rs_train_loss.append(train_loss)
        self.rs_HR.append(HR)
        self.rs_NDCG.append(NDCG) 

        # print("Average cluster HR: ", HR)
        # print("Average cluster NDCG: ", NDCG)
        # print("Average cluster Trainning Loss: ",train_loss)
        print("cluster: {}   loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(self.cluster_id, train_loss, HR, NDCG))

        cluster_logging(server_iter, cluster_iter, cluster_id, train_loss, HR, NDCG, self.running_time, self.hyper_param)

        # return HR, NDCG, train_loss

    # pFedMe
    def evaluate_personalized_model(self, server_iter, cluster_iter, cluster_id):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()

        HR = sum(stats[1]) / len(stats[1])
        NDCG = sum(stats[2]) / len(stats[2])
        train_loss = sum(stats_train[1]) / len(stats_train[1])

        self.rs_train_loss_per.append(train_loss)
        self.rs_HR_per.append(HR)
        self.rs_NDCG_per.append(NDCG) 

        # print("Average cluster Personal HR: ", HR)
        # print("Average cluster Personal NDCG: ", NDCG)
        # print("Average cluster Personal Trainning Loss: ",train_loss)
        print("Average cluster: {}   loss:{:.4f}   HR: {:.4f}   NDCG: {:.4f}\t".format(self.cluster_id, train_loss, HR, NDCG))

        cluster_logging(server_iter, cluster_iter, cluster_id, train_loss, HR, NDCG, self.running_time, self.hyper_param)

        # return HR, NDCG, train_loss
#############################################################


###################### server evaluate ######################
    # FedAvg
    def server_test(self):
        total_HR = []
        total_NDCG = []
        for c in self.users:
            HR, NDCG = c.test()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
        ids = [c.user_id for c in self.users]

        return ids, total_HR, total_NDCG

    # pFedMe
    def server_test_persionalized_model(self):
        total_HR = []
        total_NDCG = []
        for c in self.users:
            HR, NDCG = c.test_persionalized_model()
            total_HR.append(HR)
            total_NDCG.append(NDCG)
            # print("testing persionalized model client {}  HR: {:.4f}  NDCG: {:.4f}\t".format(c.user_id, HR, NDCG))
        ids = [c.user_id for c in self.users]

        return ids, total_HR, total_NDCG

    def server_evaluate(self):
        stats = self.server_test_persionalized_model()
        stats_train = self.train_error_and_loss_persionalized_model()       # clients use the personalized model

        HR = sum(stats[1]) / len(stats[1])
        NDCG = sum(stats[2]) / len(stats[2])
        train_loss = sum(stats_train[1]) / len(stats_train[1])

        return HR, NDCG, train_loss
#############################################################
        












