import torch
import os
import numpy as np

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.cluster.clusterbase import Cluster

from logging_results import eps_logging

class ClusterpFedMe(Cluster):
    def __init__(self, device, args, train_loader, test_loader, model, train_data_samples, cluster_total_train_data_sample, running_time, cluster_id, hyper_param):
        super().__init__(device, args, train_loader, test_loader, model, train_data_samples, cluster_total_train_data_sample, running_time, cluster_id, hyper_param)

        for i in range(self.num_users):
            train_data = train_loader[i]
            test_data = test_loader[i]
            user = UserpFedMe(device, args, model, train_data, test_data, train_data_samples[i], i, hyper_param, running_time)
            self.users.append(user) 

    def train(self, server_iter):
        losses = []
        for cluster_iter in range(self.cluster_iters):
            print("")
            print("-------------cluster {} iter: {} -------------".format(self.cluster_id, cluster_iter))

            epsilons_list = []
            # do update for all users not only selected users
            for user in self.users:
                if self.if_DP:
                    _, epsilons = user.train(server_iter, cluster_iter, self.cluster_id) # user.train_samples
                    epsilons_list.append(epsilons)
                else:
                    _ = user.train(server_iter, cluster_iter, self.cluster_id) # user.train_samples
            
            # choose several users to send back upated model to server
            self.selected_users = self.select_users(cluster_iter, self.num_users)

            # Evaluate personalized model on user for each interation
            print("")
            print("Evaluate personalized model")            
            self.evaluate_personalized_model(server_iter, cluster_iter, self.cluster_id)         # 验证的是更新模型前的       # logging部分已修改
            
            self.persionalized_aggregate_parameters()

            self.send_parameters()

            if self.if_DP:
                eps = sum(epsilons_list) / len(epsilons_list)
                eps_logging(cluster_iter, self.cluster_id, eps, self.running_time)                               # logging部分已修改
        if self.if_DP:
            return eps        
        
        






