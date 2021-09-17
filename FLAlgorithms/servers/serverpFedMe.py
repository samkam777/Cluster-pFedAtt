import torch
import os

from FLAlgorithms.cluster.clusterpFedMe import ClusterpFedMe
from FLAlgorithms.servers.serverbase import Server
import numpy as np

from logging_results import eps_logging

class pFedMe(Server):
    def __init__(self, device, args, train_loader, test_loader, model, train_data_samples, cluster_total_train_data_sample, server_total_train_data_sample, running_time, hyper_param):
        super().__init__(device, args, train_loader, test_loader, model, train_data_samples, cluster_total_train_data_sample, server_total_train_data_sample, running_time, hyper_param)

        for i in range(self.num_clusters):
            train_data = train_loader[i]
            test_data = test_loader[i]
            cluster = ClusterpFedMe(device, args, train_data, test_data, model, train_data_samples[i], cluster_total_train_data_sample[i], running_time, i, hyper_param)
            self.clusters.append(cluster)


# 还没完成全局的聚合更新部分
    def train(self):
        losses = []
        for server_iter in range(self.server_iters):
            print("")
            print("---------------------------server iter: ",server_iter, " ---------------------------")

            epsilons_list = []
            # do update for all clusters not only selected clusters
            for cluster in self.clusters:
                if self.if_DP:
                    epsilons = cluster.train(server_iter)
                    epsilons_list.append(epsilons)
                else:
                    cluster.train(server_iter)

            # choose several clusters to send back upated model to server
            self.selected_cluster = self.select_cluster(server_iter, self.num_clusters)

            # Evaluate personalized model on clusters for each interation
            print("")
            print("Evaluate server average model")

            self.evaluate(server_iter)

            self.aggregate_parameters()

            self.send_parameters()

            # if self.if_DP:
            #     eps = sum(epsilons_list) / len(epsilons_list)
            #     eps_logging(server_iter, eps, self.running_time)




