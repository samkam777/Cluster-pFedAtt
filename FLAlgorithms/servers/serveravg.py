import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
import numpy as np

class FedAvg(Server):
    def __init__(self, device, args, train_loader, test_loader, model, times, train_data_samples, total_train_data_sample, running_time):
        super().__init__(device, args, train_loader, test_loader, model, times, train_data_samples, total_train_data_sample, running_time)

        for i in range(args.seg_data):
            train_data = train_loader[i]
            test_data = test_loader[i]
            user = UserAVG(device, args, model, train_data, test_data, train_data_samples[i], i)
            self.users.append(user)
        self.total_train_samples = total_train_data_sample

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

    
    def train(self):

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            # self.send_parameters()

            # Evaluate model each interation
            # self.evaluate()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(glob_iter) #* user.train_samples

            self.evaluate(glob_iter)

            self.aggregate_parameters()

            self.send_parameters()













