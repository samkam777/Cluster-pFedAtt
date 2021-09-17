import os
import time
import argparse
import copy
import math
import pandas as pd
import numpy as np
import torch

from FLAlgorithms.trainmodel.model import *
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from util import str2bool
import util
import data_utils

from m_opacus.dp_model_inspector import DPModelInspector
from m_opacus.utils import module_modification
from m_opacus import PrivacyEngine


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", 
		type=int, 
		default=42, 
		help="Seed")
	parser.add_argument("--lr", 
		type=float, 
		default=0.75, 
		help="learning rate")
	parser.add_argument("--dropout", 
		type=float,
		default=0.2,  
		help="dropout rate")
	parser.add_argument("--batch_size", 
		type=int, 
		default=512, 
		help="batch size for training")
	parser.add_argument("--server_epochs", 
		type=int,
		default=60,  
		help="server training epoches")
	parser.add_argument("--cluster_epochs", 
		type=int,
		default=1,  
		help="cluster training epochs")
	parser.add_argument("--top_k", 
		type=int, 
		default=10, 
		help="compute metrics@top_k")
	parser.add_argument("--factor_num", 
		type=int,
		default=32, 
		help="predictive factors numbers in the model")
	parser.add_argument("--layers",
		nargs='+', 
		default=[64,32,16,8],
		help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
	parser.add_argument("--num_ng", 
		type=int,
		default=4, 
		help="Number of negative samples for training set")
	parser.add_argument("--num_ng_test", 
		type=int,
		default=100, 
		help="Number of negative samples for test set")
	parser.add_argument("--seg_data", 
		type=int,
		default=5, 
		help="Number of segmentation of data")
	parser.add_argument("--times",
		type=int,
		default=1,
		help="running times")
	parser.add_argument("--local_epochs",
		type=int,
		default=1)
	parser.add_argument("--algorithm",
		type=str,
		default="pFedMe",
		choices=["pFedMe", "FedAvg"]) 
	# personal setting
	parser.add_argument("--personal_learning_rate",
		type=float,
		default=0.01,
		help="Persionalized learning rate to caculate theta aproximately using K steps")
	parser.add_argument("--lamda",
		type=int,
		default=1,
		help="Regularization term")
	parser.add_argument("--K",
		type=int,
		default=1,
		help="Computation steps")
	parser.add_argument("--beta",
		type=float,
		default=1.0,
		help="Average moving parameter for pFedMe")
	# DP
	parser.add_argument('--delta',
		type=float,
		default=1e-4,
		help='DP DELTA')
	parser.add_argument('--max_grad_norm',
		type=float,
		default= 1.0,
		help='DP MAX_GRAD_NORM')
	parser.add_argument('--noise_multiplier',
		type=float,
		default= 1.0,
		help='DP NOISE_MULTIPLIER')
	parser.add_argument('--virtual_batch_size',
		type=int,
		default=512000, 
        help='DP VIRTUAL_BATCH_SIZE')
	parser.add_argument("--if_DP", 
		type=str2bool,
		default=False,
		help="if DP")
	# repeated sampling data
	parser.add_argument("--rep_sample", 
		type=str2bool,
		default=False,
		help="whether repeated sampling data")
	# balance or unbalance data
	parser.add_argument("--_balance", 
		type=str2bool,
		default=True,
		help="balance or unbalance data")
	# subsample data
	parser.add_argument("--_subsample", 
		type=str2bool,
		default=False,
		help="whether subsample data")
	# cluster param
	parser.add_argument("--cluster_num",
		type=int,
		default=2,
		help="cluster number")
	parser.add_argument("--client_num",
		type=int,
		default=5,
		help="client number")
	# get time
	parser.add_argument("--_running_time", 
        type=str, 
        default="2021-00-00-00-00-00", 
        help="running time")

	# set device and parameters
	args = parser.parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# seed for Reproducibility
	util.seed_everything(args.seed)

	# hyper-parameter
	hyper_param = "_balance_" + str(args._balance) + "_cluster_" + str(args.cluster_num) + "_user_" + str(args.client_num) + "_ServerEpochs_" + str(args.server_epochs) + "_ClusterEpochs_" + str(args.cluster_epochs) + "_"
	print("hyper_param: {}\t".format(hyper_param))

	for i in range(args.times):
		print("---------------Running time: {} ------------".format(i))
		# running time
		if args._running_time != "2021-00-00-00-00-00":
			running_time = args._running_time			# get time from *.sh file 
		else:
			running_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())		# get the present time
		print("running time: {}".format(running_time))

		# load client train data and test data
		ml_1m = pd.read_csv(
			r'./data/ml-1m/ratings.dat', 
			sep="::", 
			names = ['user_id', 'item_id', 'rating', 'timestamp'], 
			engine='python')
		# set the num_users, items
		num_users = ml_1m['user_id'].nunique()
		num_items = ml_1m['item_id'].nunique()
		# print("num_items: {} ".format(num_items))	# 3706
		# print("num_users: {} ".format(num_users))	# 6040

		# construct the train and test datasets
		data = data_utils.NCF_Data(args, ml_1m)
		if args._balance:
			print("balance data!!")
			train_loader, train_data_samples, total_train_data_sample, cluster_data_sample = data.cluster_get_train_instance()
			test_loader = data.cluster_get_test_instance()
		else:
			print("unbalance data!!")
			train_loader, train_data_samples, total_train_data_sample, cluster_data_sample = data.cluster_get_train_instance_unbalance()
			test_loader = data.cluster_get_test_instance_unbalance()

		# model
		model = NeuMF(args, num_users, num_items)
		model.to(device)

		if(args.algorithm == "pFedMe"):
			server = pFedMe(device, args, train_loader, test_loader, model, train_data_samples, cluster_data_sample, total_train_data_sample, running_time, hyper_param)

		if(args.algorithm == "FedAvg"):
			server = FedAvg(device, args, train_loader, test_loader, model, train_data_samples, cluster_data_sample, total_train_data_sample, running_time, hyper_param)

		server.train()
		server.test()


if __name__ == '__main__':
	main()





