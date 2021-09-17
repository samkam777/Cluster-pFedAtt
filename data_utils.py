import random
import numpy as np 
import pandas as pd 
import torch
import argparse

from torch.utils.data.sampler import RandomSampler

class NCF_Data(object):
	"""
	Construct Dataset for NCF
	"""
	def __init__(self, args, ratings):
		random.seed(args.seed)
		self.ratings = ratings
		self.num_ng = args.num_ng
		self.num_ng_test = args.num_ng_test
		self.batch_size = args.batch_size

		self.seg_data_len = args.seg_data

		self.preprocess_ratings = self._reindex(self.ratings)

		self.user_pool = set(self.ratings['user_id'].unique())
		self.item_pool = set(self.ratings['item_id'].unique())

		self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
		self.negatives = self._negative_sampling(self.preprocess_ratings)
		self.if_rep_sample = args.rep_sample
		
		# cluster
		self.cluster_num = args.cluster_num
		self.client_num = args.client_num

		# 用于制造不平衡数据
		self.random_list = self.shuffle_list(len(self.user_pool), self.cluster_num * self.client_num, 100)
	
	def _reindex(self, ratings):
		"""
		Process dataset to reindex userID and itemID, also set rating as binary feedback
		"""
		user_list = list(ratings['user_id'].drop_duplicates())
		user2id = {w: i for i, w in enumerate(user_list)}

		item_list = list(ratings['item_id'].drop_duplicates())
		item2id = {w: i for i, w in enumerate(item_list)}

		ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])		
		ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
		ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
		return ratings

	def _leave_one_out(self, ratings):
		"""
		leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
		"""
		ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
		test = ratings.loc[ratings['rank_latest'] == 1]	
		train = ratings.loc[ratings['rank_latest'] > 1]
		assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
		return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

	def _negative_sampling(self, ratings):
		interact_status = (
			ratings.groupby('user_id')['item_id']
			.apply(set)
			.reset_index()
			.rename(columns={'item_id': 'interacted_items'}))
		interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)		
		interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_ng_test))		
		return interact_status[['user_id', 'negative_items', 'negative_samples']]
	

################################################# balance #################################################
	def cluster_get_train_instance(self):
		train_datas = [[[], [], []] for i in range(self.cluster_num*self.client_num)]
		
		train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'negative_items']], on='user_id')	
		train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))	
		for row in train_ratings.itertuples():
			seg_data = int(int(row.user_id)/(len(self.user_pool)/(self.client_num*self.cluster_num)))
			train_datas[seg_data][0].append(int(row.user_id))	# users
			train_datas[seg_data][1].append(int(row.item_id))	# items
			train_datas[seg_data][2].append(int(row.rating))	# ratings
			for i in range(self.num_ng):
				train_datas[seg_data][0].append(int(row.user_id))
				train_datas[seg_data][1].append(int(row.negatives[i]))
				train_datas[seg_data][2].append(float(0))

		# 数据分簇分用户
		cluster_train_datas = [[[] for j in range(self.client_num)] for i in range(self.cluster_num)]
		for i in range(self.cluster_num):
			for j in range(self.client_num):
				cluster_train_datas[i][j] = train_datas[i*self.client_num + j]
				#print(i*self.client_num + j)

		train_datas_loader = [[] for i in range(self.cluster_num)]
		# print(len(train_datas_loader))
		train_data_samples = [[] for i in range(self.cluster_num)]
		# print(len(train_data_samples))
		cluster_data_sample_list = []
		total_train_data_sample = 0

		for i in range(self.cluster_num):
			cluster_data_sample = 0
			for j in range(self.client_num):
				dataset = Rating_Datset(
					user_list=cluster_train_datas[i][j][0],
					item_list=cluster_train_datas[i][j][1],
					rating_list=cluster_train_datas[i][j][2])
				if self.if_rep_sample:
					ran_sampler = RandomSampler(dataset, replacement=True, num_samples=2*len(dataset))			# 采样的数量为2倍的数据集大小
					dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=ran_sampler, drop_last=True)	# 
				else:
					dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
				train_datas_loader[i].append(dataloader)
				train_data_samples[i].append(len(cluster_train_datas[i][j][1]))	# 当前客户端的用户数量
				cluster_data_sample += len(cluster_train_datas[i][j][1])		# 累计簇端的总的用户数量
				
				total_train_data_sample += len(cluster_train_datas[i][j][1])	# 累计总的客户端的用户数量
			#print("cluster_data_sample:{}\t".format(cluster_data_sample))
			cluster_data_sample_list.append(cluster_data_sample)
			#print("cluster_data_sample_list:{}\t".format(cluster_data_sample_list[i]))

		for i in range(self.cluster_num):
			for j in range(self.client_num):
				print("client {} data samples / cluster {} data samples / total data samples: {} / {} / {} ".format(j, i, len(cluster_train_datas[i][j][1]), cluster_data_sample_list[i], total_train_data_sample))

		return train_datas_loader, train_data_samples, total_train_data_sample, cluster_data_sample_list

	def cluster_get_test_instance(self):
		test_datas = [[[], [], []] for i in range(self.client_num*self.cluster_num)]
		test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')
		for row in test_ratings.itertuples():
			seg_data = int(int(row.user_id)/(len(self.user_pool)/(self.client_num*self.cluster_num)))
			test_datas[seg_data][0].append(int(row.user_id))
			test_datas[seg_data][1].append(int(row.item_id))
			test_datas[seg_data][2].append(float(row.rating))
			for i in getattr(row, 'negative_samples'):
				test_datas[seg_data][0].append(int(row.user_id))
				test_datas[seg_data][1].append(int(i))
				test_datas[seg_data][2].append(float(0))

		cluster_test_datas = [[[] for j in range(self.client_num)] for i in range(self.cluster_num)]
		for i in range(self.cluster_num):
			for j in range(self.client_num):
				cluster_test_datas[i][j] = test_datas[i*self.client_num + j]
		
		test_datas_loader = [[] for i in range(self.cluster_num)]

		for i in range(self.cluster_num):
			for j in range(self.client_num):
				dataset = Rating_Datset(
					user_list=cluster_test_datas[i][j][0],
					item_list=cluster_test_datas[i][j][1],
					rating_list=cluster_test_datas[i][j][2])
				dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False)
				test_datas_loader[i].append(dataloader)

		return test_datas_loader
###########################################################################################################


################################################ unbalance ################################################
	def cluster_get_train_instance_unbalance(self):
		train_datas = [[[], [], []] for i in range(self.cluster_num*self.client_num)]
		train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'negative_items']], on='user_id')	# 将negative item的位置补进去
		train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))	# 添加训练的negative item
		#random_list = self.shuffle_list(len(self.user_pool), self.cluster_num*self.client_num, 100)
		index_of_random_list = 0
		for row in train_ratings.itertuples():
			if_0 = int(int(row.user_id)/self.random_list[index_of_random_list])
			if if_0 == 0:
				seg_data = index_of_random_list
				#seg_data = int(int(row.user_id)/(len(self.user_pool)/(self.cluster_num*self.client_num)))
				train_datas[seg_data][0].append(int(row.user_id))	# users
				train_datas[seg_data][1].append(int(row.item_id))	# items
				train_datas[seg_data][2].append(int(row.rating))	# ratings
				for i in range(self.num_ng):
					train_datas[seg_data][0].append(int(row.user_id))
					train_datas[seg_data][1].append(int(row.negatives[i]))
					train_datas[seg_data][2].append(float(0))
			elif if_0 > 0:
				index_of_random_list += 1

		# 数据分簇分用户
		cluster_train_datas = [[[] for j in range(self.client_num)] for i in range(self.cluster_num)]
		for i in range(self.cluster_num):
			for j in range(self.client_num):
				cluster_train_datas[i][j] = train_datas[i*self.client_num + j]

		train_datas_loader = [[] for i in range(self.cluster_num)]
		train_data_samples = [[] for i in range(self.cluster_num)]
		cluster_data_sample_list = []
		total_train_data_sample = 0

		for i in range(self.cluster_num):
			cluster_data_sample = 0
			for j in range(self.client_num):
				dataset = Rating_Datset(
					user_list=cluster_train_datas[i][j][0],
					item_list=cluster_train_datas[i][j][1],
					rating_list=cluster_train_datas[i][j][2])
				if self.if_rep_sample:
					ran_sampler = RandomSampler(dataset, replacement=True, num_samples=2*len(dataset))			# 采样的数量为2倍的数据集大小
					dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=ran_sampler, drop_last=True)	# 
				else:
					dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
				train_datas_loader[i].append(dataloader)
				train_data_samples[i].append(len(cluster_train_datas[i][j][1]))	# 当前客户端的用户数量
				cluster_data_sample += len(cluster_train_datas[i][j][1])		# 累计簇端的总的用户数量
				total_train_data_sample += len(cluster_train_datas[i][j][1])	# 累计总的客户端的用户数量
			cluster_data_sample_list.append(cluster_data_sample)

		for i in range(self.cluster_num):
			for j in range(self.client_num):
				print("client {} data samples / cluster {} data samples / total data samples: {} / {} / {} ".format(j, i, len(cluster_train_datas[i][j][1]), cluster_data_sample_list[i], total_train_data_sample))
		
		return train_datas_loader, train_data_samples, total_train_data_sample, cluster_data_sample_list

	def cluster_get_test_instance_unbalance(self):
		test_datas = [[[], [], []] for i in range(self.cluster_num*self.client_num)]
		test_datas_loader = []
		test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')
		index_of_random_list = 0
		for row in test_ratings.itertuples():
			if_0 = int(int(row.user_id)/self.random_list[index_of_random_list])
			if if_0 == 0:
				seg_data = index_of_random_list
				#seg_data = int(int(row.user_id)/(len(self.user_pool)/self.cluster_num*self.client_num))
				test_datas[seg_data][0].append(int(row.user_id))
				test_datas[seg_data][1].append(int(row.item_id))
				test_datas[seg_data][2].append(float(row.rating))
				for i in getattr(row, 'negative_samples'):
					test_datas[seg_data][0].append(int(row.user_id))
					test_datas[seg_data][1].append(int(i))
					test_datas[seg_data][2].append(float(0))
			elif if_0 > 0:
				index_of_random_list += 1

		cluster_test_datas = [[[] for j in range(self.client_num)] for i in range(self.cluster_num)]
		for i in range(self.cluster_num):
			for j in range(self.client_num):
				cluster_test_datas[i][j] = test_datas[i*self.client_num + j]
		
		test_datas_loader = [[] for i in range(self.cluster_num)]

		for i in range(self.cluster_num):
			for j in range(self.client_num):
				dataset = Rating_Datset(
					user_list=cluster_test_datas[i][j][0],
					item_list=cluster_test_datas[i][j][1],
					rating_list=cluster_test_datas[i][j][2])
				dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False)
				test_datas_loader[i].append(dataloader)

		return test_datas_loader		
###########################################################################################################


	# amount 是总的数量，num 是要切分的份数。
	def shuffle_list(self, amount, num, min_amount):
		'''
		得到类似这样的list
		list2 = [304, 708, 1341, 2985, 702]
		list3 = [304, 1012, 2353, 5338, 6040]
		'''
		list1 = []
		for i in range(0,num-1):
			a = random.randint(min_amount,amount)    # 生成 n-1 个随机节点
			list1.append(a)
		list1.sort()                        # 节点排序
		list1.append(amount)                # 设置第 n 个节点为amount

		list2 = []
		for i in range(len(list1)):
			if i == 0:
				b = list1[i]                # 第一段长度为第 1 个节点 - 0
			else:
				b = list1[i] - list1[i-1]   # 其余段为第 n 个节点 - 第 n-1 个节点
			list2.append(b)

		list3 = []
		number = 0
		for i in range(num):
			number += list2[i]
			list3.append(number)

		return list3
	

class Rating_Datset(torch.utils.data.Dataset):
	def __init__(self, user_list, item_list, rating_list):
		super(Rating_Datset, self).__init__()
		self.user_list = user_list
		self.item_list = item_list
		self.rating_list = rating_list

	def __len__(self):
		return len(self.user_list)

	def __getitem__(self, idx):
		user = self.user_list[idx]
		item = self.item_list[idx]
		rating = self.rating_list[idx]
		
		return (
			torch.tensor(user, dtype=torch.long),
			torch.tensor(item, dtype=torch.long),
			torch.tensor(rating, dtype=torch.float)
			)


if __name__ == '__main__':
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
	parser.add_argument("--epochs", 
		type=int,
		default=1000,  
		help="training epoches")
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
	parser.add_argument("--out", 
		default=True,
		help="save model or not")
	parser.add_argument("--seg_data", 
		type=int,
		default=5, 
		help="Number of segmentation of data")
	parser.add_argument("--times",
		type=int,
		default=1,
		help="running time")
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
		help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
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
		default=False,
		help="if DP")
	# sample data 重复抽样数据
	parser.add_argument("--rep_sample", 
		default=False,
		help="if repeated sampling data")
	# balance or unbalance data
	parser.add_argument("--_balance", 
		default=True,
		help="balance or unbalance data")
	# subsample data 二次采样数据
	parser.add_argument("--_subsample", 
		default=False,
		help="if subsample data")
	# cluster param
	parser.add_argument("--cluster_num",
		type=int,
		default=2,
		help="cluster number")
	parser.add_argument("--client_num",
		type=int,
		default=5,
		help="client number")

	# set device and parameters
	args = parser.parse_args()

	DATA_PATH = r'./data/ml-1m/ratings.dat'
	ml_1m = pd.read_csv(
		DATA_PATH, 
		sep="::", 
		names = ['user_id', 'item_id', 'rating', 'timestamp'], 
		engine='python')

	data = NCF_Data(args, ml_1m)
	data.cluster_get_train_instance()
	data.cluster_get_test_instance()
	data.cluster_get_train_instance_unbalance()
	data.cluster_get_test_instance_unbalance()

'''
	train_loader, _, _ =data.get_train_instance()
	#test_loader = data.get_test_instance()

	print("len of train_loader:{}\t".format(len(train_loader[0])))
	#print("len of train_loader:{}\t".format(len(train_loader)))


	#for idx, (user, item, label) in enumerate(train_loader[0]):
	for user, item, label in train_loader[0]:
		print("len of user:{}\t".format(len(user)))	# 就是batchsize
		indices = np.random.permutation(len(user))
		#print("indices:{}\t".format(indices))			# batchsize长度的一个list，随机数
		rnd_sampled = np.random.binomial(len(user), 0.34)
		#print("rnd_sampled:{}\t".format(rnd_sampled))	# 二项分布的数
		# print("user before:{}\t".format(user))
		#user1 = user[indices]							# 按照indices顺序重新排序
		#print("user after:{}\t".format(user))
		#user2 = user1[:rnd_sampled]						# 抽一部分数据
		#print("user after after:{}\t".format(user2))
		user3 = user[indices][:rnd_sampled]
		# print("user after after after:{}\t".format(user3))	# 按照indices顺序重新排序,抽一部分数据

	# print(train_loader)
'''






