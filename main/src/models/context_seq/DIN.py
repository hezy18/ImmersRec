# -*- coding: UTF-8 -*-

""" DIN
Reference:
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseModel import ContextSeqModel

class DIN(ContextSeqModel):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer.")
		return ContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.include_id = 1
		# self.user_feature_dim = sum([corpus.feature_max[f] for f in corpus.user_feature_names+['user_id']])
		# self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names+['item_id']])
		self.user_feature_dim = corpus.feature_max_categorical['user_id']
		self.item_feature_dim = corpus.feature_max_categorical['item_id']
		self.item_feature_num = len(corpus.item_feature_names) + 1  # 4
		self.user_feature_num = len(corpus.user_feature_names) + 1 # 1
		#TODO：并没有context feature(非item feature、user feature)是category的问题
		# self.context_feature_dim = sum(corpus.feature_max_numeric[f] for f in corpus.context_feature_names) 
		self.include_context_features = corpus.include_context_features
		self.include_immersion = corpus.include_immersion
		#TODO: 以下部分写死了，之后要改
		if self.include_context_features:
			if self.include_immersion:
				self.context_feature_num = 22
			else:
				self.context_feature_num = 21
		else:
			self.context_feature_num = 0

		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)

		self._define_params()
		self.apply(self.init_weights)
	
	def _define_params(self):
		self.user_embedding = nn.Embedding(self.user_feature_dim, self.vec_size)
		self.item_embedding = nn.Embedding(self.item_feature_dim, self.vec_size)
		# self.context_embedding = nn.Embedding(self.context_feature_dim, self.vec_size)
		self.user_linear = nn.Linear(1,self.vec_size)
		self.item_linear = nn.Linear(1,self.vec_size)
		self.context_linear = nn.Linear(1,self.vec_size)
		self.user_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.user_feature_num-1)]) # 这个-1也是写死的
		self.item_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.item_feature_num-1)])
		self.context_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.context_feature_num)])

		self.att_mlp_layers = torch.nn.ModuleList()
		pre_size = 4 * self.item_feature_num * self.vec_size 
		for size in self.att_layers:
			self.att_mlp_layers.append(torch.nn.Linear(pre_size, size))
			self.att_mlp_layers.append(torch.nn.Sigmoid())
			self.att_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dense = nn.Linear(pre_size, 1)

		self.dnn_mlp_layers = torch.nn.ModuleList()
		pre_size = 3 * self.item_feature_num * self.vec_size + self.user_feature_num * self.vec_size + self.context_feature_num * self.vec_size
		
		for size in self.dnn_layers:
			self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, size))
			self.dnn_mlp_layers.append(torch.nn.BatchNorm1d(num_features=size))
			self.dnn_mlp_layers.append(Dice(size))
			self.dnn_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, 1))

		if self.include_immersion:
			self.im_linear1 = nn.Linear(10, 4)
			self.im_linear2 = nn.Linear(10, 4)
			self.im_item_embedding = nn.Linear(3, 1)
			self.im_params = nn.Parameter(torch.randn(3))

			self.fc1 = nn.Linear(9, 64)
			self.fc2 = nn.Linear(64, 1)

	def attention(self, queries, keys, keys_length, softmax_stag=False, return_seq_weight=False):
		'''Reference:
			RecBole layers: SequenceAttLayer
			queries: batch * (if*vecsize)
		'''
		embedding_size = queries.shape[-1]  # H
		hist_len = keys.shape[1]  # T
		queries = queries.repeat(1, hist_len)
		queries = queries.view(-1, hist_len, embedding_size)
		# MLP Layer
		input_tensor = torch.cat(
			[queries, keys, queries - keys, queries * keys], dim=-1
		)
		output = input_tensor
		for layer in self.att_mlp_layers:
			output = layer(output)
		output = torch.transpose(self.dense(output), -1, -2)
		# get mask
		output = output.squeeze(1)
		mask = self.mask_mat.repeat(output.size(0), 1)
		mask = mask >= keys_length.unsqueeze(1)
		# mask
		if softmax_stag:
			mask_value = -np.inf
		else:
			mask_value = 0.0
		output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
		output = output.unsqueeze(1)
		output = output / (embedding_size**0.5)
		# get the weight of each user's history list about the target item
		if softmax_stag:
			output = fn.softmax(output, dim=2)  # [B, 1, T]
		if not return_seq_weight:
			output = torch.matmul(output, keys)  # [B, 1, H]
		torch.cuda.empty_cache()
		return output.squeeze()

	def attention_and_dnn(self, item_feats_emb, history_feats_emb, hislens, user_feats_emb, context_feats_numeric):
		batch_size, item_num, feats_emb = item_feats_emb.shape
		_, max_len, his_emb = history_feats_emb.shape

		item_feats_emb2d = item_feats_emb.view(-1, feats_emb) # B*num_item, item_feature*vec_size
		history_feats_emb2d = history_feats_emb.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,his_emb) # B*num_item, 1, item_feature*vec_size
		hislens2d = hislens.unsqueeze(1).repeat(1,item_num).view(-1)
		user_feats_emb2d = user_feats_emb.repeat(1,item_num,1).view(-1, user_feats_emb.shape[-1])
		user_his_emb = self.attention(item_feats_emb2d, history_feats_emb2d, hislens2d) # B*num_item, item_feature*vec_size
		if context_feats_numeric.numel() != 0:
			# context_emb = self.context_embedding(context_feats).squeeze(dim=-2).view(-1, feats_emb)
			context_feats_linear = torch.zeros(list(context_feats_numeric.shape)+[self.vec_size]).to(item_feats_emb.device)
			for i in range(context_feats_numeric.shape[2]):
				context_feats_linear[:,:,i,:] = self.context_linears[i](context_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1) # B,num_item,context_feature,vec_size
			context_emb = context_feats_linear.flatten(start_dim=-2) # (batch_size, item_num, num_features, embedding_size)
			din = torch.cat([user_his_emb, item_feats_emb2d, user_his_emb*item_feats_emb2d, user_feats_emb2d, context_emb.view(-1,context_emb.shape[-1])], dim=-1)
		else:    # B*num_item, 256 ; B*num_item, 256 ; B*num_item, 256 ; B*num_item, 64; B*num_item, 1408
			din = torch.cat([user_his_emb, item_feats_emb2d, user_his_emb*item_feats_emb2d, user_feats_emb2d], dim=-1)
		for layer in self.dnn_mlp_layers:
			din = layer(din)
		predictions = din
		return predictions.view(batch_size, item_num)

	def computing_immers(self, t, behavior_seq1, behavior_seq2, item_feature):
		sequence1_score = self.im_linear1(behavior_seq1)
		sequence2_score = self.im_linear2(behavior_seq2)
		time_score = self.im_params[0] * t**2 + self.im_params[1] * t + self.im_params[2]
		item_represent = self.im_item_embedding(item_feature)
		item_score = torch.mul(time_score, item_represent)
		combined_score = torch.cat((sequence1_score, sequence2_score, item_score), dim=2)
		hidden = torch.relu(self.fc1(combined_score))
		pred_immers = self.fc2(hidden)
		return pred_immers

	def forward(self, feed_dict):
		# Read and embedding:
		hislens = feed_dict['lengths'] # B

		## user
		user_feats = feed_dict['user_features'] # B * user features(at least user_id) # TODO:并没有考虑user feature是numeric的问题
		user_feats_emb = self.user_embedding(user_feats.int()).flatten(start_dim=-2) # B * 1 * (uf*vecsize)

		## item
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		item_feats_emb = self.item_embedding(item_feats.int()) 
		if 'item_features_numeric' in feed_dict:
			item_feats_numeric = feed_dict['item_features_numeric'] # B * item num * item features(no category feature and id)
			item_feats_linear = torch.zeros(list(item_feats_numeric.shape)+[self.vec_size]).to(item_feats_emb.device)
			for i in range(item_feats_numeric.shape[2]):
				item_feats_linear[:,:,i,:] = self.item_linears[i](item_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			item_feats_emb = torch.cat([item_feats_emb, item_feats_linear],dim=2).flatten(start_dim=-2) # B * item num * (if*vecsize)

		## his_item
		history_item_feats = feed_dict['history_item_features'] # B * hislens * item features(at least item_id)
		history_feats_emb = self.item_embedding(history_item_feats.int())
		if 'history_item_features_numeric' in feed_dict:
			history_item_feats_numeric = feed_dict['history_item_features_numeric'] #  # B * item num * item features(no category feature and id)
			history_feats_linear = torch.zeros(list(history_item_feats_numeric.shape)+[self.vec_size]).to(history_feats_emb.device)
			for i in range(item_feats_numeric.shape[2]):
				history_feats_linear[:,:,i,:] = self.item_linears[i](history_item_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			history_feats_emb = torch.cat([history_feats_emb, history_feats_linear],dim=2).flatten(start_dim=-2) # B * hislens * (if*vecsize) # hislens不同应该是取max
		
		## context and immersion
		if self.include_context_features and 'context_features' in feed_dict:
			context_features_immers = feed_dict['context_features'] # # TODO:并没有考虑context feature(非item feature、user feature)是category的问题
			behavior_seq1, behavior_seq2, t = torch.split(context_features_immers, [10, 10, 1], dim=2)
			context_result = torch.cat((t, behavior_seq1.float(), behavior_seq2.float()),dim=2)
			if self.include_immersion and 'item_features_numeric' in feed_dict:
				pred_immers = self.computing_immers(t, behavior_seq1.float(), behavior_seq2.float(), item_feats_numeric)
				context_feats_numeric = torch.cat((pred_immers, context_result), dim=2)
			else:
				context_feats_numeric = context_result
		else:
			context_feats_numeric = torch.empty(0)
		
		# since storage is not supported for all neg items to predict at once, we need to predict one by one
		self.mask_mat = (torch.arange(history_item_feats.shape[1]).view(1,-1)).to(self.device)
		
		predictions = self.attention_and_dnn(item_feats_emb, history_feats_emb, hislens, user_feats_emb, context_feats_numeric)
		sort_idx = (-predictions).argsort(axis=1)
		return {'prediction':predictions}	

class Dice(nn.Module):
	"""The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

	Input shape:
		- 2 dims: [batch_size, embedding_size(features)]
		- 3 dims: [batch_size, num_features, embedding_size(features)]

	Output shape:
		- Same shape as input.

	References
		- [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
		- https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
	"""

	def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
		super(Dice, self).__init__()
		assert dim == 2 or dim == 3

		self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
		self.sigmoid = nn.Sigmoid()
		self.dim = dim

		# wrap alpha in nn.Parameter to make it trainable
		if self.dim == 2:
			self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
		else:
			self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

	def forward(self, x):
		assert x.dim() == self.dim
		if self.dim == 2:
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
		else:
			x = torch.transpose(x, 1, 2)
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
			out = torch.transpose(out, 1, 2)
		return out
