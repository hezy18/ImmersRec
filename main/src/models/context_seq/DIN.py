""" DIN
Reference:'Deep Interest Network for Click-Through Rate Prediction', Zhou et al, KDD2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseModel import ContextSeqModel
from utils.layers import ReverseLayerF

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
		self.include_context_features = corpus.include_context_features
		self.include_immersion = corpus.include_immersion
		self.include_source_domain = corpus.include_source_domain
		if self.include_source_domain:
			self.source_data = corpus.source_data
		self.DANN = corpus.DANN
		
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
		self.user_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.user_feature_num-1)])
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

		self.domain_classifier = nn.Sequential()
		self.domain_classifier.add_module('d_fc1', nn.Linear(9, 100))
		self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
		self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
		self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
		self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

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
			# context_feats_linear = torch.zeros(list(context_feats_numeric.shape)+[self.vec_size]).to(item_feats_emb.device)
			context_feats_linear = torch.zeros((*context_feats_numeric.shape, self.vec_size), device=self.device)
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
		if len(t.shape)==2:
			combined_score = torch.cat((sequence1_score, sequence2_score, item_score), dim=1)
		else:
			combined_score = torch.cat((sequence1_score, sequence2_score, item_score), dim=2)
		hidden = torch.relu(self.fc1(combined_score))
		pred_immers = self.fc2(hidden)
		return combined_score, pred_immers

	def classify_domain(self, feature, batch=1, epoch=1, n_epoch=200, len_dataloader=200):
		p = float(batch + epoch * len_dataloader) / n_epoch / len_dataloader
		alpha = 2. / (1. + np.exp(-10 * p)) - 1
		reverse_feature = ReverseLayerF.apply(feature, alpha)
		# print(reverse_feature.shape)
		domain_output = self.domain_classifier(reverse_feature)
		return domain_output

	def forward(self, feed_dict):
		# Read and embedding:
		hislens = feed_dict['lengths'] # B

		## user
		user_feats = feed_dict['user_features'] # B * user features(at least user_id) 
		user_feats_emb = self.user_embedding(user_feats.int()).flatten(start_dim=-2) # B * 1 * (uf*vecsize)

		## item
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		item_feats_emb = self.item_embedding(item_feats.int()) 
		if 'item_features_numeric' in feed_dict:
			item_feats_numeric = feed_dict['item_features_numeric'] # B * item num * item features(no category feature and id)
			# item_feats_linear = torch.zeros(list(item_feats_numeric.shape)+[self.vec_size]).to(item_feats_emb.device)
			item_feats_linear = torch.zeros((*item_feats_numeric.shape, self.vec_size), device=self.device)
			for i in range(item_feats_numeric.shape[2]):
				item_feats_linear[:,:,i,:] = self.item_linears[i](item_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			item_feats_emb = torch.cat([item_feats_emb, item_feats_linear],dim=2).flatten(start_dim=-2) # B * item num * (if*vecsize)

		## his_item
		history_item_feats = feed_dict['history_item_features'] # B * hislens * item features(at least item_id)
		history_feats_emb = self.item_embedding(history_item_feats.int())
		if 'history_item_features_numeric' in feed_dict:
			history_item_feats_numeric = feed_dict['history_item_features_numeric'] #  # B * item num * item features(no category feature and id)
			# history_feats_linear = torch.zeros(list(history_item_feats_numeric.shape)+[self.vec_size]).to(history_feats_emb.device)
			history_feats_linear = torch.zeros((*history_item_feats_numeric.shape, self.vec_size), device=self.device)
			for i in range(item_feats_numeric.shape[2]):
				history_feats_linear[:,:,i,:] = self.item_linears[i](history_item_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			history_feats_emb = torch.cat([history_feats_emb, history_feats_linear],dim=2).flatten(start_dim=-2) # B * hislens * (if*vecsize) 
		
		## context and immersion
		if self.include_context_features and 'context_features' in feed_dict:
			context_features_immers = feed_dict['context_features'] 
			behavior_seq1, behavior_seq2, t = torch.split(context_features_immers, [10, 10, 1], dim=2)
			context_result = torch.cat((t, behavior_seq1.float(), behavior_seq2.float()),dim=2)
			if self.include_immersion and 'item_features_numeric' in feed_dict:
				combined_score, pred_immers = self.computing_immers(t, behavior_seq1.float(), behavior_seq2.float(), item_feats_numeric)
				context_feats_numeric = torch.cat((pred_immers, context_result), dim=2)
			else:
				context_feats_numeric = context_result
		else:
			context_feats_numeric = torch.empty(0)

		if self.training and self.include_source_domain and self.DANN:
			source_behavior_seq1, source_behavior_seq2, source_t, source_item_feature, source_label = torch.split(torch.Tensor(self.source_data).to(self.device),[10,10,1,3,1],dim=1)
			source_combined_score, source_pred_immers = self.computing_immers(source_t, source_behavior_seq1.float(), source_behavior_seq2.float(), source_item_feature)

			source_shape = source_combined_score.size()
			target_shape = combined_score.size()
			source_shuffled = torch.randperm(source_shape[:-1].numel())
			sample_source = source_combined_score.view(-1, source_shape[-1])[source_shuffled][:target_shape[0]*target_shape[1]]
			target_shuffled = torch.randperm(target_shape[:-1].numel())
			sample_target = combined_score.view(-1, target_shape[-1])[target_shuffled][:target_shape[0]*target_shape[1]]
			
			source_d_out = self.classify_domain(sample_source, feed_dict['batch'], feed_dict['epoch'],feed_dict['n_epoch'],feed_dict['len_dataloader'])
			target_d_out = self.classify_domain(sample_target, feed_dict['batch'], feed_dict['epoch'],feed_dict['n_epoch'],feed_dict['len_dataloader'])

		
		# since storage is not supported for all neg items to predict at once, we need to predict one by one
		# self.mask_mat = (torch.arange(history_item_feats.shape[1]).view(1,-1)).to(self.device)
		self.mask_mat = torch.arange(history_item_feats.shape[1], device=self.device).view(1, -1)

		
		predictions = self.attention_and_dnn(item_feats_emb, history_feats_emb, hislens, user_feats_emb, context_feats_numeric)

		sort_idx = (-predictions).argsort(axis=1)
		if predictions.shape==(0,):
			print('1')
		if self.training and self.DANN:
			return {'prediction':predictions, 
        			'source_pred_immers':source_pred_immers, 'source_label':source_label, 
          			'source_d_out':source_d_out, 'target_d_out':target_d_out
            		}
		else:
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
