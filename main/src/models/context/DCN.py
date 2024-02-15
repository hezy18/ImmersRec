""" DCN
Reference:
	'Deep & Cross Network for Ad Click Predictions', Wang et al, KDD2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseModel import ContextModel
from utils.layers import ReverseLayerF

class DCN(ContextModel):
	reader = 'ContextReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','cross_layer_num','dropout']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each deep layer.")
		parser.add_argument('--cross_layer_num',type=int,default=6,
							help="Number of layers.")
		parser.add_argument('--reg_weight',type=float, default=2.0) # 并未用上
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.context_feature_dim = sum(corpus.feature_max_categorical.values()) 
		# self.context_feature_num = len(corpus.feature_max_categorical)
		self.include_context_features = corpus.include_context_features
		self.include_immersion = corpus.include_immersion
		self.include_source_domain = corpus.include_source_domain
		if self.include_source_domain:
			self.source_data = corpus.source_data
		self.DANN = corpus.DANN

		if self.include_context_features:
			if not self.include_immersion:
				self.context_feature_num = 23 +len(corpus.feature_max_numeric)
			else:
				self.context_feature_num = 21 + sum(1 for value in corpus.feature_max_categorical.values() if value > 0) + len(corpus.feature_max_numeric) + corpus.include_context_features
		else:
			self.context_feature_num = sum(1 for value in corpus.feature_max_categorical.values() if value > 0) +len(corpus.feature_max_numeric)
		
		self.reg_weight = args.reg_weight
		self.vec_size = args.emb_size
		self.layers = eval(args.layers)
		self.cross_layer_num = args.cross_layer_num
		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.context_embedding = nn.Embedding(self.context_feature_dim, self.vec_size)
		self.linear_embedding = nn.Embedding(self.context_feature_dim, 1)
		self.context_linear = nn.Linear(1,self.vec_size)
		self.linear_linear = nn.Linear(1, 1)
		self.context_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.context_feature_num-2)])
		self.linear_linears = nn.ModuleList([nn.Linear(1, 1) for _ in range(self.context_feature_num-2)])

		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
		pre_size = self.context_feature_num * self.vec_size
		# cross layers
		self.cross_layer_w = nn.ParameterList(nn.Parameter(torch.randn(pre_size),requires_grad=True) 
							for l in range(self.cross_layer_num))
		self.cross_layer_b = nn.ParameterList(nn.Parameter(torch.tensor([0.01]*pre_size),requires_grad=True)
							for l in range(self.cross_layer_num))

		# deep layers
		self.deep_layers = torch.nn.ModuleList()
		for size in self.layers:
			self.deep_layers.append(torch.nn.Linear(pre_size, size))
			self.deep_layers.append(torch.nn.BatchNorm1d(num_features=size))
			self.deep_layers.append(torch.nn.ReLU())
			self.deep_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		
		# output layer
		self.predict_layer = nn.Linear(self.context_feature_num * self.vec_size + self.layers[-1], 1)

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
		domain_output = self.domain_classifier(reverse_feature)
		return domain_output
	
	def l2_reg(self, parameters):
		"""
		Reference: RecBole
		RegLoss, L2 regularization on model parameters
		"""
		reg_loss = None
		for W in parameters:
			if reg_loss is None:
				reg_loss = W.norm(2)
			else:
				reg_loss = reg_loss + W.norm(2)
		return reg_loss

	def cross_net(self, x_0):
		# x_0: batch size * item num * embedding size
		'''Reference: RecBole
		'''
		x_l = x_0
		for layer in range(self.cross_layer_num):
			xl_w = torch.tensordot(x_l, self.cross_layer_w[layer], dims=([-1],[0]))
			xl_dot = x_0 * xl_w.unsqueeze(2)
			x_l = xl_dot + self.cross_layer_b[layer] + x_l
		return x_l

	def forward(self, feed_dict):
		context_features_category = feed_dict['context_mh'] #if behavior: batch * item num * (1+2*10+['user_id','item_id']) else: batch * item num * ['user_id','item_id']
		context_features_numeric = feed_dict['context_numeric'] # if item_feature: batch * item num * ['i_duration', 'i_vvall','i_likecnt']
		context_features_immers = feed_dict['context_immers']

		if self.include_context_features:
			# divide data
			behavior_seq1, behavior_seq2, t = torch.split(context_features_immers, [10, 10, 1], dim=2)
			item_feature = context_features_numeric

			if self.include_immersion:
				# computing immers
				combined_score, pred_immers = self.computing_immers(t, behavior_seq1.float(), behavior_seq2.float(), item_feature)
				context_result = torch.cat((t, behavior_seq1.float(), behavior_seq2.float()),dim=2)
				context_features_numeric = torch.cat((item_feature, pred_immers, context_result), dim=2)
			else:
				context_result = torch.cat((t, behavior_seq1.float(), behavior_seq2.float()),dim=2)
				context_features_numeric = torch.cat((item_feature, context_result), dim=2)

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
		
		category_linear = self.linear_embedding(context_features_category).squeeze(dim=-1)
		if context_features_numeric.numel()==0:
			linear_value = self.overall_bias + category_linear
		else:
			# numeric_linear = torch.zeros(context_features_numeric.shape).to(category_linear.device)
			numeric_linear = torch.zeros_like(context_features_numeric,device=self.device)
			for i in range(context_features_numeric.shape[2]):
				numeric_linear[:,:,i] = self.linear_linears[i](context_features_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			linear_value = self.overall_bias + torch.cat([category_linear,numeric_linear],dim=2)
		linear_value = linear_value.sum(dim=-1)

		context_vectors_category = self.context_embedding(context_features_category)
		if context_features_numeric.numel()==0:
			context_emb = context_vectors_category
		else:
			# context_vectors_numeric = torch.zeros(list(context_features_numeric.shape)+[self.vec_size]).to(context_vectors_category.device)
			context_vectors_numeric = torch.zeros((*context_features_numeric.shape, self.vec_size), device=self.device)
			for i in range(context_features_numeric.shape[2]):
				context_vectors_numeric[:,:,i,:] = self.context_linears[i](context_features_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			context_emb = torch.cat([context_vectors_category,context_vectors_numeric],dim=2)  # (batch_size, item_num, num_features, embedding_size)

		context_emb = context_emb.flatten(start_dim=-2)
		# context_emb = self.context_embedding(context_features).flatten(start_dim=-2) # Batch size * item num * (feature*vec_size)

		cross_output = self.cross_net(context_emb)
		batch_size, item_num, output_emb = cross_output.shape
		deep_output = context_emb.view(-1,output_emb)
		for layer in self.deep_layers:
			deep_output = layer(deep_output)
		deep_output = deep_output.view(batch_size, item_num, self.layers[-1])
		
		output = self.predict_layer(torch.cat([cross_output, deep_output],dim=-1))
		predictions = output.squeeze(-1)

		if self.training and self.DANN:
			return {'prediction':predictions, 
        			'source_pred_immers':source_pred_immers, 'source_label':source_label, 
          			'source_d_out':source_d_out, 'target_d_out':target_d_out
            		}
		else:
			return {'prediction':predictions}
	
	def loss(self, out_dict: dict):
		l2_loss = self.reg_weight * self.l2_reg(self.cross_layer_w)
		loss = super().loss(out_dict)
		return loss + l2_loss
