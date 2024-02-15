# -*- coding: UTF-8 -*-
""" Reference:
  Wide {\&} Deep Learning for Recommender Systems, Cheng et al. 2016.
	The 1st workshop on deep learning for recommender systems.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from models.BaseModel import ContextModel
from utils.layers import ReverseLayerF

class WideDeep(ContextModel):
	reader = 'ContextReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','layers']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each layer.")
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.context_feature_dim = sum(corpus.feature_max_categorical.values()) 
		self.include_context_features = corpus.include_context_features
		self.include_immersion = corpus.include_immersion
		self.include_source_domain = corpus.include_source_domain
		self.DANN = corpus.DANN
		self.save_source_immers = args.save_source_immers
		if self.save_source_immers:
			self.save_source_path = os.path.join(args.log_immers_dir, args.model_path.split('model/')[1].split('.pt')[0]+'_source.npy')
			print('Saving source immersion to:',self.save_source_path)

		if self.include_context_features:
			if not self.include_immersion:
				self.context_feature_num = 23 +len(corpus.feature_max_numeric)
			else:
				self.context_feature_num = 21 + sum(1 for value in corpus.feature_max_categorical.values() if value > 0) + len(corpus.feature_max_numeric) + corpus.include_context_features
		else:
			self.context_feature_num = sum(1 for value in corpus.feature_max_categorical.values() if value > 0) +len(corpus.feature_max_numeric)
		
		self.vec_size = args.emb_size
		self.layers = eval(args.layers)
		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.deep_embedding = nn.Embedding(self.context_feature_dim, self.vec_size)
		self.wide_embedding = nn.Embedding(self.context_feature_dim, 1)
		self.deep_linear = nn.Linear(1,self.vec_size)
		self.wide_linear = nn.Linear(1, 1)
		self.deep_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.context_feature_num-2)])
		self.wide_linears = nn.ModuleList([nn.Linear(1, 1) for _ in range(self.context_feature_num-2)])
  
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
		pre_size = self.context_feature_num * self.vec_size
		self.deep_layers = torch.nn.ModuleList()
		for size in self.layers:
			self.deep_layers.append(torch.nn.Linear(pre_size, size))
			# self.deep_layers.append(torch.nn.BatchNorm1d(size))
			self.deep_layers.append(torch.nn.ReLU())
			self.deep_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		if len(self.layers):
			self.deep_layers.append(torch.nn.Linear(pre_size, 1))

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
		print(reverse_feature.shape)
		domain_output = self.domain_classifier(reverse_feature)
		return domain_output

	def forward(self, feed_dict):
		context_features_category = feed_dict['context_mh']
		context_features_numeric = feed_dict['context_numeric']
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
			source_behavior_seq1, source_behavior_seq2, source_t, source_item_feature, source_label = torch.split(feed_dict['source_data'][0],[10,10,1,3,1],dim=1)
			source_combined_score, source_pred_immers = self.computing_immers(source_t, source_behavior_seq1.float(), source_behavior_seq2.float(), source_item_feature)

			source_shape = source_combined_score.size()
			target_shape = combined_score.size()
			source_shuffled = torch.randperm(source_shape[:-1].numel())
			sample_source = source_combined_score.view(-1, source_shape[-1])[source_shuffled][:target_shape[0]*target_shape[1]]
			target_shuffled = torch.randperm(target_shape[:-1].numel())
			sample_target = combined_score.view(-1, target_shape[-1])[target_shuffled][:target_shape[0]*target_shape[1]]
			
			source_d_out = self.classify_domain(sample_source, feed_dict['batch'], feed_dict['epoch'],feed_dict['n_epoch'],feed_dict['len_dataloader'])
			target_d_out = self.classify_domain(sample_target, feed_dict['batch'], feed_dict['epoch'],feed_dict['n_epoch'],feed_dict['len_dataloader'])
		
  		# wide
		category_wide = self.wide_embedding(context_features_category).squeeze(dim=-1)
		if context_features_numeric.numel()==0:
			wide_prediction = self.overall_bias + category_wide.sum(dim=-1)
		else:
			numeric_wide = torch.zeros(context_features_numeric.shape).to(category_wide.device)
			for i in range(context_features_numeric.shape[2]):
				numeric_wide[:,:,i] = self.wide_linears[i](context_features_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			wide_prediction = self.overall_bias + torch.cat([category_wide,numeric_wide],dim=2).sum(dim=-1)
		
		# deep
		category_deep = self.deep_embedding(context_features_category)
		if context_features_numeric.numel()==0:
			deep_vectors = category_deep.flatten(start_dim=-2)
		else:
			numeric_deep = torch.zeros(list(context_features_numeric.shape)+[self.vec_size]).to(category_deep.device)
			for i in range(context_features_numeric.shape[2]):
				numeric_deep[:,:,i,:] = self.deep_linears[i](context_features_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			deep_vectors = torch.cat([category_deep,numeric_deep],dim=2).flatten(start_dim=-2)
		if len(self.layers):
			for layer in self.deep_layers:
				deep_vectors = layer(deep_vectors)
			deep_prediction = deep_vectors.squeeze(dim=-1)
			predictions = deep_prediction + wide_prediction
		else:
			predictions = wide_prediction
		
		if self.save_source_immers and not os.path.exists(self.save_source_path):
			source_behavior_seq1, source_behavior_seq2, source_t, source_item_feature, source_label = torch.split(feed_dict['source_data'][0],[10,10,1,3,1],dim=1)
			_, source_pred_immers = self.computing_immers(source_t, source_behavior_seq1.float(), source_behavior_seq2.float(), source_item_feature)
			print(source_pred_immers.shape)
			np.save(self.save_source_path, source_pred_immers.cpu().data.numpy())
			np.save(self.save_source_path.split('.npy')[0]+'_data.npy', feed_dict['source_data'][0].cpu().data.numpy())
			print('save source done')

		if self.training and self.DANN:
			return {'prediction':predictions, 
        			'source_pred_immers':source_pred_immers, 'source_label':source_label, 
          			'source_d_out':source_d_out, 'target_d_out':target_d_out
            		}
		else:
			if self.include_immersion:
				return {'prediction':predictions,'immersion':pred_immers.squeeze(dim=-1)}
			else:
				return {'prediction':predictions}