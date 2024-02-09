# -*- coding: UTF-8 -*-

""" SDIM
Reference:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pandas.core.common import flatten

from models.BaseModel import ContextSeqModel
from utils.layers import MultiHeadTargetAttention, MLP_Block

import logging

class SDIM(ContextSeqModel):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--dnn_hidden_units',type=str,default='[128,64]') # [512,128,64]
		parser.add_argument('--dnn_activations',type=str,default='ReLU')
		parser.add_argument('--attention_dim',type=int,default=64)
		parser.add_argument('--use_qkvo',type=int,default=1, help="True/False")
		parser.add_argument('--num_heads',type=int,default=1)
		parser.add_argument('--use_scale',type=int,default=1, help="True/False")
		parser.add_argument('--attention_dropout',type=float,default=0)
		parser.add_argument('--reuse_hash',type=int,default=1, help="True/False")
		parser.add_argument('--num_hashes',type=int,default=1)
		parser.add_argument('--hash_bits',type=int,default=4)
		parser.add_argument('--net_dropout',type=float,default=0)
		parser.add_argument('--batch_norm',type=int,default=0, help="whether use batch_norm or not")
		parser.add_argument('--short_target_field',type=str,default='["item","context"]',
					  help="select from item (will include id), item id, and situation.")
		parser.add_argument('--short_sequence_field',type=str,default='["item","context"]')
		parser.add_argument('--long_target_field',type=str,default='["item","context"]')
		parser.add_argument('--long_sequence_field',type=str,default='["item","context"]')
		parser.add_argument('--output_sigmoid',type=int,default=0)
		parser.add_argument('--group_attention',type=int,default=1)
		parser.add_argument('--all_group_one',type=int,default=0)
		parser.add_argument('--short_history_max',type=int,default=10)
		return ContextSeqModel.parse_model_args(parser)

	def get_feature_from_field(self, field_list,status="target"):
		feature_list = []
		for field in field_list:
			if field == "item":
				features = ['item_id']
				if self.include_item_features:
					features.append('item_features')
			elif field == "context" and self.include_context_features:
				features = ['context_behavior1', 'context_behavior2', 'context_t']
				if self.include_immersion:
					features.append('context_immers')
			else:
				logging.info("Field %s not defined!"%(field))
				continue
			if status == 'seq' and field == "item":
				features = ['his_'+f for f in features]
			if self.group_attention:
				feature_list.append(tuple(features))
			else:
				feature_list += features 
		if self.all_group_one:
			feature_list_new = []
			for f in feature_list:
				if type(f)==tuple:
					feature_list_new += list(f)
				else:
					feature_list_new.append(f)
			return [tuple(feature_list_new)]
		return feature_list

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
		self.include_item_features = corpus.include_item_features
		self.include_immersion = corpus.include_immersion
		#TODO: 以下部分写死了，之后要改
		if self.include_context_features:
			if self.include_immersion:
				self.context_feature_num = 22
			else:
				self.context_feature_num = 21
		else:
			self.context_feature_num = 0

		# Hash
		self.reuse_hash = args.reuse_hash # True/False
		self.num_hashes = args.num_hashes
		self.hash_bits = args.hash_bits
		self.group_attention = args.group_attention
		self.all_group_one = args.all_group_one

		# feature_emb
		self.short_history_max = args.short_history_max
		
		self.vec_size = args.emb_size

		# target           
		# self.short_target_field = eval(args.short_target_field)
		# self.short_sequence_field = eval(args.short_sequence_field)
		# self.long_target_field = eval(args.long_target_field)
		# self.long_sequence_field = eval(args.long_sequence_field)
		self.short_target_field = self.get_feature_from_field(eval(args.short_target_field))
		self.short_sequence_field = self.get_feature_from_field(eval(args.short_sequence_field),"seq")
		self.long_target_field = self.get_feature_from_field(eval(args.long_target_field))
		self.long_sequence_field = self.get_feature_from_field(eval(args.long_sequence_field),"seq")
		assert len(self.short_target_field) == len(self.short_sequence_field) \
			   and len(self.long_target_field) == len(self.long_sequence_field), \
			   "Config error: target_field mismatches with sequence_field."

		self._define_params(args)
		self.apply(self.init_weights)
	
	def _define_params(self,args):
		# embeddings TODO：之后再对照
		self.user_embedding = nn.Embedding(self.user_feature_dim, self.vec_size)
		self.item_embedding = nn.Embedding(self.item_feature_dim, self.vec_size)
		self.user_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.user_feature_num-1)]) # 这个-1也是写死的
		self.item_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.item_feature_num-1)])
		self.context_linears = nn.ModuleList([nn.Linear(1, self.vec_size) for _ in range(self.context_feature_num)])

		self.powers_of_two = nn.Parameter(torch.tensor([2.0 ** i for i in range(self.hash_bits)]),requires_grad=False)

		
		pre_feature_num = 0
		# short
		self.short_attention = nn.ModuleList()
		for target_field in self.short_target_field:
			if type(target_field) == tuple:
				for target in target_field:
					if target.find('item')!=-1:
						input_dim = self.vec_size * self.item_feature_num
						pre_feature_num += self.item_feature_num
						break
					elif target.find('context')!=-1:
						input_dim = self.vec_size * self.context_feature_num
						pre_feature_num += self.context_feature_num
						break
			else:
				if target_field.find('item')!=-1:
					input_dim = self.vec_size * self.item_feature_num
					pre_feature_num += self.item_feature_num
				elif target.find('context')!=-1:
					input_dim = self.vec_size * self.context_feature_num
					pre_feature_num += self.context_feature_num
			self.short_attention.append(MultiHeadTargetAttention(
				input_dim, args.attention_dim, args.num_heads,
				args.attention_dropout, args.use_scale, args.use_qkvo,
			))
		# long
		self.random_rotations = nn.ParameterList()
		for target_field in self.long_target_field:
			if type(target_field) == tuple:
				for target in target_field:
					if target.find('item')!=-1:
						input_dim = self.vec_size * self.item_feature_num
						pre_feature_num += self.item_feature_num
						break
					elif target.find('context')!=-1:
						input_dim = self.vec_size * self.context_feature_num
						pre_feature_num += self.context_feature_num
						break
			else:
				if target_field.find('item')!=-1:
					input_dim = self.vec_size * self.item_feature_num
					pre_feature_num += self.item_feature_num
				elif target.find('context')!=-1:
					input_dim = self.vec_size * self.context_feature_num
					pre_feature_num += self.context_feature_num
			self.random_rotations.append(nn.Parameter(torch.randn(input_dim,
								self.num_hashes, self.hash_bits), requires_grad=False))

		# TODO：之后再看加output activation的问题：
		# self.output_activation = self.get_output_activation(args.output_sigmoid)

		# DNN
		# pre_feature_num = self.item_feature_num + self.context_feature_num + \
  					# len(list(flatten(self.long_sequence_field)))+len(list(flatten(self.short_sequence_field)))
		self.dnn = MLP_Block(input_dim=pre_feature_num * self.vec_size,
							 output_dim=1,
							 hidden_units=eval(args.dnn_hidden_units),
							 hidden_activations=args.dnn_activations,
							#  output_activation=self.output_activation, 
							 dropout_rates=args.net_dropout,
							 batch_norm=args.batch_norm)

		# immersion
		if self.include_immersion:
			self.im_linear1 = nn.Linear(10, 4)
			self.im_linear2 = nn.Linear(10, 4)
			self.im_item_embedding = nn.Linear(3, 1)
			self.im_params = nn.Parameter(torch.randn(3))

			self.fc1 = nn.Linear(9, 64)
			self.fc2 = nn.Linear(64, 1)


	def forward(self, feed_dict):
		# Read and embedding:
		hislens = feed_dict['lengths'] # B
		mask = torch.arange(feed_dict['history_items'].shape[1])[None,:].to(self.device) < hislens[:,None] # B * h
		feature_emb_dict = self.get_embeddings(feed_dict)

		feature_emb = []
		# include_features=[]
		# short interest attention
		for idx, (target_field, sequence_field) in enumerate(zip(self.short_target_field, 
																 self.short_sequence_field)):
			target_emb = self.concat_embedding(target_field, feature_emb_dict).flatten(start_dim=-2) # batch * item num * embedding
			sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict).flatten(start_dim=-2) # batch * his_len/item_num * embedding
			target_emb_flatten = target_emb.view(-1,target_emb.size(-1)) # 700*256
			if sequence_field[0].find('item')!=-1:
				sequence_emb_flatten = sequence_emb.unsqueeze(1).repeat(1,target_emb.size(1),1,1).view(
	   						-1,sequence_emb.size(1),sequence_emb.size(2)) # 700*1*256
			elif sequence_field[0].find('context')!=-1:
				sequence_emb_flatten = sequence_emb.unsqueeze(2).repeat(1,int(max(hislens)),1,1).view(
	   					-1,int(max(hislens)),sequence_emb.size(2))
			mask_flatten = mask.unsqueeze(1).repeat(1,target_emb.size(1),1).view(-1,sequence_emb.size(1)) #700*1
			# seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first field
			# mask = X[seq_field].long() != 0 # padding_idx = 0 required in input data
			short_interest_emb_flatten = self.short_attention[idx](target_emb_flatten, sequence_emb_flatten, mask_flatten)
			short_interest_emb = short_interest_emb_flatten.view(target_emb.shape)
			feature_emb.append(short_interest_emb)
			# if sequence_field[0].find('item')!=-1:
			# 	for field, field_emb in zip(list(flatten([sequence_field])),
			# 							short_interest_emb.split([self.vec_size,self.vec_size * 3], dim=-1)):
			# 		feature_emb_dict[field+'_short'] = field_emb
			# 		include_features.append(field+'_short')
			# elif sequence_field[0].find('context')!=-1:
			# 	pass #TODO
				
  
		# long interest attention
		for idx, (target_field, sequence_field) in enumerate(zip(self.long_target_field, 
																 self.long_sequence_field)):
			target_emb = self.concat_embedding(target_field, feature_emb_dict).flatten(start_dim=-2)
			sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict).flatten(start_dim=-2)
			target_emb_flatten = target_emb.view(-1,target_emb.size(-1))
			if sequence_field[0].find('item')!=-1:
				sequence_emb_flatten = sequence_emb.unsqueeze(1).repeat(1,target_emb.size(1),1,1).view(
	   						-1,sequence_emb.size(1),sequence_emb.size(2)) # 700*1*256
			elif sequence_field[0].find('context')!=-1:
				sequence_emb_flatten = sequence_emb.unsqueeze(2).repeat(1,int(max(hislens)),1,1).view(
	   					-1,int(max(hislens)),sequence_emb.size(2))
			long_interest_emb_flatten = self.lsh_attentioin(self.random_rotations[idx], 
													target_emb_flatten, sequence_emb_flatten)
			long_interest_emb = long_interest_emb_flatten.view(target_emb.shape)

			feature_emb.append(long_interest_emb)
			# for field, field_emb in zip(list(flatten([sequence_field])),
			# 							long_interest_emb.split(self.embedding_dim, dim=-1)):
			# 	feature_emb_dict[field+'_long'] = field_emb
			# 	include_features.append(field+'_long')
		
		# concat
		# feature_emb = []
		# for f in sorted(include_features):
		# 	feature_emb.append(feature_emb_dict[f])
		feature_emb = torch.cat(feature_emb,dim=-1)
		# DNN
		batch_size, item_num, emb_dim = feature_emb.shape
		predictions = self.dnn(feature_emb.view(-1,emb_dim)).view(batch_size, item_num, -1).squeeze(-1)
		sort_idx = (-predictions).argsort(axis=1) # For test
		return {'prediction':predictions}	

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

	def get_embeddings(self, feed_dict):		
		feature_emb_dict={}
	
		hislens = feed_dict['lengths'] # B
  
		# TODO: 仅仅为我自己的数据服务，很多情况都没有考虑，之后要改
		## user
		user_feats = feed_dict['user_features'] # B * user features(at least user_id) 
		feature_emb_dict['user_id'] = self.user_embedding(user_feats.int()) #.flatten(start_dim=-2) # B * 1 * 1 * vecsize

		## item
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		feature_emb_dict['item_id'] = self.item_embedding(item_feats.int()) # B * item num * 1 * 64
		if 'item_features_numeric' in feed_dict:
			item_feats_numeric = feed_dict['item_features_numeric'] # B * item num * item features(no category feature and id)
			item_feats_linear = torch.zeros(list(item_feats_numeric.shape)+[self.vec_size]).to(item_feats.device)
			for i in range(item_feats_numeric.shape[2]):
				item_feats_linear[:,:,i,:] = self.item_linears[i](item_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			# item_feats_emb = torch.cat([item_feats_emb, item_feats_linear],dim=2).flatten(start_dim=-2) # B * item num * (if*vecsize)
			feature_emb_dict['item_features'] = item_feats_linear # B * item num * item_feautre * 64

		## his_item
		history_item_feats = feed_dict['history_item_features'] # B * his_len * item features(at least item_id)
		feature_emb_dict['his_item_id'] = self.item_embedding(history_item_feats.int()) # B * his_len * 1 * emb_size
		if 'history_item_features_numeric' in feed_dict:
			history_item_feats_numeric = feed_dict['history_item_features_numeric'] #  # B * his_len * item features(no category feature and id)
			history_feats_linear = torch.zeros(list(history_item_feats_numeric.shape)+[self.vec_size]).to(history_item_feats.device)
			for i in range(item_feats_numeric.shape[2]):
				history_feats_linear[:,:,i,:] = self.item_linears[i](history_item_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			# history_feats_emb = torch.cat([history_feats_emb, history_feats_linear],dim=2).flatten(start_dim=-2) # B * hislens * (if*vecsize) # hislens不同应该是取max
			feature_emb_dict['his_item_features'] = history_feats_linear # B * his_len * item_feature * emb_size
		
		## context and immersion
		if self.include_context_features and 'context_features' in feed_dict:
			context_features = feed_dict['context_features'] 
			behavior_seq1, behavior_seq2, t = torch.split(context_features, [10, 10, 1], dim=2)
			context_result = torch.cat((t, behavior_seq1.float(), behavior_seq2.float()),dim=2)
			if self.include_immersion and 'item_features_numeric' in feed_dict:
				pred_immers = self.computing_immers(t, behavior_seq1.float(), behavior_seq2.float(), item_feats_numeric)
				context_feats_numeric = torch.cat((pred_immers, context_result), dim=2)
				context_vectors_numeric = torch.zeros(list(context_feats_numeric.shape)+[self.vec_size]).to(context_features.device)
				for i in range(context_feats_numeric.shape[2]):
					context_vectors_numeric[:,:,i,:] = self.context_linears[i](context_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
				b1_vec, b2_vec, t_vec, immers_vec = torch.split(context_vectors_numeric, [10, 10, 1, 1], dim=2)
				feature_emb_dict.update({'context_behavior1':b1_vec, 'context_behavior2': b2_vec, 'context_t': t_vec, 'context_immers':immers_vec})
			
			else:
				context_feats_numeric = context_result
				context_vectors_numeric = torch.zeros(list(context_feats_numeric.shape)+[self.vec_size]).to(context_features.device)
				for i in range(context_feats_numeric.shape[2]):
					context_vectors_numeric[:,:,i,:] = self.context_linears[i](context_feats_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
				b1_vec, b2_vec, t_vec, immers_vec = torch.split(context_features, [10, 10, 1], dim=2)
				feature_emb_dict.update({'context_behavior1':b1_vec, 'context_behavior2': b2_vec, 'context_t': t_vec})

		return feature_emb_dict

	def concat_embedding(self, field, feature_emb_dict):
		if type(field) == tuple:
			emb_list = [feature_emb_dict[f] for f in field]
			return torch.cat(emb_list, dim=-2)
		else:
			return feature_emb_dict[field]

	def lsh_attentioin(self, random_rotations, target_item, history_sequence):
		if not self.reuse_hash:
			random_rotations = torch.randn(target_item.size(1), self.num_hashes, 
										   self.hash_bits, device=target_item.device)
		target_bucket = self.lsh_hash(history_sequence, random_rotations)
		sequence_bucket = self.lsh_hash(target_item.unsqueeze(1), random_rotations)
		bucket_match = (sequence_bucket - target_bucket).permute(2, 0, 1) # num_hashes x B x seq_len
		collide_mask = (bucket_match == 0).float()
		hash_index, collide_index = torch.nonzero(collide_mask.flatten(start_dim=1), as_tuple=True)
		offsets = collide_mask.sum(dim=-1).long().flatten().cumsum(dim=0)
		attn_out = F.embedding_bag(collide_index, history_sequence.view(-1, target_item.size(1)), 
								   offsets, mode='sum') # (num_hashes x B) x d
		attn_out = attn_out.view(self.num_hashes, -1, target_item.size(1)).mean(dim=0) # B x d
		return attn_out

	def lsh_hash(self, vecs, random_rotations):
		""" See the tensorflow-lsh-functions for reference:
			https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py
			
			Input: vecs, with shape B x seq_len x d
			Output: hash_bucket, with shape B x seq_len x num_hashes
		"""
		rotated_vecs = torch.einsum("bld,dht->blht", vecs, random_rotations) # B x seq_len x num_hashes x hash_bits
		hash_code = torch.relu(torch.sign(rotated_vecs))
		hash_bucket = torch.matmul(hash_code, self.powers_of_two.unsqueeze(-1)).squeeze(-1)
		return hash_bucket