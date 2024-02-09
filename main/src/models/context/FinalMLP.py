import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseModel import ContextModel
from utils.layers import MLP_Block
from utils.layers import ReverseLayerF

class FinalMLP(ContextModel):
	reader = 'ContextReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','num_heads','use_fs']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--mlp1_hidden_units', type=str,default='[64,64,64]')
		parser.add_argument('--mlp1_hidden_activations',type=str,default='ReLU')
		parser.add_argument('--mlp1_dropout',type=float,default=0)
		parser.add_argument('--mlp1_batch_norm',type=int,default=0)
		parser.add_argument('--mlp2_hidden_units', type=str,default='[64,64,64]')
		parser.add_argument('--mlp2_hidden_activations',type=str,default='ReLU')
		parser.add_argument('--mlp2_dropout',type=float,default=0)
		parser.add_argument('--mlp2_batch_norm',type=int,default=0)
		parser.add_argument('--use_fs',type=int,default=1)
		parser.add_argument('--fs_hidden_units',type=str,default='[64]')
		parser.add_argument('--fs1_context',type=str,default='[]')
		parser.add_argument('--fs2_context',type=str,default='[]')
		parser.add_argument('--num_heads',type=int,default=1)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)

		self.include_context_features = corpus.include_context_features
		self.include_immersion = corpus.include_immersion
		self.include_source_domain = corpus.include_source_domain
		self.DANN = corpus.DANN
		
		self.context_feature_dim = sum(corpus.feature_max_categorical.values()) 
		if self.include_context_features:
			if not self.include_immersion:
				self.context_feature_num = 23 +len(corpus.feature_max_numeric)
			else:
				self.context_feature_num = 21 + sum(1 for value in corpus.feature_max_categorical.values() if value > 0) + len(corpus.feature_max_numeric) + corpus.include_context_features
		else:
			self.context_feature_num = sum(1 for value in corpus.feature_max_categorical.values() if value > 0) +len(corpus.feature_max_numeric)
		

		self.embedding_dim = args.emb_size

		self.use_fs = args.use_fs
		self.feature_max = {**corpus.feature_max_categorical, **corpus.feature_max_numeric}
		if self.include_immersion:
			self.feature_max['c_immersion'] = 0
		self.fs1_context = eval(args.fs1_context)
		self.fs2_context = eval(args.fs2_context)
		
		self._define_params(args)
		self.apply(self.init_weights)

	def _define_params(self,args):
		
		self.context_embedding = nn.Embedding(self.context_feature_dim, self.embedding_dim)
		self.linear_embedding = nn.Embedding(self.context_feature_dim, 1)
		self.context_linear = nn.Linear(1,self.embedding_dim)
		self.linear_linear = nn.Linear(1, 1)
		self.context_linears = nn.ModuleList([nn.Linear(1, self.embedding_dim) for _ in range(self.context_feature_num-2)])
		self.linear_linears = nn.ModuleList([nn.Linear(1, 1) for _ in range(self.context_feature_num-2)])
		
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
		
		self.feature_dim = self.context_feature_num * self.embedding_dim # 要改

		# MLP 1
		self.mlp1 = MLP_Block(input_dim=self.feature_dim,output_dim=None,hidden_units=eval(args.mlp1_hidden_units),
						hidden_activations=args.mlp1_hidden_activations,dropout_rates=args.mlp1_dropout,
						batch_norm=args.mlp1_batch_norm)
		self.mlp2 = MLP_Block(input_dim=self.feature_dim,output_dim=None,hidden_units=eval(args.mlp2_hidden_units),
						hidden_activations=args.mlp2_hidden_activations,dropout_rates=args.mlp2_dropout,
						batch_norm=args.mlp2_batch_norm)
		if self.use_fs:
			self.fs_module = FeatureSelection({},self.feature_dim,
									 self.embedding_dim, eval(args.fs_hidden_units),
									 self.fs1_context,self.fs2_context,self.feature_max)
		self.fusion_module = InteractionAggregation(eval(args.mlp1_hidden_units)[-1],
									eval(args.mlp2_hidden_units)[-1],output_dim=1,num_heads=args.num_heads)

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
		# print(reverse_feature.shape)
		domain_output = self.domain_classifier(reverse_feature)
		return domain_output

	def forward(self, feed_dict):
		"""
		Inputs: [X,y]
		"""
		context_features_category = feed_dict['context_mh'] 
		context_features_numeric = feed_dict['context_numeric']
		context_features_immers = feed_dict['context_immers']
		if self.include_context_features:
			# divide data
			behavior_seq1, behavior_seq2, t = torch.split(context_features_immers, [10, 10, 1], dim=2)
			item_feature = context_features_numeric

			feed_dict['c_behavior_like'] = behavior_seq1
			feed_dict['c_behavior_view'] = behavior_seq2
			feed_dict['c_session_order'] = t

			if self.include_immersion:
				# computing immers
				combined_score, pred_immers = self.computing_immers(t, behavior_seq1.float(), behavior_seq2.float(), item_feature)
				context_result = torch.cat((t, behavior_seq1.float(), behavior_seq2.float()),dim=2)
				context_features_numeric = torch.cat((item_feature, pred_immers, context_result), dim=2)
				feed_dict['c_immersion'] = pred_immers
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

		category_linear = self.linear_embedding(context_features_category).squeeze(dim=-1)
		if context_features_numeric.numel()==0:
			linear_value = self.overall_bias + category_linear
		else:
			numeric_linear = torch.zeros(context_features_numeric.shape).to(category_linear.device)
			for i in range(context_features_numeric.shape[2]):
				numeric_linear[:,:,i] = self.linear_linears[i](context_features_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			linear_value = self.overall_bias + torch.cat([category_linear,numeric_linear],dim=2)
		linear_value = linear_value.sum(dim=-1)

		context_vectors_category = self.context_embedding(context_features_category)
		if context_features_numeric.numel()==0:
			context_emb = context_vectors_category
		else:
			context_vectors_numeric = torch.zeros(list(context_features_numeric.shape)+[self.embedding_dim]).to(context_vectors_category.device)
			for i in range(context_features_numeric.shape[2]):
				context_vectors_numeric[:,:,i,:] = self.context_linears[i](context_features_numeric[:,:,i].unsqueeze(-1)).squeeze(-1)
			context_emb = torch.cat([context_vectors_category,context_vectors_numeric],dim=2)  # (batch_size, item_num, num_features, embedding_size)

		flat_emb = context_emb.flatten(start_dim=-2)
		
		if self.use_fs:
			feat1, feat2 = self.fs_module(feed_dict, flat_emb)
		else:
			feat1, feat2 = flat_emb, flat_emb
		emb_dim1, emb_dim2 = feat1.shape[-1], feat2.shape[-1]
		batch_size, item_num = feat1.shape[0], feat1.shape[1]
		mlp1_output = self.mlp1(feat1.view(-1,emb_dim1)).view(batch_size, item_num, -1)
		mlp2_output = self.mlp1(feat1.view(-1,emb_dim2)).view(batch_size, item_num, -1)
		y_pred = self.fusion_module(mlp1_output, mlp2_output) 
		# y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
		# y_pred = self.output_activation(y_pred)
		predictions = y_pred # 这里看一下用不用squeeze

		if self.training and self.DANN:
			return {'prediction':predictions, 
        			'source_pred_immers':source_pred_immers, 'source_label':source_label, 
          			'source_d_out':source_d_out, 'target_d_out':target_d_out
            		}
		else:
			return {'prediction':predictions}

	def get_output_activation(self, output_sigmoid):
		if output_sigmoid:
			return nn.Sigmoid()
		else:
			return nn.Identity()

	# class Dataset(ContextModel.Dataset):
	# 	def __init__(self, model, corpus, phase):
	# 		super().__init__(model, corpus, phase)
	# 		self.remain_features = list(set(model.fs1_context)|set(model.fs2_context))

	# 	def _get_feed_dict(self, index):
	# 		feed_dict = super()._get_feed_dict(index)
	# 		for f in self.remain_features:
	# 			if f.startswith('c_'):
	# 				feed_dict[f] = self.data[f][index]
	# 			elif f.startswith('i_'):
	# 				feed_dict[f] = np.array([self.corpus.item_features[iid][f] for iid in feed_dict['item_id']])
	# 		return feed_dict

class FeatureSelection(nn.Module):
	def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], 
				 fs1_context=[], fs2_context=[],feature_maxn=dict()):
		super(FeatureSelection, self).__init__()
		self.fs1_context = fs1_context
		if len(fs1_context) == 0:
			self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
		else:
			'''
			https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/embeddings/feature_embedding.py
			'''
			self.fs1_ctx_emb = []
			for ctx in fs1_context:
				if ctx in feature_maxn and feature_maxn[ctx]<=2:
					self.fs1_ctx_emb.append(nn.Linear(1,embedding_dim))
				elif ctx in feature_maxn:
					self.fs1_ctx_emb.append(nn.Embedding(feature_maxn[ctx],embedding_dim))
			
				else:
					raise ValueError("Undifined context %s"%(ctx))
			self.fs1_ctx_emb = nn.ModuleList(self.fs1_ctx_emb)
			# self.fs1_ctx_emb = nn.Embedding(feature_map, embedding_dim, # 应该是feature_embedding；
			# 									required_feature_columns=fs1_context)
		self.fs2_context = fs2_context
		if len(fs2_context) == 0:
			self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
		else:
			self.fs2_ctx_emb = []
			for ctx in fs2_context:
				if ctx in feature_maxn and feature_maxn[ctx]<=2:
					self.fs2_ctx_emb.append(nn.Linear(1,embedding_dim))
				elif ctx in feature_maxn:
					self.fs2_ctx_emb.append(nn.Embedding(feature_maxn[ctx],embedding_dim))
				# if ctx.endswith("_c") and ctx in feature_maxn:
				# 	self.fs2_ctx_emb.append(nn.Embedding(feature_maxn[ctx],embedding_dim))
				# elif ctx.endswith("_f"):
				# 	self.fs2_ctx_emb.append(nn.Linear(1,embedding_dim))
				else:
					raise ValueError("Undifined context %s"%(ctx))
			self.fs2_ctx_emb = nn.ModuleList(self.fs2_ctx_emb)
			# self.fs2_ctx_emb = nn.Embedding(feature_map, embedding_dim,
			# 									required_feature_columns=fs2_context)
		self.fs1_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
								  output_dim=feature_dim,
								  hidden_units=fs_hidden_units,
								  hidden_activations="ReLU",
								  output_activation="Sigmoid",
								  batch_norm=False)
		self.fs2_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs2_context)),
								  output_dim=feature_dim,
								  hidden_units=fs_hidden_units,
								  hidden_activations="ReLU",
								  output_activation="Sigmoid",
								  batch_norm=False)
		
	def forward(self, feed_dict, flat_emb):# flat_emb: (batch_size, num_item, num_feauture * emb_size)
		if len(self.fs1_context) == 0:
			fs1_input = self.fs1_ctx_bias.unsqueeze(1).repeat(flat_emb.size(0),flat_emb.size(1), 1)
		else:
			fs1_input = []
			for i,ctx in enumerate(self.fs1_context):
				if ctx=='user_id': 
					ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx].int().unsqueeze(-1)) # 'user_id': (batch_size) -> (batch_size, 1, emb_size) 
					fs1_input.append(ctx_emb.repeat(1,flat_emb.size(1),1))
				elif ctx=='item_id':
					ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx].int()) # 'item_id': (batch_size, num_item) -> (batch_size, 1, num_item, emb_size)
					fs1_input.append(ctx_emb) 
				# 其实还应该考虑其他的category feature的处理
				elif ctx.startswith('u'):
					ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx].unsqueeze(-1))
					fs1_input.append(ctx_emb.repeat(1,flat_emb.size(1),1))
				else: 
					ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx]) # 对于c_immersion：(batch_size, num_item, 1) -> (batch_size, num_item, emb_size)
					fs1_input.append(ctx_emb)
			fs1_input = torch.cat(fs1_input,dim=-1)
		gt1 = self.fs1_gate(fs1_input) * 2
		feature1 = flat_emb * gt1
		if len(self.fs2_context) == 0:
			fs2_input = self.fs2_ctx_bias.unsqueeze(1).repeat(flat_emb.size(0),flat_emb.size(1), 1)
		else:
			fs2_input = []
			for i,ctx in enumerate(self.fs2_context):
				if ctx=='user_id': 
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx].int().unsqueeze(-1)) # 'user_id': (batch_size) -> (batch_size, 1, emb_size) 
					fs2_input.append(ctx_emb.repeat(1,flat_emb.size(1),1))
				elif ctx=='item_id':
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx].int()) # 'item_id': (batch_size, num_item) -> (batch_size, 1, num_item, emb_size)
					fs2_input.append(ctx_emb) 
				# 其实还应该考虑其他的category feature的处理
				elif ctx.startswith('u'):
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx].unsqueeze(-1))
					fs2_input.append(ctx_emb.repeat(1,flat_emb.size(1),1))
				else: 
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx]) # 对于c_immersion：(batch_size, num_item, 1) -> (batch_size, num_item, emb_size)
					fs2_input.append(ctx_emb) 
			fs2_input = torch.cat(fs2_input,dim=-1)
		gt2 = self.fs2_gate(fs2_input) * 2
		feature2 = flat_emb * gt2
		return feature1, feature2

class InteractionAggregation(nn.Module):
	def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
		super(InteractionAggregation, self).__init__()
		assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
			"Input dim must be divisible by num_heads!"
		self.num_heads = num_heads
		self.output_dim = output_dim
		self.head_x_dim = x_dim // num_heads
		self.head_y_dim = y_dim // num_heads
		self.w_x = nn.Linear(x_dim, output_dim)
		self.w_y = nn.Linear(y_dim, output_dim)
		self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
											  output_dim))
		nn.init.xavier_normal_(self.w_xy)

	def forward(self, x, y):
		batch_size, item_num = x.shape[0], x.shape[1]
		output = self.w_x(x) + self.w_y(y)
		head_x = x.view(batch_size, item_num, self.num_heads, self.head_x_dim).flatten(start_dim=0,end_dim=1)
		head_y = y.view(batch_size, item_num, self.num_heads, self.head_y_dim).flatten(start_dim=0,end_dim=1)
		xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
									   self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
							   .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
						  head_y.unsqueeze(-1)).squeeze(-1)
		xy_reshape = xy.sum(dim=1).view(batch_size,item_num,-1)
		output += xy_reshape
		return output.squeeze(-1)