import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from helpers.BaseReader import BaseReader

# from utils.logger import logger_loss


class BaseModel(nn.Module):
	reader, runner = None, None  # choose helpers in specific model classes
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--model_path', type=str, default='',
							help='Model save path.')
		parser.add_argument('--buffer', type=int, default=1,
							help='Whether to buffer feed dicts for dev/test')
		return parser

	@staticmethod
	def init_weights(m):
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

	def __init__(self, args, corpus: BaseReader):
		super(BaseModel, self).__init__()
		self.device = args.device
		self.model_path = args.model_path
		self.buffer = args.buffer
		self.optimizer = None
		self.check_list = list()  # observe tensors in check_list every check_epoch

	"""
	Key Methods
	"""
	def _define_params(self):
		pass

	def forward(self, feed_dict: dict) -> dict:
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		pass

	def loss(self, out_dict: dict) -> torch.Tensor:
		pass

	"""
	Auxiliary Methods
	"""
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def save_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		utils.check_dir(model_path)
		torch.save(self.state_dict(), model_path)
		# logging.info('Save model to ' + model_path[:50] + '...')

	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)

	def count_variables(self) -> int:
		total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return total_parameters

	def actions_after_train(self):  # e.g., save selected parameters
		pass

	"""
	Define Dataset Class
	"""
	class Dataset(BaseDataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict()
			#self.data = utils.df_to_dict(corpus.data_df[phase])#this raise the VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences warning
			self.data = corpus.data_df[phase].to_dict('list')
			# ↑ DataFrame is not compatible with multi-thread operations

		def __len__(self):
			if type(self.data) == dict:
				for key in self.data:
					return len(self.data[key])
			return len(self.data)

		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			return self._get_feed_dict(index)

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			pass

		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)

		# Called before each training epoch (only for the training dataset)
		def actions_before_epoch(self):
			pass

		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]) -> dict:
			feed_dict = dict()
			for key in feed_dicts[0]:
				if isinstance(feed_dicts[0][key], np.ndarray):
					tmp_list = [len(d[key]) for d in feed_dicts]
					if any([tmp_list[0] != l for l in tmp_list]):
						stack_val = np.array([d[key] for d in feed_dicts], dtype=np.object)
					else:
						stack_val = np.array([d[key] for d in feed_dicts])
				else:
					stack_val = np.array([d[key] for d in feed_dicts])
				if stack_val.dtype == np.object:  # inconsistent length (e.g., history)
					feed_dict[key] = pad_sequence([torch.from_numpy(x).float() for x in stack_val], batch_first=True)
				elif key=='context_mh': 
					feed_dict[key] = torch.from_numpy(stack_val).long()
				else: # '_numeric' or 'source_data'
					feed_dict[key] = torch.from_numpy(stack_val).float()
			feed_dict['batch_size'] = len(feed_dicts)
			feed_dict['phase'] = self.phase
			return feed_dict

class GeneralModel(BaseModel):
	reader, runner = 'BaseReader', 'BaseRunner'
	extra_log_args = ['tradeoff_DA','tradeoff_Clf']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--num_neg', type=int, default=1,
							help='The number of negative items during training.')
		parser.add_argument('--dropout', type=float, default=0,
							help='Dropout probability for each deep layer')
		parser.add_argument('--test_all', type=int, default=0,
							help='Whether testing on all the items.')
		parser.add_argument('--tradeoff_DA', type=float, default=0.001,
                      		help='The weight of DA module.')
		parser.add_argument('--tradeoff_Clf', type=float, default=0.1,
                      		help='The weight of domain classfier loss in DA.')
		return BaseModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items
		self.num_neg = args.num_neg
		self.dropout = args.dropout
		self.test_all = args.test_all
		self.tradeoff_DA = args.tradeoff_DA
		self.tradeoff_Clf = args.tradeoff_Clf
		self.DANN = corpus.DANN
		self.loss_domain = torch.nn.NLLLoss()

	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
		"Recurrent neural networks with top-k gains for session-based recommendations"
		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
		:return:
		"""
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
		
		if self.include_source_domain and self.DANN:
			mse = F.mse_loss(out_dict['source_pred_immers'], out_dict['source_label'])
			# print(mse)
			source_d_out = out_dict['source_d_out']
			target_d_out = out_dict['target_d_out']
				
			self.loss_domain = self.loss_domain.to(target_d_out.device)
			target_loss = self.loss_domain(target_d_out, torch.ones(target_d_out.shape[0]).to(target_d_out.device).long())
			source_loss = self.loss_domain(source_d_out, torch.zeros(source_d_out.shape[0]).to(target_d_out.device).long())
			loss += self.tradeoff_DA*(mse + self.tradeoff_Clf*(target_loss + source_loss))
			# logger_loss.info('final_loss={:.3f}, BPR={:.3f}, mse={:.3f}, target_loss={:.3f}, source_loss={:.3f}'.format(
			# 					loss.item(),loss.item(), mse.item(),target_loss.item(),source_loss.item()))
		# print('BPR={:.3f}'.format(loss.item()))
		# neg_pred = (neg_pred * neg_softmax).sum(dim=1)
		# loss = F.softplus(-(pos_pred - neg_pred)).mean()
		# ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
		# print(loss)
		return loss

	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			if self.phase != 'train' and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			else:
				neg_items = self.data['neg_items'][index]
			item_ids = np.concatenate([[target_item], neg_items]).astype(int)
			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}
			return feed_dict

		# Sample negative items for all the instances
		def actions_before_epoch(self):
			neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
			for i, u in enumerate(self.data['user_id']):
				clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
				# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
				for j in range(self.model.num_neg):
					while neg_items[i][j] in clicked_set:
						neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
			self.data['neg_items'] = neg_items

class SequentialModel(GeneralModel):
	reader = 'SeqReader'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			if self.data['split'][index]:
				user_seq = self.corpus.user_his['dev_test'][feed_dict['user_id']][:pos]
			else:
				user_seq = self.corpus.user_his['train'][feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			if self.corpus.include_his_context:
				history_context=[]
				for x in user_seq:
					tmp_context=[]
					for element in x[2:]:
						if isinstance(element, list):
							tmp_context.extend(element)
						else:
							tmp_context.append(element)
					history_context.append(tmp_context)
				feed_dict['history_context'] = np.array(history_context)
			feed_dict['lengths'] = len(feed_dict['history_items'])
			return feed_dict

class ContextModel(GeneralModel):
	reader = 'ContextReader'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n',type=str,default='BPR',
							help='Type of loss functions.')
		parser.add_argument('--include_id',type=int,default=1,
							help='Whether add ids in context information.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.loss_n = args.loss_n
		self.include_id = args.include_id
	
	def loss(self, out_dict: dict):
		"""
		utilize BPR loss (as general model) or log loss as most context-aware models
		"""
		if self.loss_n == 'BPR':
			return super().loss(out_dict)
		elif self.loss_n == 'BCE':
			predictions = out_dict['prediction'].sigmoid()
			pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
			loss = - (pos_pred.log() + (1-neg_pred).log().sum(dim=1)).mean()
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))
		return loss
	
	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			self.include_id = model.include_id


		def _get_feed_dict(self, index): 
			feed_dict = super()._get_feed_dict(index)
			# if self.corpus.include_source_domain:
			# 	feed_dict = self._get_source_dict(feed_dict) 
			All_context_numeric = [] # concatenate all context information
			All_context_categorical = []
			categorical_feature_list=[]
			All_context_immers =[]
			if len(self.corpus.user_feature_names):
				# user_features = np.array([self.corpus.user_features[feed_dict['user_id']][c] 
				# 			for c in self.corpus.user_feature_names]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				user_features_numeric=[]
				user_features_categorical=[]
				for c in self.corpus.user_feature_names:
					if (c.find(':numeric') != -1): # numeric
						user_features_numeric.append(self.corpus.user_features[feed_dict['user_id']][c])
					else: # categorical
						categorical_feature_list.append(c)
						user_features_categorical.append(self.corpus.user_features[feed_dict['user_id']][c])
				user_features_numeric = np.array(user_features_numeric).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				user_features_categorical = np.array(user_features_categorical).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				# print('for debug', user_features_numeric.shape, user_features_categorical.shape)
				All_context_numeric = user_features_numeric
				All_context_categorical = user_features_categorical
			if len(self.corpus.item_feature_names):
				# items_features = np.array([[self.corpus.item_features[iid][c] for c in self.corpus.item_feature_names] 
				# 							for iid in feed_dict['item_id'] ])
				item_features_numeric=[]
				item_features_categorical=[]
				for iid in feed_dict['item_id']:
					current_features_numeric = []
					current_features_categorical = []
					for c in self.corpus.item_feature_names:
						if (c.find(':numeric') != -1): # numeric
							current_features_numeric.append(self.corpus.item_features[iid][c])
						else: # categorical
							if c not in categorical_feature_list:
								categorical_feature_list.append(c)
							current_features_categorical.append(self.corpus.item_features[iid][c])
					item_features_numeric.append(current_features_numeric)
					item_features_categorical.append(current_features_categorical)
				item_features_numeric = np.array(item_features_numeric)
				item_features_categorical = np.array(item_features_categorical)
				# print('for debug', item_features_numeric.shape, item_features_categorical.shape)
				All_context_numeric = np.concatenate([All_context_numeric, item_features_numeric],axis=-1) if len(All_context_numeric) else item_features_numeric
				All_context_categorical = np.concatenate([All_context_categorical, item_features_categorical],axis=-1) if len(All_context_categorical) else item_features_categorical
			if len(self.corpus.context_feature_names):
				# context_features = np.array([self.data[f][index] for f in self.corpus.context_feature_names]).reshape(1,-1).repeat(
				# 			feed_dict['item_id'].shape[0],axis=0)
				context_features_numeric=[]
				context_features_categorical=[]
				immers_context_features=[]
				for f in self.corpus.context_feature_names:
					if (f.find(':numeric') != -1): # numeric
						context_features_numeric.append(self.data[f][index])
					elif f.find('c_behavior')!=-1 or f.find('c_session_order')!=-1:
						categorical_feature_list.append(f)
						immers_context_features.append(self.data[f][index])
					else: # categorical
						categorical_feature_list.append(f)
						context_features_categorical.append(self.data[f][index])
				context_features_numeric = np.array(context_features_numeric).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				context_features_categorical = np.array(context_features_categorical).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				immers_context_features_flatten = []
				for element in immers_context_features:
					if isinstance(element, list):
						immers_context_features_flatten.extend(element)
					else:
						immers_context_features_flatten.append(element)# print('for debug', context_features_numeric.shape,  context_features_categorical.shape)
				immers_context_features = np.array(immers_context_features_flatten).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				All_context_numeric = np.concatenate([All_context_numeric, context_features_numeric],axis=-1) if len(All_context_numeric) else context_features_numeric
				All_context_categorical = np.concatenate([All_context_categorical, context_features_categorical],axis=-1) if len(All_context_categorical) else context_features_categorical
				All_context_immers = immers_context_features
			id_names = []
			if self.include_id:
				user_id = np.array([feed_dict['user_id']]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				item_ids = feed_dict['item_id'].reshape(-1,1)
				All_context_categorical = np.concatenate([All_context_categorical, user_id, item_ids],axis=-1) if len(All_context_categorical) else np.concatenate([user_id,item_ids],axis=-1)
				id_names = ['user_id','item_id']
			# transfer into multi-hot embedding
			# change:
			base = 0
			# for i, feature in enumerate(categorical_feature_list+id_names):
			for i, feature in enumerate(id_names):
				# if feature.find('c_behavior')!=-1 or feature.find('c_session_order')!=-1:
				# 	continue
				All_context_categorical[:,i] += base
				base += self.corpus.feature_max_categorical[feature]
			feed_dict['context_mh'] = All_context_categorical # item num * feature num (will repeat 'item num' times for user and context features)
			feed_dict['context_numeric'] = All_context_numeric
			feed_dict['context_immers'] = All_context_immers		
			return feed_dict

class ContextSeqModel(ContextModel):
	reader='ContextSeqReader'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return ContextModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(SequentialModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]
			self.include_id = model.include_id
		
		def _convert_multihot(self, fname_list, data):
			base = 0
			for i, feature in enumerate(fname_list):
				data[:,i] += base
				# if feature.find(':numeric')!=-1:
				# 	base += self.corpus.feature_max_numeric[feature]
				base += self.corpus.feature_max_categorical[feature]
			return data

		def _get_feed_dict(self, index):
			# get item features, user features, and context features separately
			feed_dict = super()._get_feed_dict(index)
			# if self.corpus.include_source_domain:
			# 	feed_dict = self._get_source_dict(feed_dict) 
			if len(self.corpus.user_feature_names):
				user_fnames = self.corpus.user_feature_names
				user_features = [self.corpus.user_features[feed_dict['user_id']][c] for c in self.corpus.user_feature_names] 
				if self.include_id:
					user_fnames = ['user_id'] + user_fnames
					user_features = [feed_dict['user_id']] + user_features
				feed_dict['user_features'] = self._convert_multihot(user_fnames, np.array(user_features).reshape(1,-1))
			elif self.include_id:
				feed_dict['user_features'] = feed_dict['user_id'].reshape(-1,1)
			
			if len(self.corpus.item_feature_names):
				item_fnames = self.corpus.item_feature_names
				# items_features = np.array([[self.corpus.item_features[iid][c] for c in self.corpus.item_feature_names] for iid in feed_dict['item_id'] ])
				# items_features_history = np.array([[self.corpus.item_features[iid][c] for c in self.corpus.item_feature_names] for iid in feed_dict['history_items'] ])
				item_features, item_features_numeric = [], []
				history_item_features, history_item_features_numeric = [], []
				for iid in feed_dict['item_id']:
					current_features_numeric, current_features_categorical = [],[]
					for c in item_fnames:
						if (c.find(':numeric') != -1): # numeric
							current_features_numeric.append(self.corpus.item_features[iid][c])
						else:
							current_features_categorical.append(self.corpus.item_features[iid][c])
					item_features_numeric.append(current_features_numeric)
					item_features.append(current_features_categorical)
				for iid in feed_dict['history_items']:
					current_features_numeric, current_features_categorical = [],[]
					for c in item_fnames:
						if (c.find(':numeric') != -1): # numeric
							current_features_numeric.append(self.corpus.item_features[iid][c])
						else:
							current_features_categorical.append(self.corpus.item_features[iid][c])
					history_item_features_numeric.append(current_features_numeric)
					history_item_features.append(current_features_categorical)
				item_features = np.array(item_features)
				item_features_numeric = np.array(item_features_numeric)
				history_item_features = np.array(history_item_features)
				history_item_features_numeric = np.array(history_item_features_numeric)
				if self.include_id:
					item_fnames = ['item_id']+item_fnames
					item_ids = feed_dict['item_id'].reshape(-1,1)
					item_features = np.concatenate([item_ids, item_features],axis=1)
					his_item_ids = feed_dict['history_items'].reshape(-1,1)
					history_item_features = np.concatenate([his_item_ids,history_item_features],axis=1)
				feed_dict['item_features'] = self._convert_multihot(['item_id'], np.array(item_features))
				feed_dict['history_item_features'] = self._convert_multihot(['item_id'],np.array(history_item_features))
				feed_dict['item_features_numeric'] = item_features_numeric
				feed_dict['history_item_features_numeric'] = history_item_features_numeric
			elif self.include_id:
				feed_dict['item_features'] = feed_dict['item_id'].reshape(-1,1)
				feed_dict['history_item_features'] = feed_dict['history_items'].reshape(-1,1)

			if len(self.corpus.context_feature_names): 
				# context_features = np.array([self.data[f][index] for f in self.corpus.context_feature_names]).reshape(1,-1).repeat(
				# 			feed_dict['item_id'].shape[0],axis=0)
				# context_features_numeric=[]
				# context_features_categorical=[]
				immers_context_features=[]
				for f in self.corpus.context_feature_names:
					# if (f.find(':numeric') != -1): # numeric
					# 	context_features_numeric.append(self.data[f][index])
					if f.find('c_behavior')!=-1 or f.find('c_session_order')!=-1:
						immers_context_features.append(self.data[f][index])
					# else: # categorical
					# 	context_features_categorical.append(self.data[f][index])
				# context_features_numeric = np.array(context_features_numeric).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				# context_features_categorical = np.array(context_features_categorical).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				immers_context_features_flatten = []
				for element in immers_context_features:
					if isinstance(element, list) or isinstance(element,np.ndarray):
						immers_context_features_flatten.extend(list(element))
					else:
						immers_context_features_flatten.append(element)# print('for debug', context_features_numeric.shape,  context_features_categorical.shape)
				immers_context_features = np.array(immers_context_features_flatten).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				# feed_dict['context_features'] = self._convert_multihot(self.corpus.context_feature_names, context_features)
				feed_dict['context_features'] = immers_context_features
			return feed_dict
