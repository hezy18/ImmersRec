import logging
import numpy as np
import pandas as pd
import os
import sys

from helpers.BaseReader import BaseReader

class ContextReader(BaseReader):
	@staticmethod
	def parse_data_args(parser):
		parser.add_argument('--include_item_features',type=int, default=0,
								help='Whether include item context features.')
		parser.add_argument('--include_user_features',type=int, default=0,
								help='Whether include user context features.')
		parser.add_argument('--include_context_features',type=int, default=0,
								help='Whether include dynamic context features.')
		parser.add_argument('--include_immersion', type=int, default=0,
								help='Whether contain immersion.')
		parser.add_argument('--include_source_domain', type=int, default=0,
								help='Whether contain source data for domain adaptation.')
		parser.add_argument('--DANN', type=int, default=0)
		return BaseReader.parse_data_args(parser)

	def __init__(self, args):
		super().__init__(args)
		self.include_item_features = args.include_item_features
		self.include_user_features = args.include_user_features
		self.include_context_features = args.include_context_features
		self.include_immersion = args.include_immersion
		self.include_source_domain = args.include_source_domain
		self.DANN = args.DANN
		self._load_ui_metadata()
		self._define_context()
		if self.include_source_domain:
			self._load_source_metadata()
			self._add_source()

	def _load_source_metadata(self):
		self.source_data_df = None
		source_data_path = os.path.join(self.prefix, self.dataset, 'source_data.csv')
		if os.path.exists(source_data_path) and self.include_source_domain:
			self.source_data_df = pd.read_csv(source_data_path,sep=self.sep)
			self.source_names = sorted([c for c in self.source_data_df.columns if c[:2]=='c_']) + sorted([c for c in self.source_data_df.columns if c[:2]=='i_'])
			self.source_names.append('rating_immersion')
		else:
			self.source_names=[]

	def _add_source(self):
		logging.info('Collect source domain data...')
		All_source_data = []
		if self.source_data_df is not None and self.include_source_domain:
			for f in self.source_names:
				if f=='rating_immersion':
					# normalized
					self.source_data_df[f] = (self.source_data_df[f] - self.source_data_df[f].min()) / (self.source_data_df[f].max() - self.source_data_df[f].min())
				elif (f.find('behavior')!=-1):
					self.source_data_df[f] = self.source_data_df[f].apply(eval)
			for f in self.source_names:
				if f.find('behavior')!=-1:
					All_source_data.append(np.array(self.source_data_df[f].tolist()))
				else:
					All_source_data.append(self.source_data_df[f].values.reshape(-1,1))
			All_source_data = np.concatenate(All_source_data, axis=1)
			self.source_data = All_source_data
			logging.info('# Source domain data shape: %d, %d' % (self.source_data.shape[0], self.source_data.shape[1]))
			del self.source_data_df

	def _load_ui_metadata(self):
		self.item_meta_df, self.user_meta_df = None, None
		item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
		user_meta_path = os.path.join(self.prefix, self.dataset, 'user_meta.csv')
		if os.path.exists(item_meta_path) and self.include_item_features:
			self.item_meta_df = pd.read_csv(item_meta_path,sep=self.sep)
			self.item_feature_names = sorted([c for c in self.item_meta_df.columns if c[:2]=='i_'])
		else:
			self.item_feature_names = []
		if os.path.exists(user_meta_path) and self.include_user_features:
			self.user_meta_df = pd.read_csv(user_meta_path,sep=self.sep)		
			self.user_feature_names = sorted([c for c in self.user_meta_df.columns if c[:2]=='u_'])
		else:
			self.user_feature_names = []
		if self.include_context_features:
			self.context_feature_names = sorted([c for c in self.data_df['train'].columns if c[:2]=='c_'])
		else:
			self.context_feature_names = []

	def _define_context(self):
		logging.info('Collect context features...')
		id_columns = ['user_id','item_id']
		self.context_feature_df = dict()
		self.item_features, self.user_features = None, None # dict
		self.feature_max_categorical = dict()
		self.feature_max_numeric = dict()
		for key in ['train', 'dev', 'test']:
			logging.info('Loading context for %s set...'%(key))
			ids_df = self.data_df[key][id_columns]
			for f in id_columns:
				self.feature_max_categorical[f] = max(self.feature_max_categorical.get(f,0), int(ids_df[f].max())+1)
			# include context features
			if self.include_context_features and len(self.context_feature_names):
				for f in self.context_feature_names:
					if (f.find(':numeric') != -1): # numeric
						# normalized
						self.data_df[key][f] = (self.data_df[key][f] - self.data_df[key][f].min()) / (self.data_df[key][f].max() - self.data_df[key][f].min())
						self.feature_max_numeric[f] = max(self.feature_max_numeric.get(f,0), int(self.data_df[key][f].max()) + 1 )
					elif (f.find('c_session')!=-1):
						self.data_df[key][f] = (self.data_df[key][f] - self.data_df[key][f].min()) / (self.data_df[key][f].max() - self.data_df[key][f].min())
						self.feature_max_categorical[f] = 0
					else: #categorical
						if (f.find('c_behavior')!=-1) or (f.find('c_session')!=-1):
							self.feature_max_categorical[f] = 0
						else:
							self.feature_max_categorical[f] = max(self.feature_max_categorical.get(f,0), int(self.data_df[key][f].max()) + 1 )
				context_df = self.data_df[key][id_columns+['time']+self.context_feature_names]
				self.context_feature_df[key] = context_df
				logging.info('#Context Feauture: %d'%(context_df.shape[1]-3))
		# include item features
		if self.item_meta_df is not None and self.include_item_features:
			for f in self.item_feature_names:
				if (f.find(':numeric') != -1): # numeric
					# normalized
					self.item_meta_df[f] = (self.item_meta_df[f] - self.item_meta_df[f].min()) / (self.item_meta_df[f].max() - self.item_meta_df[f].min())
					self.feature_max_numeric[f] = max(self.feature_max_numeric.get(f,0), int(self.item_meta_df[f].max()) + 1 )
				else: # categorical
					self.feature_max_categorical[f] = max( self.feature_max_categorical.get(f,0), int(self.item_meta_df[f].max())+1 )
			item_df = self.item_meta_df[['item_id']+self.item_feature_names]
			self.item_features = item_df.set_index('item_id').to_dict(orient='index')
			if 'i_category_c' in item_df.columns:
				self.category2items = item_df.groupby('i_category_c').item_id.apply(lambda x:list(x)).to_dict()
				item2category = item_df.set_index('item_id')['i_category_c'].to_dict()
				category_all = np.array([ item2category[iid] for iid in self.data_df['train']['item_id']])
				self.category_frequency = np.zeros(self.feature_max_categorical['i_category_c'])
				for c in category_all:
					self.category_frequency[c] += 1
			logging.info('# Item Features: %d'%(item_df.shape[1]))
		# include user features
		if self.user_meta_df is not None and self.include_user_features:
			for f in self.user_feature_names:
				if (f.find(':numeric') != -1): # numeric
					# normalized
					self.user_meta_df[f] = (self.user_meta_df[f] - self.user_meta_df[f].min()) / (self.user_meta_df[f].max() - self.user_meta_df[f].min())
					self.feature_max_numeric[f] = max(self.feature_max_numeric.get(f,0), int(self.user_meta_df.max()) + 1 )
				else: # categorical
					self.feature_max_categorical[f] = max( self.feature_max_categorical.get(f,0), int(self.user_meta_df[f].max())+1 )
			user_df = self.user_meta_df[['user_id']+self.user_feature_names].set_index('user_id')
			self.user_features = user_df.to_dict(orient='index')
			logging.info('# User Features: %d'%(user_df.shape[1]))

