import logging
import pandas as pd
import os
import sys

from helpers.ContextReader import ContextReader

class ContextSeqReader(ContextReader):
	@staticmethod
	def parse_data_args(parser):
		parser.add_argument('--include_his_context',type=int, default=0,
								help='Whether include history context features.')
		return ContextReader.parse_data_args(parser)

	def __init__(self, args):
		super().__init__(args)
		self.include_his_context = args.include_his_context
		self._append_his_info()
        

	def _append_his_info(self):
		"""
		self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
		add the 'position' of each interaction in user_his to data_df
		"""
		logging.info('Appending history info...')
		if self.include_his_context:
			all_df_train = self.data_df['train'][['user_id', 'item_id', 'time', 'c_behavior_like','c_behavior_view','c_session_order']]
			all_df_dev_test = pd.concat([self.data_df[key][['user_id', 'item_id', 'time', 'c_behavior_like','c_behavior_view','c_session_order']] for key in ['dev', 'test']])
		else:
			all_df_train = self.data_df['train'][['user_id', 'item_id', 'time']]
			all_df_dev_test = pd.concat([self.data_df[key][['user_id', 'item_id', 'time']] for key in ['dev', 'test']])
		sort_df_train = all_df_train.sort_values(by=['time', 'user_id'], kind='mergesort')
		sort_df_dev_test = all_df_dev_test.sort_values(by=['time', 'user_id'], kind='mergesort')
		
		self.user_his = {'dev_test':{},'train':{}}  # store the already seen sequence of each user
		# for devtest
		position = list()
		if self.include_his_context:
			for uid, iid, t, c_behavior1, c_behavior2, order in zip(sort_df_dev_test['user_id'], sort_df_dev_test['item_id'], sort_df_dev_test['time'],
                                                           sort_df_dev_test['c_behavior_like'], sort_df_dev_test['c_behavior_view'], sort_df_dev_test['c_session_order']):
				if uid not in self.user_his['dev_test']:
					self.user_his['dev_test'][uid] = list()
				position.append(len(self.user_his['dev_test'][uid]))
				self.user_his['dev_test'][uid].append([iid, t, c_behavior1, c_behavior2, order])
			sort_df_dev_test = sort_df_dev_test.drop(columns = ['c_behavior_like', 'c_behavior_view', 'c_session_order'])
		else:
			for uid, iid, t in zip(sort_df_dev_test['user_id'], sort_df_dev_test['item_id'], sort_df_dev_test['time']):
				if uid not in self.user_his['dev_test']:
					self.user_his['dev_test'][uid] = list()
				position.append(len(self.user_his['dev_test'][uid]))
				self.user_his['dev_test'][uid].append((iid, t))
		sort_df_dev_test['position'] = position
		sort_df_dev_test['split']=1 # for dev_test
		# for train
		position = list()
		if self.include_his_context:
			for uid, iid, t, c_behavior1, c_behavior2, order in zip(sort_df_train['user_id'], sort_df_train['item_id'], sort_df_train['time'],
                                                           sort_df_train['c_behavior_like'], sort_df_train['c_behavior_view'], sort_df_train['c_session_order']):
				if uid not in self.user_his['train']:
					self.user_his['train'][uid] = list()
				position.append(len(self.user_his['train'][uid]))
				self.user_his['train'][uid].append([iid, t, c_behavior1, c_behavior2, order])
			sort_df_train = sort_df_train.drop(columns = ['c_behavior_like', 'c_behavior_view', 'c_session_order'])

		else:
			for uid, iid, t in zip(sort_df_train['user_id'], sort_df_train['item_id'], sort_df_train['time']):
				if uid not in self.user_his['train']:
					self.user_his['train'][uid] = list()
				position.append(len(self.user_his['train'][uid]))
				self.user_his['train'][uid].append((iid, t))
		sort_df_train['position'] = position
		sort_df_train['split']=0 # for train
		sort_df = pd.concat([sort_df_train,sort_df_dev_test], ignore_index=True)
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.merge(
				left=self.data_df[key], right=sort_df, how='left',
				on=['user_id', 'item_id', 'time'])
		del sort_df
