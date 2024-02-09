# -*- coding: UTF-8 -*-
'''
Jiayu Li 2023.05.20
'''

import logging
import pandas as pd
import os
import sys

from helpers.ContextReader import ContextReader

class ContextSeqReader(ContextReader):
	
	def __init__(self, args):
		super().__init__(args)
		self._append_his_info()

	def _append_his_info(self):
		"""
		self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
		add the 'position' of each interaction in user_his to data_df
		"""
		logging.info('Appending history info...')
		sort_df_train = self.all_df_train.sort_values(by=['time', 'user_id'], kind='mergesort')
		sort_df_dev_test = self.all_df_dev_test.sort_values(by=['time', 'user_id'], kind='mergesort')
		position = list()
		self.user_his = {'dev_test':{},'train':{}}  # store the already seen sequence of each user
		for uid, iid, t in zip(sort_df_dev_test['user_id'], sort_df_dev_test['item_id'], sort_df_dev_test['time']):
			if uid not in self.user_his['dev_test']:
				self.user_his['dev_test'][uid] = list()
			position.append(len(self.user_his['dev_test'][uid]))
			self.user_his['dev_test'][uid].append((iid, t))
		sort_df_dev_test['position'] = position
		sort_df_dev_test['split']=1 # for dev_test
		position = list()
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