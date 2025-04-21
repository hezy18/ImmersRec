# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils


class BaseReader(object):
    extra_log_args = []
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='MIND',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--include_immersion', type=int, default=0,
								help='Whether contain immersion.')
        parser.add_argument('--sample_rate', type=float, default=1.0)
        parser.add_argument('--woIndicator', type=int, default=0)
        parser.add_argument('--DANN', type=int, default=0)
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)
        self.include_immersion = args.include_immersion
        self.DANN = args.DANN

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        # self.all_df_train = self.data_df['train'][['user_id', 'item_id', 'time']]
        # self.all_df_dev_test = pd.concat([self.data_df[key][['user_id', 'item_id', 'time']] for key in ['dev', 'test']])
        all_df = pd.concat([self.data_df[key][['user_id', 'item_id', 'time']] for key in ['train', 'dev', 'test']])
        # # self.all_df = pd.concat([self.all_df_train, self.all_df_dev_test], ignore_index=True)
        self.n_users, self.n_items = all_df['user_id'].max() + 1, all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(all_df)))
        
