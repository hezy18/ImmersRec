# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel
import re

def contains_rank_g(metrics):
    if 'Rank' in metrics or 'RANK' in metrics:
        return 'RANK@2'
    pattern = r'^RANK@\d+$'
    for item in metrics:
        if re.match(pattern, item):
            return item
    return None

class BaseRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--epoch', type=int, default=200,
							help='Number of epochs.')
		parser.add_argument('--check_epoch', type=int, default=1,
							help='Check some tensors every check_epoch.')
		parser.add_argument('--test_epoch', type=int, default=-1,
							help='Print test results every test_epoch (-1 means no print).')
		parser.add_argument('--early_stop', type=int, default=10,
							help='The number of epochs when dev results drop continuously.')
		parser.add_argument('--lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--l2', type=float, default=0,
							help='Weight decay in optimizer.')
		parser.add_argument('--batch_size', type=int, default=256,
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=256,
							help='Batch size during testing.')
		parser.add_argument('--optimizer', type=str, default='Adam',
							help='optimizer: SGD, Adam, Adagrad, Adadelta')
		parser.add_argument('--num_workers', type=int, default=1,
							help='Number of processors when prepare batches in DataLoader')
		parser.add_argument('--pin_memory', type=int, default=0,
							help='pin_memory in DataLoader')
		parser.add_argument('--topk', type=str, default='20,50',
							help='The number of items recommended to each user.')
		parser.add_argument('--metric', type=str, default='NDCG,HR,AUC',
							help='metric: NDCG, HR, AUC, Rank@G')
		parser.add_argument('--multiGPU', type=int, default=0, help='whether to use multiGPU')
		return parser

	@staticmethod
	def evaluate_method(self, predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
		"""
		:param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()
		sort_idx = (-predictions).argsort(axis=1)
		gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1 # (29008,)
		n_samples, n_candidates = predictions.shape
		ground_truth = np.zeros((n_samples, n_candidates), dtype=int)
		ground_truth[:, 0] = 1
		if 'AUC' in metrics:
			auc_values = np.zeros(n_samples)
			for i in range(n_samples):
				pos_scores = predictions[i][ground_truth[i] == 1]
				neg_scores = predictions[i][ground_truth[i] == 0]
				pos_samples = pos_scores.shape[0]
				neg_samples = neg_scores.shape[0]
				auc_values[i] = np.sum(pos_scores > neg_scores[:, np.newaxis]) / (pos_samples * neg_samples)
			evaluations['AUC'] = auc_values.mean()
		
		rank_g = contains_rank_g(metrics)
		if rank_g != None:
			n_groups = int(rank_g.split('@')[-1])
			gt_rank = gt_rank.reshape(gt_rank.shape[0] // n_groups, n_groups) 
			org_gt_ranks = np.sum(gt_rank, axis=1) - (n_groups - 1)
			for k in topk:
				org_hit = (org_gt_ranks <= k)
				for metric in metrics:
					evaluations[f'org_HR@{k}'] = org_hit.mean()
					evaluations[f'org_NDCG@{k}'] = (org_hit / np.log2(org_gt_ranks + 1)).mean()
		print(rank_g)
		print(evaluations)

		for k in topk:
			hit = (gt_rank <= k)
			for metric in metrics:
				key = '{}@{}'.format(metric, k)
				if metric == 'HR':
					evaluations[key] = hit.mean()
				elif metric == 'NDCG':
					evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
				elif metric == 'AUC':
					continue
				elif 'RANK' in metric:
					continue
				else:
					raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		return evaluations

	def __init__(self, args):
		self.train_models = args.train
		self.epoch = args.epoch
		self.check_epoch = args.check_epoch
		self.test_epoch = args.test_epoch
		self.early_stop = args.early_stop
		self.learning_rate = args.lr
		self.batch_size = args.batch_size
		self.eval_batch_size = args.eval_batch_size
		self.l2 = args.l2
		self.optimizer_name = args.optimizer
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.topk.split(',')]
		self.metrics = [m.strip().upper() for m in args.metric.split(',')]
		self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric

		self.time = None  # will store [start_time, last_step_time]
		self.multiGPU = args.multiGPU

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

	def _build_optimizer(self, model):
		logging.info('Optimizer: ' + self.optimizer_name)
		optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
			model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
		return optimizer

	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)
		try:
			for epoch in range(self.epoch):
				# Fit
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				loss = self.fit(data_dict['train'], epoch=epoch + 1)
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = self._check_time()

				# Observe selected tensors
				if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
					utils.check(model.check_list)

				# Record dev results
				# dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
				dev_result = self.evaluate(data_dict['dev'], self.topk, self.metrics)
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[self.main_metric])
				logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
					epoch + 1, loss, training_time, utils.format_metric(dev_result))
				# logger_loss.info('Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
				# 	epoch + 1, loss, training_time, utils.format_metric(dev_result)))

				# Test
				if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
					# test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
					test_result = self.evaluate(data_dict['test'], self.topk, self.metrics)
					logging_str += ' test=({})'.format(utils.format_metric(test_result))
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)

				# Save model and early stop
				if max(main_metric_results) == main_metric_results[-1] or \
						(hasattr(model, 'stage') and model.stage == 1):
					model.save_model()
					logging_str += ' *'
				logging.info(logging_str)

				if self.early_stop > 0 and self.eval_termination(main_metric_results):
					logging.info("Early stop at %d based on dev result." % (epoch + 1))
					break
		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)

		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model()

	def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		dataset.actions_before_epoch()  # must sample before multi thread start

		model.train()
		loss_lst = list()
		if self.multiGPU:
			dl = DataLoader(dataset, batch_size=self.args.batch_size, sampler=DistributedSampler(dataset))
		else:
			dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		i = 0
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)
			model.optimizer.zero_grad()
			batch['batch']=i
			batch['epoch']=epoch
			batch['n_epoch']=self.epoch
			batch['len_dataloader']=len(dl)
			out_dict = model(batch)
			loss = model.loss(out_dict)
			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
		return np.mean(loss_lst).item()

	def eval_termination(self, criterion: List[float]) -> bool:
		if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
			return True
		elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False

	def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""

		predictions = self.predict(dataset)
		return self.evaluate_method(self, predictions, topks, metrics)

	def predict(self, dataset: BaseModel.Dataset) -> np.ndarray:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.model.eval()
		predictions = list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			else:
				prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			predictions.extend(prediction.cpu().data.numpy())
		predictions = np.array(predictions)

		if dataset.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf

		if not self.train_models:
			now_time = int(time())
			logging.info("Saving predictions to %d.npy"%(now_time))
			np.save( '%d.npy'%(now_time) ,predictions)
		return predictions

	def print_res(self, dataset: BaseModel.Dataset) -> str:
		"""
		Construct the final result string before/after training
		:return: test result string
		"""
		result_dict = self.evaluate(dataset, self.topk, self.metrics)
		res_str = '(' + utils.format_metric(result_dict) + ')'
		return res_str
