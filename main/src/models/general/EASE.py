# -*- coding: UTF-8 -*-
# @Author  : Zhiyu He
# @Email   : hezy22@mails.tsinghua.edu.cn
"""EASE Model (Embarrassingly Shallow Autoencoders for Sparse Data)

Reference:
    "Embarrassingly Shallow Autoencoders for Sparse Data"
    Steck et al., WWW'2019.
    
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix
from models.BaseModel import GeneralModel

class EASE(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['lambda_reg']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--lambda_reg', type=float, default=1e3,
                          help='Regularization parameter for EASE')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.lambda_reg = args.lambda_reg
        self._define_params()
        
        self.n_users, self.n_items = corpus.n_users, corpus.n_items
        
        self.user_hist = {}
        for user_id, item_id in zip(corpus.data_df['train']['user_id'].values, 
                                    corpus.data_df['train']['item_id'].values):
            if user_id not in self.user_hist:
                self.user_hist[user_id] = set()
            self.user_hist[user_id].add(item_id)
        
        self._compute_B_matrix()

    def _define_params(self):
        pass 

    def _compute_B_matrix(self):
        rows, cols = [], []
        for user_id, items in self.user_hist.items():
            for item_id in items:
                rows.append(user_id)
                cols.append(item_id)
        data = np.ones(len(rows), dtype=np.float32)
        X = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        
        C = X.T.dot(X).toarray().astype(np.float32)
        C = torch.FloatTensor(C).to(self.device)
        
        reg = self.lambda_reg * torch.eye(self.n_items, device=self.device)
        C_reg = C + reg
        
        try:
            L = torch.linalg.cholesky(C_reg)
            inv_C_reg = torch.cholesky_inverse(L)
        except RuntimeError: 
            inv_C_reg = torch.pinverse(C_reg)
        
        B = inv_C_reg @ C
        B.fill_diagonal_(0.0)
        
        self.B_matrix = B

    def forward(self, feed_dict):
        user_ids = feed_dict['user_id'].long()  # [batch_size]
        item_ids = feed_dict['item_id'].long()  # [batch_size, n_candidates]
        
        user_hist = torch.zeros((len(user_ids), self.n_items), 
                            dtype=torch.float32, device=self.device)
        for i, uid in enumerate(user_ids.cpu().numpy()):
            if uid in self.user_hist:
                user_hist[i][list(self.user_hist[uid])] = 1.0
        
        scores = user_hist @ self.B_matrix
        predictions = torch.gather(scores, 1, item_ids)
        return {'prediction': predictions}