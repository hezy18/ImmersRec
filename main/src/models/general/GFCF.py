# -*- coding: UTF-8 -*-
# @Author  : Zhiyu He
# @Email   : hezy22@mails.tsinghua.edu.cn
""" GFCF (Graph Filtering for Recommendation)
Reference:
    "How powerfulis graph convolution for recommendation?"
    Yifei Shen et al., CIKM'2021.
CMD example:
    python main.py --model_name GFCF --alpha 0.5 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
from models.BaseModel import GeneralModel
import numpy as np
from scipy.sparse import csr_matrix, diags
import scipy.sparse.linalg as splinalg

class GFCF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['alpha']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--alpha', type=float, default=0.5, 
                          help='Graph filtering parameter')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.alpha = args.alpha
        self._define_params()
        
        self.n_users, self.n_items = corpus.n_users, corpus.n_items
        self.user_hist = {}  # {user_id: set(item_id)}
        for user_id, item_id in zip(corpus.data_df['train']['user_id'].values, 
                                    corpus.data_df['train']['item_id'].values):
            if user_id not in self.user_hist:
                self.user_hist[user_id] = set()
            self.user_hist[user_id].add(item_id)
        self._build_interaction_matrix()
        
        self._compute_filtered_matrix()
        import pdb; pdb.set_trace()

    def _define_params(self):
        pass 

    def _build_interaction_matrix(self):
        rows, cols = [], []
        for user_id, items in self.user_hist.items():
            for item_id in items:
                rows.append(user_id)
                cols.append(item_id)
        data = np.ones(len(rows), dtype=np.float32)
        self.inter_matrix = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def _compute_filtered_matrix(self):
        user_degree = np.array(self.inter_matrix.sum(axis=1)).flatten() 
        item_degree = np.array(self.inter_matrix.sum(axis=0)).flatten()
        
        user_degree_diag = diags(user_degree)
        item_degree_diag = diags(item_degree)
        
        L_user = user_degree_diag - self.inter_matrix.dot(self.inter_matrix.T)
        L_item = item_degree_diag - self.inter_matrix.T.dot(self.inter_matrix)
        
        L_user = diags(np.ones(self.n_users)) - self.alpha * L_user
        L_item = diags(np.ones(self.n_items)) - self.alpha * L_item
        # filtered_matrix = (np.eye(self.n_users) - self.alpha * L_user).dot(self.inter_matrix).dot(np.eye(self.n_items) - self.alpha * L_item)
        
        # self.filtered_matrix = torch.from_numpy(filtered_matrix.A).float().to(self.device)
        
        # from scipy.sparse.linalg import cg

        # filtered_matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        
        # for j in range(self.n_items):
        #     b = self.inter_matrix[:, j].toarray().flatten()
        #     x, _ = cg(L_user, b)  # 求解 A_user @ x = b
        #     filtered_matrix[:, j] = x

        # for i in range(self.n_users):
        #     b = filtered_matrix[i, :]
        #     y, _ = cg(L_item, b)  
        #     filtered_matrix[i, :] = y
        # R = self.inter_matrix.toarray()

        # x, _ = cg(L_user, R, M=None, tol=1e-5)  # 求解 A_user @ x = R

        # filtered_matrix = x @ cg(L_item, np.eye(self.n_items))[0]

        # self.filtered_matrix = torch.from_numpy(filtered_matrix).float().to(self.device)
        L_user = L_user.tocsr()
        L_item_csr = L_item.tocsr()
        R_csr = self.inter_matrix.tocsr()
        filtered_matrix_csr = splinalg.inv(L_user) @ R_csr @ splinalg.inv(L_item)

        self.filtered_matrix = torch.from_numpy(filtered_matrix_csr.toarray()).float().to(self.device)

    def forward(self, feed_dict):
        user_ids = feed_dict['user_id'].long()  # [batch_size]
        item_ids = feed_dict['item_id'].long()  # [batch_size, n_candidates]
        
        user_hist = torch.zeros((len(user_ids), self.n_items), dtype=torch.float32, device=self.device)
        for i, user_id in enumerate(user_ids.cpu().numpy()):
            if user_id in self.user_hist:
                user_hist[i][list(self.user_hist[user_id])] = 1.0
        
        scores = user_hist @ self.filtered_matrix  # [batch_size, n_items]
        
        predictions = torch.gather(scores, 1, item_ids)  # [batch_size, n_candidates]
        return {'prediction': predictions}