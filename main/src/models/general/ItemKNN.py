# -*- coding: UTF-8 -*-
# @Author  : Zhiyu He
# @Email   : hezy22@mails.tsinghua.edu.cn
""" ItemKNN with Shrinkage
Reference:
    "Item-based collaborative filtering recommendation algorithms"
    Sarwar et al., WWW'2001. (with shrinkage)
CMD example:
    python main.py --model_name ItemKNN --lambda_shrink 100 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
from models.BaseModel import GeneralModel
import numpy as np
from scipy.sparse import csr_matrix

class ItemKNN(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['lambda_shrink']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--lambda_shrink', type=float, default=100, 
                          help='Shrinkage regularization')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.lambda_shrink = args.lambda_shrink
        self._define_params()
        
        self.n_users, self.n_items = corpus.n_users, corpus.n_items
        
        self.user_hist = {}  # {user_id: set(item_id)}
        for user_id, item_id in zip(corpus.data_df['train']['user_id'].values, 
                                    corpus.data_df['train']['item_id'].values):
            if user_id not in self.user_hist:
                self.user_hist[user_id] = set()
            self.user_hist[user_id].add(item_id)
        
        self._compute_similarity_KNN()

    def _define_params(self):
        pass 

    def _compute_similarity_base(self):
        X = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for user_id, items in self.user_hist.items():
            for item_id in items:
                X[user_id][item_id] = 1.0
        
        X = torch.FloatTensor(X).to(self.device)
        norm = torch.norm(X, p=2, dim=0)
        X_norm = X / (norm + 1e-8)
        sim = X_norm.T @ X_norm
        sim = sim / (norm.view(1, -1) * norm.view(-1, 1) + self.lambda_shrink)
        sim.fill_diagonal_(0)
        self.sim_matrix = sim
    
    def _compute_similarity_sparse(self): 
        rows, cols = [], []
        for user_id, items in self.user_hist.items():
            for item_id in items:
                rows.append(user_id)
                cols.append(item_id)
        data = np.ones(len(rows), dtype=np.float32)
        X = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        X = self._csr_to_torch_sparse(X).to(self.device)
        
        norm = torch.sqrt(torch.sparse.sum(X.pow(2), dim=0).to_dense())  # [n_items]
        norm = norm + 1e-8
        
        norm_diag = torch.diag(1.0 / norm)  # [n_items, n_items]
        norm_diag = norm_diag.to_sparse() 
        
        # 归一化：X_norm = X @ norm_diag
        X_norm = torch.sparse.mm(X, norm_diag)
        
        sim = torch.sparse.mm(X_norm.T, X_norm)  # [n_items, n_items]
        sim = sim.to_dense() 
        
        sim = sim / (norm.view(1, -1) * norm.view(-1, 1) + self.lambda_shrink)
        sim.fill_diagonal_(0)
        self.sim_matrix = sim
        
    def _csr_to_torch_sparse(self, csr_mat):
        coo = csr_mat.tocoo()
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))
        
    def _compute_similarity_block(self):
        block_size = 1000 
        n_blocks = (self.n_items + block_size - 1) // block_size
    
        sim = torch.zeros((self.n_items, self.n_items), device=self.device)
        
        for i in range(n_blocks):
            start_i = i * block_size
            end_i = min((i + 1) * block_size, self.n_items)
            
            for j in range(n_blocks):
                start_j = j * block_size
                end_j = min((j + 1) * block_size, self.n_items)
                
                X_block = torch.FloatTensor(self.inter_mat[:, start_j:end_j].toarray()).to(self.device)
                norm_block = torch.norm(X_block, p=2, dim=0)
                X_norm_block = X_block / (norm_block + 1e-8)
                
                sim_block = X_norm_block.T @ X_norm_block
                sim[start_i:end_i, start_j:end_j] = sim_block
        
        sim = sim / (norm.view(1, -1) * norm.view(-1, 1) + self.lambda_shrink)
        sim.fill_diagonal_(0)
        self.sim_matrix = sim

    def _compute_similarity_KNN(self):
        rows, cols = [], []
        for user_id, items in self.user_hist.items():
            for item_id in items:
                rows.append(user_id)
                cols.append(item_id)
        data = np.ones(len(rows), dtype=np.float32)
        X = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        
        norm = np.sqrt(X.power(2).sum(axis=0).A1)  # [n_items]
        norm = norm + 1e-8
        
        # 归一化：X_norm = X / norm
        X_norm = X.multiply(1.0 / norm)
        
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=100, metric='cosine')
        knn.fit(X_norm.T)
        distances, indices = knn.kneighbors(X_norm.T)
        
        from scipy.sparse import coo_matrix
        rows, cols, data = [], [], []
        for i in range(self.n_items):
            for j, dist in zip(indices[i], distances[i]):
                rows.append(i)
                cols.append(j)
                data.append(1.0 - dist)
        sim = coo_matrix((data, (rows, cols)), shape=(self.n_items, self.n_items))
        
        sim = sim / (norm.reshape(1, -1) * norm.reshape(-1, 1) + self.lambda_shrink)
        np.fill_diagonal(sim.A, 0)
        
        self.sim_matrix = torch.from_numpy(sim.A).to(torch.float32).to(self.device)

    def forward(self, feed_dict):
        user_ids = feed_dict['user_id'].long() # [batch_size]
        item_ids = feed_dict['item_id'].long() # [batch_size, n_candidates]
        
        user_hist = torch.zeros((len(user_ids), self.n_items), dtype=torch.float32, device=self.device)
        for i, user_id in enumerate(user_ids.cpu().numpy()):
            if user_id in self.user_hist:
                user_hist[i][list(self.user_hist[user_id])] = 1.0
        
        scores = user_hist @ self.sim_matrix  # [batch_size, n_items]
        
        predictions = torch.gather(scores, 1, item_ids)  # [batch_size, n_candidates]
        return {'prediction': predictions}