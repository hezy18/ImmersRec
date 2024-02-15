import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Formula_Model(nn.Module):
    def __init__(self, sequence_length, hidden_dim_seq, hidden_dim_last):
        super(Formula_Model, self).__init__()
        self.im_linear1 = nn.Linear(sequence_length,hidden_dim_seq)
        self.im_linear2 = nn.Linear(sequence_length,hidden_dim_seq)
        self.im_item_embedding = nn.Linear(3, 1)
        self.fc1 = nn.Linear(hidden_dim_seq*2 + 1, hidden_dim_last)
        self.fc2 = nn.Linear(hidden_dim_last, 1)
        self.im_params = nn.Parameter(torch.randn(3))

    def forward(self, t, sequence, item_feature):
        behavior_seq1 = sequence[:,0,:]
        behavior_seq2 = sequence[:,1,:]
        sequence1_score = self.im_linear1(behavior_seq1)
        sequence2_score = self.im_linear2(behavior_seq2)
        time_score = self.im_params[0] * t**2 + self.im_params[1] * t + self.im_params[2]
        item_represent = self.im_item_embedding(item_feature)
        # item_score1 = torch.mm(time_score.unsqueeze(0), item_represent)
        # print(item_score1.shape)
        item_score = torch.mul(time_score.unsqueeze(1), item_represent)
        combined_score = torch.cat((sequence1_score, sequence2_score, item_score), dim=1)
        hidden = torch.relu(self.fc1(combined_score))
        pred_immers = self.fc2(hidden).squeeze(-1)
        return pred_immers
    