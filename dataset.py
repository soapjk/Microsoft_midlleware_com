import torch
from utils.DataPrepare import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        silent = []
        if mode == 'train':
            silent = ['HasDetections']
            origin_df, col_value_dict = data_nn_encode(filename='train.csv',silent=silent)
        else:
            origin_df, col_value_dict = data_nn_encode(filename='test.csv')
        self.columns = origin_df.columns
        self.embed_columns = self.columns-silent
        self.data = origin_df.as_matrix()
        self.col_value_dict = col_value_dict
        del origin_df

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item = self.data[index]
        dict(zip(self.columns,item))
        return dict(zip(self.columns,item))
