import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train'):
        super(Dataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return item
