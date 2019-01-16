import torch.nn as nn
import torch

class Mymodel(nn.Module):
    def __init__(self, embed_columns, col_value_dict):
        self.embed_list = nn.ModuleList()
        for col in embed_columns:
            self.embed_list.append(nn.Embedding(col_value_dict[col], 250))

        self.ff = nn.Sequential(
            nn.Linear(len(embed_columns)*250, 128),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self,X):
        emb_indices = X.long()
        emb_outs = []
        for i, emb_layer in enumerate(self.emb_layers):
            emb_out = emb_layer(emb_indices[:, i])
            emb_out = self.dropout(emb_out)
            emb_outs.append(emb_out)

        embs = torch.cat(emb_outs, dim=1)
        out = self.ff(embs)
        return out