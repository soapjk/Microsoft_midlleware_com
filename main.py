import torch
from utils.DataPrepare import *
from sklearn.model_selection import train_test_split
import numpy as np
from mymodel import *
import torch.nn as nn
import time
from sklearn.metrics import roc_auc_score

silent = ['HasDetections']
train_input = pd.read_csv('../Data/train.csv', dtype=dtypes)
test_input = pd.read_csv('../Data/test.csv', dtype=dtypes)
train_origin_df, col_value_dict = data_nn_encode(filename='train.csv', silent=silent, base_data_path='../Data/')
test_origin_df, col_value_dict = data_nn_encode(filename='test.csv', base_data_path='../Data/')
col_value_dict = col_value_dict
columns = train_origin_df.columns - silent
X_train, X_val, y_train, y_val = train_test_split(train_origin_df.drop(['HasDetections'], axis=1),
                                                  train_origin_df['HasDetections'], test_size=0.33, random_state=2019)
torch_X_train = torch.FloatTensor(train_origin_df.values)
torch_X_val = torch.FloatTensor(X_val.values)
torch_y_train = torch.FloatTensor(y_train.values.astype(np.int32))
torch_y_val = torch.FloatTensor(y_val.values.astype(np.int32))
torch_test = torch.FloatTensor(test_origin_df.values)

model = Mymodel(columns, col_value_dict)
loss_fn = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())
batch_size = 32
torch_train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
train_loader = torch.utils.data.DataLoader(torch_train, batch_size=batch_size, shuffle=True)
torch_val = torch.utils.data.TensorDataset(torch_X_val, torch_y_val)
valid_loader = torch.utils.data.DataLoader(torch_val, batch_size=batch_size, shuffle=False)
n_epochs = 1000
train_preds = np.zeros((torch_X_train.size(0)))
valid_preds = np.zeros((torch_X_val.size(0)))
for epoch in range(n_epochs):
    start_time = time.time()
    avg_loss = 0.
    # set the module in training mode.
    model.train()

    for x_batch, y_batch in tqdm(train_loader, disable=True):
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    model.eval()

    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_val_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_val_pred, y_batch).item() / len(valid_loader)

        valid_preds[i * batch_size:(i + 1) * batch_size] = y_val_pred.cpu().numpy()[:, 0]
    elapsed_time = time.time() - start_time
    print('\nEpoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
    print('AUC_VAL{} '.format(roc_auc_score(torch_y_val.cpu(), valid_preds).round(3)))
