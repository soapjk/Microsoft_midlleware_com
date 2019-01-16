import lightgbm as lgb
from DataAnalysis import *
train = pd.read_csv('../Data/train.csv', dtype=dtypes)
test = pd.read_csv('../Data/test.csv', dtype=dtypes)
train['flag'] = 0
test['flag'] = 1
data_all = pd.concat([train,test])
