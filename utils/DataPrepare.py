import pandas as pd
from DataAnalysis import *
import os


def Data_preprocess():
    if not os.path.exists('../Data/feature_value_num.csv'):
        train = pd.read_csv('../Data/train.csv', dtype=dtypes)
        test = pd.read_csv('../Data/test.csv', dtype=dtypes)
        retained_columns = numerical_columns + categorical_columns
        retained_columns.remove('HasDetections')
        df_all = pd.concat((train, test), axis=0)
        columns = df_all.columns
        embed_cols = []
        len_embed_cols = []
        feature_id_dict = {}
        col_value_dict = {'feature_name': [], 'value_num': []}
        for c in tqdm(columns):
            embed_cols.append(c)
            len_embed_cols.append(df_all[c].nunique())
            feature_id_dict[c] = list(set(df_all[c].values))
            col_value_dict['feature_name'].append(c)
            col_value_dict['value_num'].append(df_all[c].nunique())
            print(c + ': %d values' % df_all[c].nunique())  # look at value counts to know the embedding dimensions
        print('\n Number of embed features :', len(embed_cols))
        col_value_df = pd.DataFrame(col_value_dict)
        col_value_df.sort_values(by='value', inplace=True)
        col_value_df.to_csv('../Data/feature_value_num.csv', index=None, encoding='utf-8')
        feature_id_dict.pop('MachineIdentifier')
        f = open('../Data/feature_id_dict.csv', 'w', encoding='utf-8')
        f.write(str(feature_id_dict))
        f.close()

