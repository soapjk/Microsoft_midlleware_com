import pandas as pd
from DataAnalysis import *
import os
from tqdm import tqdm


def data_preprocess(full_path_list,base_data_path='../../Data/',encoded_dict_name='feature_id_dict.csv',dtypes=None):
    """
    :param full_path_list: list of str,path of all data files,include  train,test,and vali
    :param base_data_path: directory to save generated files
    :param encoded_dict_name: name of encoded value's dict file,default 'feature_id_dict.csv'
    :param dtypes: dtyeps of every feature
    :return: dict object of encoded feature
    """
    dict_path=base_data_path + encoded_dict_name
    if not os.path.exists(base_data_path+'feature_value_num.csv'):
        df_list = []
        for one_path in full_path_list:
            temp_df = pd.read_csv(one_path, dtype=dtypes)
            df_list.append(temp_df)

        retained_columns = numerical_columns + categorical_columns
        retained_columns.remove('HasDetections')
        df_all = pd.concat(df_list, axis=0)
        columns = df_all.columns
        embed_cols = []
        len_embed_cols = []
        feature_id_dict = {}
        col_value_dict = {'feature_name': [], 'value_num': []}
        for c in tqdm(columns):
            if df_all[c].nunique()<500:
                embed_cols.append(c)
                len_embed_cols.append(df_all[c].nunique())
                feature_id_dict[c] = list(set(df_all[c].values))
                col_value_dict['feature_name'].append(c)
                col_value_dict['value_num'].append(df_all[c].nunique())
                print(c + ': %d values' % df_all[c].nunique())  # look at value counts to know the embedding dimensions
        print('\n Number of embed features :', len(embed_cols))
        col_value_df = pd.DataFrame(col_value_dict)
        col_value_df.sort_values(by='value', inplace=True)
        col_value_df.to_csv(base_data_path+'feature_value_num.csv', index=None, encoding='utf-8')
        feature_id_dict.pop('MachineIdentifier')
        f = open(dict_path, 'w', encoding='utf-8')
        f.write(str(feature_id_dict))
        f.close()
    else:
        f = open(dict_path, 'r', encoding='utf-8')
        feature_id_dict = eval(f.readline())
        f.close()
    return feature_id_dict,col_value_dict


def data_nn_encode(filename='train.csv', base_data_path='../../Data/', full_path_list=['train.csv', 'test.csv'], drop_list=None,silent=[]):
    """
    :param filename:
    :param full_path_list:
    :param drop_list:
    :param silent:
    :return:
"""
    for i in range(len(full_path_list)):
        full_path_list[i]=base_data_path+full_path_list[i]
    encoded_path = base_data_path + filename + '_encoded.csv'
    feature_id_dict, col_value_dict = data_preprocess(full_path_list=full_path_list)

    if not os.path.exists(encoded_path):
        origin_df = pd.read_csv(base_data_path+filename)
        if drop_list:
            origin_df.drop(drop_list, axis=1)

        def map_func(lis):
            return feature_id_dict[lis].index(lis)
        need_code_feature=origin_df.columns-silent
        for col in need_code_feature:
            origin_df[col] = origin_df[col].apply(map_func)
        origin_df.tocsv(encoded_path, index=None, encoding='utf-8')
    else:
        origin_df = pd.read_csv(encoded_path)
    return origin_df, col_value_dict


def main():
    train = data_nn_encode(drop_list=['MachineIdentifier'])


if __name__ == '__main__':
    main()
