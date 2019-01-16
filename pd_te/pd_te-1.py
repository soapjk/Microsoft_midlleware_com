import pandas as pd
import time
import numpy as np
col_value_dict = {'feature_name': ['a','b','c',np.nan], 'value': [2,3,1,np.nan]}
col_value_df = pd.DataFrame(col_value_dict)
col_value_df['feature_name']=col_value_df['feature_name'].astype('category')
print(col_value_df['feature_name'].cat.add_categories(['isnan']))

col_value_df.drop(['feature_name'],axis=1)
print(col_value_df['value'].values)
col_value_df.sort_values(by='value',inplace=True)
col_value_df.reset_index(inplace=True)
print(col_value_df)
