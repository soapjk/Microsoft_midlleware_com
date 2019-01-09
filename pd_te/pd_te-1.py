import pandas as pd
import time
print(time.time())
col_value_dict = {'feature_name': ['a','b','c'], 'value': [2,3,1]}

col_value_df = pd.DataFrame(col_value_dict)
print(col_value_df['value'].values)
col_value_df.sort_values(by='value',inplace=True)
col_value_df.reset_index(inplace=True)
print(col_value_df)
