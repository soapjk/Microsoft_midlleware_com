import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from feature_get import *
import gc

data_base_path = '../Data/'
use_data_file = 0
TARGET_INDEX = 'MachineIdentifier'
TARGET = 'HasDetections'
params_dict = dict({
    'objective': ['binary', 'binary'],
    'boosting_type': ['gbdt', 'gbdt', 'gbdt', 'random_forest', 'random_forest'],
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
    'num_leaves': [31, 25, 40, 100, 160],
    'max_depth': [6, 20, 11, -1, 40],
    "n_estimators": [1000,800,600,1200,1300],
    'min_data_in_leaf': [450, 450, 450, 450, 450],
    # 'feature_fraction': [0.9,0.9,0.9,0.9,0.9],
    'bagging_fraction': [1, 0.9,0.8,0.7,0.95],
    'bagging_freq': [0, 3,5,3,4],
    'feature_fraction': [1, 0.9,0.9,0.9,0.9],
    "bagging_seed": [2018, 2019, 2020, 2021, 2022],
    'reg_alpha': [0.005,0.005,0.005,0.005,0.005],
    'reg_lambda': [0.1,0.1,0.1,0.1,0.1],
    # 'lambda_l1': [1,1,1,1,1],
    # 'lambda_l2': [0.001,0.001,0.001,0.001,0.001],  # 越小l2正则程度越高
    'min_gain_to_split': [0.1, 0.01,0.01,0.01,0.01],
    'verbose': [50, 50,50,-1,50],
    'cat_l2': 25.0,
    'cat_smooth': 2.0,
    # 'is_unbalance': [True,True] ,
    'random_state': [0, 2018,201812,2019,2020],
    'metric': [{'auc'}, {'auc'}],
})
dtypes = {
    # 'MachineIdentifier':                                    'category',
    'ProductName': 'category',
    'EngineVersion': 'category',
    'AppVersion': 'category',
    'AvSigVersion': 'category',
    'IsBeta': 'int8',
    'RtpStateBitfield': 'float16',
    'IsSxsPassiveMode': 'int8',
    'DefaultBrowsersIdentifier': 'float16',
    'AVProductStatesIdentifier': 'float32',
    'AVProductsInstalled': 'float16',
    'AVProductsEnabled': 'float16',
    'HasTpm': 'int8',
    'CountryIdentifier': 'int32',
    'CityIdentifier': 'float32',
    'OrganizationIdentifier': 'float16',
    'GeoNameIdentifier': 'float32',
    'LocaleEnglishNameIdentifier': 'int32',
    'Platform': 'category',
    'Processor': 'category',
    'OsVer': 'category',
    'OsBuild': 'int16',
    'OsSuite': 'int16',
    'OsPlatformSubRelease': 'category',
    'OsBuildLab': 'category',
    'SkuEdition': 'category',
    'IsProtected': 'float16',
    'AutoSampleOptIn': 'int8',
    'PuaMode': 'category',
    'SMode': 'float16',
    'IeVerIdentifier': 'float32',
    'SmartScreen': 'category',
    'Firewall': 'float16',
    'UacLuaenable': 'float64',
    'Census_MDC2FormFactor': 'category',
    'Census_DeviceFamily': 'category',
    'Census_OEMNameIdentifier': 'float32',
    'Census_OEMModelIdentifier': 'float32',
    'Census_ProcessorCoreCount': 'float16',
    'Census_ProcessorManufacturerIdentifier': 'float16',
    'Census_ProcessorModelIdentifier': 'float32',
    'Census_ProcessorClass': 'category',
    'Census_PrimaryDiskTotalCapacity': 'float32',
    'Census_PrimaryDiskTypeName': 'category',
    'Census_SystemVolumeTotalCapacity': 'float32',
    'Census_HasOpticalDiskDrive': 'int8',
    'Census_TotalPhysicalRAM': 'float32',
    'Census_ChassisTypeName': 'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float32',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'float32',
    'Census_InternalPrimaryDisplayResolutionVertical': 'float32',
    'Census_PowerPlatformRoleName': 'category',
    'Census_InternalBatteryType': 'category',
    'Census_InternalBatteryNumberOfCharges': 'float32',
    'Census_OSVersion': 'category',
    'Census_OSArchitecture': 'category',
    'Census_OSBranch': 'category',
    'Census_OSBuildNumber': 'int32',
    'Census_OSBuildRevision': 'int32',
    'Census_OSEdition': 'category',
    'Census_OSSkuName': 'category',
    'Census_OSInstallTypeName': 'category',
    'Census_OSInstallLanguageIdentifier': 'float16',
    'Census_OSUILocaleIdentifier': 'int32',
    'Census_OSWUAutoUpdateOptionsName': 'category',
    'Census_IsPortableOperatingSystem': 'int8',
    'Census_GenuineStateName': 'category',
    'Census_ActivationChannel': 'category',
    'Census_IsFlightingInternal': 'float16',
    'Census_IsFlightsDisabled': 'float16',
    'Census_FlightRing': 'category',
    'Census_ThresholdOptIn': 'float16',
    'Census_FirmwareManufacturerIdentifier': 'float32',
    'Census_FirmwareVersionIdentifier': 'float32',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsWIMBootEnabled': 'float16',
    'Census_IsVirtualDevice': 'float16',
    'Census_IsTouchEnabled': 'int8',
    'Census_IsPenCapable': 'int8',
    'Census_IsAlwaysOnAlwaysConnectedCapable': 'float16',
    'Wdft_IsGamer': 'float16',
    'Wdft_RegionIdentifier': 'float32',
    'HasDetections': 'int8'
}
origin_cate_feature = ['ProductName', 'EngineVersion', 'AppVersion',
                       'AvSigVersion', 'IsBeta', 'RtpStateBitfield', 'IsSxsPassiveMode',
                       'DefaultBrowsersIdentifier', 'AVProductStatesIdentifier',
                       'AVProductsInstalled', 'AVProductsEnabled', 'HasTpm',
                       'CountryIdentifier', 'CityIdentifier', 'OrganizationIdentifier',
                       'GeoNameIdentifier', 'LocaleEnglishNameIdentifier', 'Platform',
                       'Processor', 'OsVer', 'OsBuild', 'OsSuite', 'OsPlatformSubRelease',
                       'OsBuildLab', 'SkuEdition', 'IsProtected', 'AutoSampleOptIn', 'PuaMode',
                       'SMode', 'IeVerIdentifier', 'SmartScreen', 'Firewall', 'UacLuaenable',
                       'Census_MDC2FormFactor', 'Census_DeviceFamily',
                       'Census_OEMNameIdentifier', 'Census_OEMModelIdentifier',
                       'Census_ProcessorCoreCount', 'Census_ProcessorManufacturerIdentifier',
                       'Census_ProcessorModelIdentifier', 'Census_ProcessorClass',
                       'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
                       'Census_SystemVolumeTotalCapacity', 'Census_HasOpticalDiskDrive',
                       'Census_TotalPhysicalRAM', 'Census_ChassisTypeName',
                       'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                       'Census_InternalPrimaryDisplayResolutionHorizontal',
                       'Census_InternalPrimaryDisplayResolutionVertical',
                       'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
                       'Census_InternalBatteryNumberOfCharges', 'Census_OSVersion',
                       'Census_OSArchitecture', 'Census_OSBranch', 'Census_OSBuildNumber',
                       'Census_OSBuildRevision', 'Census_OSEdition', 'Census_OSSkuName',
                       'Census_OSInstallTypeName', 'Census_OSInstallLanguageIdentifier',
                       'Census_OSUILocaleIdentifier', 'Census_OSWUAutoUpdateOptionsName',
                       'Census_IsPortableOperatingSystem', 'Census_GenuineStateName',
                       'Census_ActivationChannel', 'Census_IsFlightingInternal',
                       'Census_IsFlightsDisabled', 'Census_FlightRing',
                       'Census_ThresholdOptIn', 'Census_FirmwareManufacturerIdentifier',
                       'Census_FirmwareVersionIdentifier', 'Census_IsSecureBootEnabled',
                       'Census_IsWIMBootEnabled', 'Census_IsVirtualDevice',
                       'Census_IsTouchEnabled', 'Census_IsPenCapable',
                       'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
                       'Wdft_RegionIdentifier']
origin_cate_feature.remove('Census_ProcessorClass')
id_columns = 'MachineIdentifier'


# -----------------------------param part
def get_feature(data):
    for column in origin_cate_feature:
        la_e = LabelEncoder()
        data[column] = data[column].astype(str)
        data[column] = la_e.fit_transform(data[column])
    return data


def predict_cross_validation(test, clfs):
    sub_preds = np.zeros(test.shape[0])
    for model in clfs:
        test_preds = model.predict_proba(test, num_iteration=model.best_iteration_)
        sub_preds += test_preds[:, 1]

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret


def predict_test_chunk(features, clfs, filename='result.csv', chunks=100000):
    for i_c, df in enumerate(pd.read_csv(data_base_path + 'test.csv', chunksize=chunks, iterator=True)):

        df.set_index(TARGET_INDEX, inplace=True)
        df = get_feature(df)
        preds_df = predict_cross_validation(df[features], clfs)
        preds_df = preds_df.to_frame(TARGET)
        print(i_c)
        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=True)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=True)

        del preds_df
        gc.collect()


# ------------ func part end--------------------------

if use_data_file and os.path.exists(data_base_path + 'data.csv'):
    data = pd.read_csv(data_base_path + 'data.csv')
else:
    train = pd.read_csv(data_base_path + 'train.csv', nrows=2000000).set_index(TARGET_INDEX)
    # train = train.sample(n=200000, random_state=201812)
    train = get_feature(train)
    train['label'] = train.HasDetections.astype(int)
    try:
        train.to_csv(data_base_path + 'data.csv', index=None)
        print('store success')
    except:
        print('warning: store failed')
print(" feature finish !!! ")

# ----------------------------------------------------feature part end--------------------------------------------------------------------------
feature = origin_cate_feature
train_features = [f for f in train.columns if f != TARGET and f != 'Census_ProcessorClass']

X = train[feature]
y = train.label
cv_pred = []
skf = StratifiedKFold(n_splits=5, random_state=201812, shuffle=True)
models = []
for index, (train_index, test_index) in enumerate(skf.split(X, y)):
    params = {
        "objective": "binary",
        "boosting_type": params_dict["boosting_type"][index],
        "learning_rate": params_dict["learning_rate"][index],
        "num_leaves": params_dict["num_leaves"][index],
        "max_depth": params_dict['max_depth'][index],
        "n_estimators": params_dict['n_estimators'][index],
        "bagging_fraction": params_dict["bagging_fraction"][index],
        "feature_fraction": params_dict["feature_fraction"][index],
        "bagging_freq": params_dict["bagging_freq"][index],
        "bagging_seed": params_dict["bagging_seed"][index],
        'reg_alpha': params_dict['reg_alpha'][index],
        'reg_lambda': params_dict['reg_lambda'][index],
        'subsample_for_bin': 25000,
        'min_data_per_group': 100,
        # 'max_cat_to_onehot': 4,
        'cat_l2': 25.0,
        'cat_smooth': 2.0,
        # 'max_cat_threshold': 32,
        "random_state": params_dict["random_state"][index],
        'min_child_samples': 80,
        'min_child_weight': 100.0,
        'min_split_gain': 0.1,
        "silent": True,
        "metric": "auc",
    }
    model_params = {
        'device': 'cpu',
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "max_depth": 11,
        "num_leaves": 31,
        "n_estimators": 1000,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_freq": 5,
        "bagging_seed": 2018,
        'min_child_samples': 80,
        'min_child_weight': 100.0,
        'min_split_gain': 0.1,
        'reg_alpha': 0.005,
        'reg_lambda': 0.1,
        'subsample_for_bin': 25000,
        'min_data_per_group': 100,
        'max_cat_to_onehot': 4,
        'cat_l2': 25.0,
        'cat_smooth': 2.0,
        'max_cat_threshold': 32,
        "random_state": 1,
        "silent": True,
        "metric": "auc",
    }
    lgb_model = lgb.LGBMClassifier(**model_params)
    train_x, valid_x, train_y, valid_y = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    eval_set = [(valid_x, valid_y)]
    lgb_model.fit(train_x, train_y, eval_set=eval_set, verbose=-1, eval_metric='auc', early_stopping_rounds=100,
                  categorical_feature=origin_cate_feature)
    y_socre = lgb_model.predict(valid_x, num_iteration=lgb_model.best_iteration_)
    score = roc_auc_score(valid_y, y_socre)
    models.append(lgb_model)
current_time = str(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))

predict_test_chunk(feature, models, filename=current_time+'result.csv', chunks=100000)
