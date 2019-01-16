""
import os
import gc
from functools import partial, wraps
from datetime import datetime as dt
import warnings
warnings.simplefilter('ignore', FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ensemble.feature_get import *
import lightgbm as lgb

data_base_path = '../Data/'




def modeling_cross_validation(params, X, y, nr_folds=5):
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = StratifiedKFold(n_splits=nr_folds, shuffle=False, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        # LightGBM Regressor estimator
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=-1, eval_metric='auc',
            early_stopping_rounds=100
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)

    #score = roc_auc_score(y, oof_preds)
    score = 0.545
    print(score)
    return clfs, score


def get_importances(clfs, feature_names):
    # Make importance dataframe
    importances = pd.DataFrame()
    for i, model in enumerate(clfs, 1):
        # Feature importance
        imp_df = pd.DataFrame({
                "feature": feature_names, 
                "gain": model.booster_.feature_importance(importance_type='gain'),
                "fold": model.n_features_,
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    importances['gain_log'] = importances['gain']
    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
    importances.to_csv('importance.csv', index=False)
    # plt.figure(figsize=(8, 12))
    # sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
    return importances


def predict_cross_validation(test, clfs):
    sub_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):    
        test_preds = model.predict_proba(test, num_iteration=model.best_iteration_)
        sub_preds += test_preds[:,1]

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret


def predict_test_chunk(features, clfs, dtypes, filename='tmp.csv', chunks=100000):

    for i_c, test_df_chunk in enumerate(pd.read_csv(data_base_path+'test.csv',
                                         chunksize=chunks, 
                                         dtype=dtypes, 
                                         iterator=True)):

        test_df_chunk.set_index(TARGET_INDEX, inplace=True)
        test_df_chunk = feature_func(test_df_chunk)
        preds_df = predict_cross_validation(test_df_chunk[features], clfs)
        preds_df = preds_df.to_frame(TARGET)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=True)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=True)

        del preds_df
        gc.collect()


def main():

    dtypes = {
        #'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'category',
        'RtpStateBitfield':                                     'category',
        'IsSxsPassiveMode':                                     'category',
        'DefaultBrowsersIdentifier':                            'category',
        'AVProductStatesIdentifier':                            'category',
        'AVProductsInstalled':                                  'category',
        'AVProductsEnabled':                                    'category',
        'HasTpm':                                               'category',
        'CountryIdentifier':                                    'category',
        'CityIdentifier':                                       'category',
        'OrganizationIdentifier':                               'category',
        'GeoNameIdentifier':                                    'category',
        'LocaleEnglishNameIdentifier':                          'category',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'category',
        'OsSuite':                                              'category',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'category',
        'AutoSampleOptIn':                                      'category',
        'PuaMode':                                              'category',
        'SMode':                                                'category',
        'IeVerIdentifier':                                      'category',
        'SmartScreen':                                          'category',
        'Firewall':                                             'category',
        'UacLuaenable':                                         'category',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'category',
        'Census_OEMModelIdentifier':                            'category',
        'Census_ProcessorCoreCount':                            'category',
        'Census_ProcessorManufacturerIdentifier':               'category',
        'Census_ProcessorModelIdentifier':                      'category',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'category',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'category',
        'Census_HasOpticalDiskDrive':                           'category',
        'Census_TotalPhysicalRAM':                              'category',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'category',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'category',
        'Census_InternalPrimaryDisplayResolutionVertical':      'category',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'category',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'category',
        'Census_OSBuildRevision':                               'category',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'category',
        'Census_OSUILocaleIdentifier':                          'category',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'category',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'category',
        'Census_IsFlightsDisabled':                             'category',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'category',
        'Census_FirmwareManufacturerIdentifier':                'category',
        'Census_FirmwareVersionIdentifier':                     'category',
        'Census_IsSecureBootEnabled':                           'category',
        'Census_IsWIMBootEnabled':                              'category',
        'Census_IsVirtualDevice':                               'category',
        'Census_IsTouchEnabled':                                'category',
        'Census_IsPenCapable':                                  'category',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'category',
        'Wdft_IsGamer':                                         'category',
        'Wdft_RegionIdentifier':                                'category',
        'HasDetections':                                        'category',
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

    train_df = pd.read_csv(data_base_path+'train.csv', nrows=2000000, dtype=dtypes).set_index(TARGET_INDEX)
    train_df = feature_func(train_df)
    print('feature finish')
    train_features = [f for f in train_df.columns if f != TARGET and f != 'Census_ProcessorClass']
    print(train_features)
    clfs, score = modeling_cross_validation(model_params, train_df[train_features], train_df[TARGET])
    filename = 'subm_{:.6f}_{}_{}.csv'.format(score, 'LGBM', dt.now().strftime('%Y-%m-%d-%H-%M'))
    predict_test_chunk(train_features, clfs, dtypes, filename=filename, chunks=100000)


if __name__ == '__main__':
    main()
