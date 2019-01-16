import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
TARGET = 'HasDetections'
TARGET_INDEX = 'MachineIdentifier'
data_base_path = '../Data/'
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
categorical_columns = [c for c,v in dtypes.items() if v not in numerics]
def generate_piece():
    test = pd.read_csv(data_base_path+'test.csv',nrows=10)
    test = test[0:1]
    train = pd.read_csv(data_base_path+'train.csv',nrows=10)
    train = train[0:3]
    columns = list(train.columns)

    file = open('piece_of_train.txt','a',encoding='utf-8')
    for i in columns:
        value = list(train[i])
        ready_to_write = i
        for j in value:
            ready_to_write += '\t'+str(j)
        ready_to_write += '\n'
        file.write(ready_to_write)
    file.close()
    train.to_csv('piece.csv',index=None)


def count(filename):
    i=0
    file = open(filename)
    while(file.readline()):
        i += 1
        print(i)

    print(i)


def values_of_columns_count():
    train = pd.read_csv('../Data/train.csv', dtype=dtypes)
    test = pd.read_csv('../Data/test.csv', dtype=dtypes)
    retained_columns = numerical_columns + categorical_columns
    retained_columns.remove('HasDetections')
    df_all = pd.concat((train, test), axis=0)
    columns = df_all.columns
    embed_cols = []
    len_embed_cols = []

    col_value_dict={'feature_name':[],'value':[]}
    for c in tqdm(columns[1:]):
        embed_cols.append(c)
        len_embed_cols.append(df_all[c].nunique())
        col_value_dict['feature_name'].append(c)
        col_value_dict['value'].append(df_all[c].nunique())
        print(c + ': %d values' % df_all[c].nunique())  # look at value counts to know the embedding dimensions
    print('\n Number of embed features :', len(embed_cols))
    col_value_df = pd.DataFrame(col_value_dict)
    col_value_df.sort_values(by='value', inplace=True)
    col_value_df.to_csv('feature_value_num.csv', index=None, encoding='utf-8')


def distribution_of_feature(df, useful_feature,folder = 'jpg/'):
    for col in useful_feature:

        plt.figure(figsize=(20, 20))
        sns.distplot(df[col].values, bins=50, kde=False)
        plt.title("Histogram of yield")
        plt.xlabel(col, fontsize=12)
        if '/' in col:
            col = col.replace('/','d')
        if '*' in col:
            col = col.replace('*', 'pd')
        plt.savefig(folder+col+'_count.jpg')
        plt.close()


if __name__=='__main__':
    train = pd.read_csv('../Data/train.csv', dtype=dtypes)
    test = pd.read_csv('../Data/test.csv', dtype=dtypes)
    # train['data_flag'] = 1
    # test['data_flag'] = 0
    train_features = [f for f in train.columns if f != TARGET and f != 'Census_ProcessorClass']
    distribution_of_feature(train, train_features, folder='train_diagram/')
    distribution_of_feature(train, train_features, folder='test_diagram/')


