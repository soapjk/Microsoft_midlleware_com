from utils.DataPrepare import *


def feature_func(df_train):
    #Data_preprocess()
    #origin_feature_set = eval(open('../Data/feature_value_num.csv', 'r').readline())
    df_train['EngineVersion_encode'] = df_train['EngineVersion'].replace({'1.1.10302.0':0, '1.1.10401.0':1, '1.1.10701.0':2, '1.1.11104.0':3, '1.1.11202.0':4, '1.1.11302.0':5, '1.1.11400.0':6, '1.1.11502.0':7,
'1.1.11602.0':8, '1.1.11701.0':9, '1.1.11804.0':10, '1.1.11903.0':11, '1.1.12002.0':12, '1.1.12101.0':13, '1.1.12205.0':14, '1.1.12300.0':15,
'1.1.12400.0':16, '1.1.12505.0':17, '1.1.12603.0':18, '1.1.12706.0':19, '1.1.12802.0':20, '1.1.12804.0':21, '1.1.12805.0':22, '1.1.12902.0':23,
'1.1.13000.0':24, '1.1.13102.0':25, '1.1.13103.0':26, '1.1.13202.0':27, '1.1.13301.0':28, '1.1.13303.0':29, '1.1.13406.0':30, '1.1.13407.0':31,
'1.1.13503.0':32, '1.1.13504.0':33, '1.1.13601.0':34, '1.1.13701.0':35, '1.1.13704.0':36, '1.1.13802.0':37, '1.1.13803.0':38, '1.1.13804.0':39,
'1.1.13902.0':40, '1.1.13903.0':41, '1.1.14001.0':42, '1.1.14002.0':43, '1.1.14003.0':44, '1.1.14102.0':45, '1.1.14103.0':46, '1.1.14104.0':47,
'1.1.14201.0':48, '1.1.14202.0':49, '1.1.14303.0':50, '1.1.14305.0':51, '1.1.14306.0':52, '1.1.14405.2':53, '1.1.14500.2':54, '1.1.14500.5':55,
'1.1.14600.4':56, '1.1.14700.3':57, '1.1.14700.4':58, '1.1.14700.5':59, '1.1.14800.1':60, '1.1.14800.3':61, '1.1.14901.3':62, '1.1.14901.4':63,
'1.1.15000.1':64, '1.1.15000.2':65, '1.1.15100.1':66, '1.1.15200.1':67, '1.1.15300.5':68, '1.1.15300.6':69, '1.1.15400.3':70, '1.1.15400.4':71,
'1.1.15400.5':72, '1.1.9700.0':73})
    df_train['AppVersion_encode'] = df_train['AppVersion'].replace({'4.4.304.0':0, '4.4.306.0':1, '4.5.212.0':2, '4.5.216.0':3, '4.5.218.0':4, '4.6.302.0':5, '4.6.305.0':6, '4.7.205.0':7, '4.7.209.0':8, '4.8.10240.16384':9,
'4.8.10240.16425':10, '4.8.10240.17071':11, '4.8.10240.17113':12, '4.8.10240.17146':13, '4.8.10240.17184':14, '4.8.10240.17202':15, '4.8.10240.17319':16,
'4.8.10240.17354':17, '4.8.10240.17394':18, '4.8.10240.17443':19, '4.8.10240.17446':20, '4.8.10240.17533':21, '4.8.10240.17609':22, '4.8.10240.17770':23,
'4.8.10240.17797':24, '4.8.10240.17861':25, '4.8.10240.17889':26, '4.8.10240.17914':27, '4.8.10240.17918':28, '4.8.10240.17943':29, '4.8.10240.17946':30,
'4.8.10240.18036':31, '4.8.203.0':32, '4.8.204.0':33, '4.8.207.0':34, '4.9.10586.0':35, '4.9.10586.1045':36, '4.9.10586.1106':37, '4.9.10586.1177':38,
'4.9.10586.456':39, '4.9.10586.461':40, '4.9.10586.494':41, '4.9.10586.589':42, '4.9.10586.672':43, '4.9.10586.839':44, '4.9.10586.873':45,
'4.9.10586.916':46, '4.9.10586.962':47, '4.9.10586.965':48, '4.9.218.0':49, '4.9.219.0':50, '4.10.14393.0':51, '4.10.14393.1066':52,
'4.10.14393.1198':53, '4.10.14393.1532':54, '4.10.14393.1593':55, '4.10.14393.1613':56, '4.10.14393.1794':57, '4.10.14393.2248':58, '4.10.14393.2273':59,
'4.10.14393.2457':60, '4.10.14393.2608':61, '4.10.14393.726':62, '4.10.14393.953':63, '4.10.205.0':64, '4.10.207.0':65, '4.10.209.0':66,
'4.11.15063.0':67, '4.11.15063.1154':68, '4.11.15063.1155':69, '4.11.15063.447':70, '4.11.15063.994':71, '4.12.16299.0':72, '4.12.16299.15':73,
'4.12.17007.17121':74, '4.12.17007.17123':75, '4.12.17007.18011':76, '4.12.17007.18021':77, '4.12.17007.18022':78, '4.13.17133.1':79,
'4.13.17134.1':80, '4.13.17134.112':81, '4.13.17134.191':82, '4.13.17134.226':83, '4.13.17134.228':84, '4.13.17134.319':85, '4.13.17134.320':86,
'4.13.17604.1000':87, '4.13.17618.1000':88, '4.13.17623.1002':89, '4.13.17627.1000':90, '4.13.17634.1000':91, '4.13.17639.1000':92, '4.14.17613.18038':93,
'4.14.17613.18039':94, '4.14.17639.18041':95, '4.15.17643.1000':96, '4.15.17650.1001':97, '4.15.17655.1000':98, '4.15.17661.1001':99,
'4.15.17666.1000':100, '4.16.17656.18051':101, '4.16.17656.18052':102, '4.17.17672.1000':103, '4.17.17677.1000':104, '4.17.17682.1000':105,
'4.17.17685.20082':106, '4.17.17686.1003':107, '4.17.17686.1004':108, '4.18.1806.18062':109, '4.18.1806.20015':110, '4.18.1806.20021':111,
'4.18.1806.20033':112, '4.18.1807.18070':113, '4.18.1807.18072':114, '4.18.1807.18075':115, '4.18.1807.20063':116, '4.18.1809.2':117,
'4.18.1809.20006':118, '4.18.1810.20017':119, '4.18.1810.20021':120, '4.18.1810.20029':121, '4.18.1810.20037':122, '4.18.1810.5':123})
    df_train['OsVer_encode'] = df_train['OsVer'].replace({'6.1.0.0':0, '6.1.0.112':1, '6.1.0.128':2, '6.1.1.0':3, '6.1.16.36':4, '6.1.2.0':5, '6.1.3.0':6, '6.1.4.0':7, '6.1.6.0':8, '6.1.7.0':9, '6.1.80.0':10,
'6.3.0.0':11, '6.3.0.112':12, '6.3.0.117':13, '6.3.0.16':14, '6.3.0.2':15, '6.3.1.0':16, '6.3.1.144':17, '6.3.153.153':18, '6.3.16.0':19, '6.3.2.0':20,
'6.3.3.0':21, '6.3.32.72':22, '6.3.4.0':23, '6.3.5.0':24, '6.3.7.0':25, '6.3.80.0':26, '10.0.0.0':27, '10.0.0.1':28, '10.0.0.112':29, '10.0.0.2':30,
'10.0.0.22':31, '10.0.0.250':32, '10.0.0.3':33, '10.0.0.80':34, '10.0.0.96':35, '10.0.1.0':36, '10.0.1.144':37, '10.0.1.244':38, '10.0.1.44':39,
'10.0.153.153':40, '10.0.16.0':41, '10.0.16.36':42, '10.0.19.80':43, '10.0.2.0':44, '10.0.2.80':45, '10.0.2.86':46, '10.0.21.0':47, '10.0.23.0':48,
'10.0.26.128':49, '10.0.3.0':50, '10.0.3.232':51, '10.0.3.80':52, '10.0.32.0':53, '10.0.32.72':54, '10.0.4.0':55, '10.0.4.80':56, '10.0.48.0':57,
'10.0.5.0':58, '10.0.5.117':59, '10.0.5.18':60, '10.0.6.0':61, '10.0.64.150':62, '10.0.7.0':63, '10.0.7.101':64, '10.0.7.80':65, '10.0.72.0':66,
'10.0.8.0':67, '10.0.80.0':67})
    return df_train