#import
import pandas as pd
from pickle import dump, load
import sys

args = sys.argv

ml_data_path = args[1]
ml_data = pd.read_csv(ml_data_path)

#reduce memory size for data
cols = ml_data.keys().tolist()
for i,tp in enumerate(ml_data.dtypes):
    col = cols[i]
    if tp == 'int64':
        ml_data[col] = ml_data[col].astype('int8')
    elif tp == 'float64':
        ml_data[col] = ml_data[col].astype('float32')
#####################################################

stacked = pd.concat([ml_data]*6, ignore_index=True)

nmof = len(ml_data)

pressure = [1]*nmof + [5.8]*nmof + [35]*nmof + [65]*nmof + [100]*nmof + [248]*nmof

stacked['pressure'] = pressure

vol_uptake = []
uptake_1 = stacked['CH4_v/v_1_bar'].tolist()
uptake_5 = stacked['CH4_v/v_5.8_bar'].tolist()
uptake_35 = stacked['CH4_v/v_35_bar'].tolist()
uptake_65 = stacked['CH4_v/v_65_bar'].tolist()
uptake_100 = stacked['CH4_v/v_100_bar'].tolist()
uptake_248 = stacked['CH4_v/v_248_bar'].tolist()

#create uptake column
for ind, val in enumerate(pressure):
    if val == 1:
        vol_uptake.append(uptake_1[ind])
    elif val == 5.8:
        vol_uptake.append(uptake_5[ind])
    elif val == 35:
        vol_uptake.append(uptake_35[ind])
    elif val == 65:
        vol_uptake.append(uptake_65[ind])
    elif val == 100:
        vol_uptake.append(uptake_100[ind])
    else:
        vol_uptake.append(uptake_248[ind])
############################################

stacked['vol_uptake'] = vol_uptake

stacked = stacked.drop(['CH4_v/v_1_bar','CH4_v/v_5.8_bar','CH4_v/v_35_bar','CH4_v/v_65_bar','CH4_v/v_100_bar',
             'CH4_v/v_248_bar'], axis=1)

stacked = stacked[stacked['vol_uptake'].notna()].reset_index()

stacked = stacked.drop('index', axis=1)

stacked.to_csv('stacked.csv') #save the file