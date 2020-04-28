import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 
from os import path
import pandas as pd 
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
#K.clear_session()
from tensorflow.keras import layers

#ensure that not all of GPU is used
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
import datetime
import math
import time
import pickle
import random
from scipy.spatial import cKDTree
from sklearn import preprocessing
from sklearn.decomposition import PCA
import sys
from sklearn.metrics import r2_score as r2
from rdkit import Chem
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from multiprocessing import Pool

#following must be defined
# algo = 'xgb' #am I using XGBoost (xgb) or Neural Nets (nn)?
# total_frac = 1 #total fraction of data set to work with
# training_pct = .7 #how much percent of total fraction should be used for training
# random_split = False #make True if the training data should be chosen randomly
# n_remote = 2000 #the n_remote most remote points will be added to training set if random_split = False
# del_defective_mofs = False #make True if you want to remove all MOFs which a '0' value for at least one geometric property
# add_size_fp = False #make True if you want to add 20 feature columns, where each feature is the number of atoms in a linker
# size_dependent = False #make True if the input ML-ready data contains fingerprint which does not normalize each PG feature by number of atoms
# stacked = False #make True if the input ML-ready data contains pressure as feature
# n_core = 1 #number of cores to use
# ML_DATA_PATH = '/data/rgur/efrc/prep_data/all_v1/ml_data.csv'
# start_str = 'filename'
# end_str = 'valence_pa'
# del_geometric_fp = False #make True if you want to ignore the geometric features
# cat_col_names = ['oh_1', 'oh_2', 'oh_3', 'oh_4'] #names for interpenetration columns
# Y_DATA_PATH = '/data/rgur/efrc/data_DONOTTOUCH/hMOF_allData_March25_2013.xlsx' #path to original hMOF data
# default_params = {'objective':'reg:linear', 'colsample_bytree':0.3, 'learning_rate':0.1,
#                 'max_depth':15, 'alpha':10, 'n_estimators':10}
# n_trees = 30 #number of weak learners. Bigger is better until 5000
# save_pp = False #make True if you want to save the parity plot
#########################################################################
tickfontsize=20
labelfontsize = tickfontsize

def load_ml_data(ML_DATA_PATH):
    return pd.read_csv(ML_DATA_PATH)

def reduce_size(df):
    '''
    Remove unnecessary columns from df to decrease size
    '''
    return df.drop([col for col in df.keys() if 'Smiles' in col] + [col for col in 
                                                                             df.keys() if 'Unnamed' in col] + ['#_of_Linkers'] + ['Metal_ID'], axis=1)

def slim(df, total_frac):
    '''
    Return data set which contains total_frac of original data set
    '''
    if total_frac != 1:
        df = df.sample(frac=total_frac, random_state=0)
    return df

def getPGcolNames(df, start_str, end_str):
    '''
    Return list of PG-relevant column names
    '''
    for ind, col in enumerate(df.columns):
        if start_str == col:
            start_col = ind + 1
        elif end_str == col:
            end_col = ind
    return list(df.columns[start_col:end_col])

def getNonPGcolNames(add_size_fp, stacked, del_geometric_fp, cat_col_names):
    features = ['norm_valence_pa',
   'norm_atomic_rad_pa_(angstroms)',
     'norm_affinity_pa_(eV)',
       'norm_ionization_potential_pa_(eV)',
           'norm_electronegativity_pa']
    
    if not del_geometric_fp:
        features += ['norm_Dom._Pore_(ang.)',
                     'norm_Max._Pore_(ang.)',
                     'norm_Void_Fraction',
                     'norm_Surf._Area_(m2/g)',
                     'norm_Vol._Surf._Area',
                     'norm_Density']
    features += cat_col_names
    
    if stacked:
        features += ['norm_log_pressure']
        
    if add_size_fp:
        features += ['size_%s' %n for n in range(20)]
    
    return features

def merge_data(df, stacked, Y_DATA_PATH):
    '''
    Merge gravimetric uptake
    '''
    if not stacked:
        y_data = pd.read_excel(Y_DATA_PATH)
        df = df.join(y_data[['Crystal ID#', 'CH4 cm3/g 35 bar']].set_index('Crystal ID#'), on='Crystal_ID#')
        for key in df.keys():
            if ' ' in key:
                new_key = key.replace(' ', '_')
                df[new_key] = df[key]
                df = df.drop(key, axis=1)

    return df

def norm_col(df, col_name):
    
    mean = df[col_name].mean()
    std = df[col_name].std()
    
    property_used = 'norm_' + col_name
    df[property_used] = (df[col_name] - mean) / std

    return mean, std, property_used

def prepare_target(df, stacked):
    if stacked:
        to_norm = 'vol_uptake'
    else:
        to_norm = 'CH4_cm3/g_35_bar'
    mean, std, property_used = norm_col(df, to_norm)
    return mean, std, property_used

def delMissing(df, property_used, reset=True):
    '''
    Remove rows from df with missing values in target
    '''
    df = df.loc[df[property_used].dropna().index]
    if reset:
        df = df.reset_index().drop('index', axis=1)
    return df

def normPressure(df):
    '''
    Normalize pressure feature and globally save important metrics. Only use if stacked == True!
    '''
    df['log_pressure'] = np.log(df['pressure'].tolist())
    
    
    log_p_mean, log_p_std, property_used = norm_col(df, 'log_pressure')

    max_p = max(df[property_used].tolist())
    min_p = min(df[property_used].tolist())
    
    return df, log_p_mean, log_p_std, max_p, min_p

def get_cat(s):
    '''
    Returns interpenetration from filename
    '''
    if 'cat' in s:
        return int(s.split('_')[-1][0])
    else:
        return 0

def addCat(df, cat_col_names):
    '''
    Add normed one-hot encoded features for catenation (interpenetration) of MOF
    '''
    one_hot = [[0]*4 for i in range(len(df))]
    for i, f in enumerate(df['filename'].tolist()):
        one_hot[i][get_cat(f)] = 1
    oh_1 = []
    oh_2 = []
    oh_3 = []
    oh_4 = []
    for i in one_hot:
        oh_1.append(i[0])
        oh_2.append(i[1])
        oh_3.append(i[2])
        oh_4.append(i[3])
    df[cat_col_names[0]] = oh_1
    df[cat_col_names[1]] = oh_2
    df[cat_col_names[2]] = oh_3
    df[cat_col_names[3]] = oh_4

def getNAtoms(s):
    if type(s) == float:
        return 0
    mol = Chem.MolFromSmiles(s)
    mol2 = Chem.AddHs(mol)
    return len(mol2.GetAtoms())

def makeSizeCol(df, col_name, N_CORE=1):
    #print(col_name)
    smiles_list = df[col_name].tolist()
    n = col_name.split('_')[0][1:]
    #df['size_%s' %n] = [getNAtoms(s) for s in smiles_list] #not parallelized
    df['size_%s' %n] = Parallel(n_jobs=N_CORE)(delayed(getNAtoms)(s) for s in smiles_list)

def parallelize_dataframe(df, func, n_cores=1):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df    

def initializeSizeCols(df):
    size_names = []
    smiles_names = [col for col in df.keys() if 'Smiles' in col]
    for col in smiles_names:
        n = col.split('_')[0][1:]    
        df['size_%s' %n] = 0
        size_names.append('size_%s' %n)
    return size_names, smiles_names
    
def prepToSplit(algo, cat_si_sd, SD_ML_DATA_PATH, SI_ML_DATA_PATH, start_str_sd, end_str_sd, start_str_si, end_str_si, total_frac, 
                                        del_defective_mofs, add_size_fp, srt_size_fp, size_dependent, stacked, n_core, 
                                        del_geometric_fp, cat_col_names, Y_DATA_PATH, LS_DICT=None, return_features=False):
    '''
    Carry out a series of tasks to prepare the data to be split into test and train sets
    '''
    
    
    if stacked:
        print('\nStarting to Construct Isotherm Stacked Data Frame')
    else:
        print('\nStarting to Construct Gravimetric Uptake Data Frame')
    print('Using start_str_sd %s' %start_str_sd)
    print('Using end_str_sd %s' %end_str_sd)
    print('Using start_str_si %s' %start_str_si)
    print('Using end_str_si %s' %end_str_si)
    print('Total frac equals %s' %total_frac)
    
    if cat_si_sd:
        try:
            ml_data_sd = pd.read_csv(SD_ML_DATA_PATH)
        except:
            ml_data_sd = pd.read_csv(SD_ML_DATA_PATH, compression='gzip')
        try:
            ml_data_si = pd.read_csv(SI_ML_DATA_PATH)
        except:
            ml_data_si = pd.read_csv(SI_ML_DATA_PATH, compression='gzip')

        ml_data_si.columns = [col+'_si' for col in ml_data_si.columns]
        si_cols = ml_data_si.keys().tolist()
        si_start_ind = si_cols.index(start_str_si+'_si')
        si_end_ind = si_cols.index(end_str_si+'_si')
        si_pg_cols = si_cols[si_start_ind + 1:si_end_ind]

        ml_data = ml_data_sd.join(ml_data_si[si_pg_cols], rsuffix='_si')

        
        sd_cols = ml_data_sd.keys().tolist()
        sd_start_ind = sd_cols.index(start_str_sd)
        sd_end_ind = sd_cols.index(end_str_sd)
        sd_pg_cols = sd_cols[sd_start_ind + 1:sd_end_ind]
        
        pg_cols = si_pg_cols + sd_pg_cols
    else:
        if size_dependent:
            ML_DATA_PATH=SD_ML_DATA_PATH
            start_str = start_str_sd
            end_str= end_str_sd
        else:
            ML_DATA_PATH=SI_ML_DATA_PATH
            start_str = start_str_si
            end_str= end_str_si
        ml_data = load_ml_data(ML_DATA_PATH)
        pg_cols = getPGcolNames(ml_data, start_str, end_str)
    
    
    non_pg_cols = getNonPGcolNames(add_size_fp, stacked, del_geometric_fp, cat_col_names)
    
    features = pg_cols + non_pg_cols
    if return_features:
        return features
    print('\n')
    #print("Using following %s features" %len(features))
#     for i in features:
#         print(i)
        
    ml_data = slim(ml_data, total_frac)
    
    
    if add_size_fp:
        size_names = ['size_%s' %n for n in range(20)] 
        print('Starting To Make Linker Size Columns')        
        if LS_DICT == None:
            for col in [col for col in ml_data.keys() if 'Smiles' in col]:
                makeSizeCol(ml_data, col, n_core)
            if srt_size_fp:
                print('Starting to sort Linker Size Columns')           
                a = ml_data[size_names].values
                a.sort(axis=1)
                a = a[:, ::-1]
                ml_data[size_names] = a
        else:
            smiles_cols = [col for col in ml_data.keys() if 'Smiles' in col]
            def makeLSfromDict(row):
                f_name = row['filename']
                try:
                    LS_list = LS_DICT[f_name].tolist()
                except:
                    smiles = row[smiles_cols]
                    LS_list = [getNAtoms(s) for s in smiles]
                return pd.Series(row.tolist() + LS_list)
            
            old_cols = ml_data.columns.tolist()
            ml_data = ml_data.apply(makeLSfromDict, axis=1) #not parallelized
            ml_data.columns = old_cols + size_names
        print('Finished Making Linker Size Columns')
        if algo == 'nn':
            ml_data[size_names]=(ml_data[size_names]-ml_data[size_names].mean())/ml_data[size_names].std()

    ml_data = reduce_size(ml_data)

    if del_defective_mofs:
        check_cols = ['Dom._Pore_(ang.)',
                     'Max._Pore_(ang.)',
                     'Void_Fraction',
                     'Surf._Area_(m2/g)',
                     'Vol._Surf._Area',
                     'Density']
        to_drop = ml_data[(ml_data[check_cols].T == 0).any()].index
        ml_data = ml_data.drop(to_drop).reset_index().drop('index', axis=1)
    
    ml_data = merge_data(ml_data, stacked, Y_DATA_PATH)

    target_mean, target_std, property_used = prepare_target(ml_data, stacked)
    ml_data = delMissing(ml_data, property_used)
    addCat(ml_data, cat_col_names)

    if stacked:
        ml_data, log_p_mean, log_p_std, max_p, min_p = normPressure(ml_data)

    drop_cols = ml_data[features].columns[ml_data[features].isna().any()].tolist()
    print("The following columns have been dropped: %s" %drop_cols)
    ml_data = ml_data.drop(drop_cols, axis=1) #remove feature cols with NaN value

    features = [i for i in features if i not in drop_cols]
    
    if stacked:
        return ml_data, property_used, target_mean, target_std, features, (log_p_mean, log_p_std, max_p, min_p)
    else:
        return ml_data, property_used, target_mean, target_std, features

def NNtree(df, stacked, n_core, features, use_pca, N_COMPONENTS):
    start = time.time()
    print("Starting to make KDTree")
    if stacked:
        non_p_features = [col for col in features if col != 'norm_log_pressure']
        filenames = df[non_p_features + ['filename']].drop_duplicates()['filename'].tolist()
        if not use_pca:
            mat = df[non_p_features].drop_duplicates().to_numpy()
        else:
            pca = PCA(n_components=N_COMPONENTS)
            mat = pca.fit_transform(df[non_p_features].drop_duplicates())
    else:
        non_p_features = features
        filenames = df['filename'].tolist()
        if not use_pca:
            mat = df[non_p_features].to_numpy()
        else:
            pca = PCA(n_components=N_COMPONENTS)
            mat = pca.fit_transform(df[features])            
        
    sys.setrecursionlimit(10000)

    tree = cKDTree(mat)
    
    if n_core < 4:
        N_JOBS = 18
    else:
        N_JOBS = n_core
    
    nn_tree = tree.query(mat, k=2, n_jobs=N_JOBS)

    sys.setrecursionlimit(3000)
    
    end = time.time()
    print("Finished making KDTree in %s seconds" %(end-start))
    
    return nn_tree, filenames

def trainTestSplit(df, property_used, training_pct, stacked, n_core, rand, n_remote, features, use_pca, N_COMPONENTS):
    if rand:
        filenames = df['filename'].unique().tolist()
        random.shuffle(filenames)
        n_files = len(filenames)
        n_train = round(n_files*training_pct)
        train_fn = filenames[0:n_train]
        test_fn = filenames[n_train:]
    else:

        nn_tree, filenames = NNtree(df, stacked, n_core, features, use_pca, N_COMPONENTS)
        dt = nn_tree[0] #distance tree
        nt = nn_tree[1] #neighbor tree

    
        to_sort = list(zip(nt[:, 0],dt[:, 1])) 
        #l = list(to_sort) #so we can see the length
        #len(l)
        len(to_sort)

        srt_d = {}
        for i in to_sort:
            try:
                srt_d[i[0]] += i[1]
            except:
                srt_d[i[0]] = [i[1]]

        srt_ind_set = [(k, max(v)) for k,v in zip(srt_d.keys(), srt_d.values())]

        srt = sorted(srt_ind_set, key=lambda x: x[1], reverse=True)

        srt_inds = list([x[0] for x in srt])

        train_inds = srt_inds[:n_remote]

        print("Top five distance: ", train_inds[0:5])
        remaining = (list(set(range(len(filenames))) - set(train_inds))) 

        seed = 0
        random.Random(seed).shuffle(remaining)

        n_train_left = round(len(filenames)*training_pct - n_remote)

        train_inds += remaining[:n_train_left]

        train_fn = set([filenames[ind] for ind in train_inds])

        test_inds = remaining[n_train_left:]

        test_fn = set([filenames[ind] for ind in test_inds])   

    train_df = df[df['filename'].isin(train_fn)].reset_index().drop('index', axis=1)
    test_df = df[df['filename'].isin(test_fn)].reset_index().drop('index', axis=1)
    print("Total len of test_df + train_df: %s" %(len(train_df) + len(test_df)))
#     if stacked:
#         isotherm_inds = [0, 20, 50, 100, 500]
#         isotherm_files = [test_df.iloc[i]['filename'] for i in isotherm_inds]
#         isotherm_df = test_df[test_dataset['filename'].isin(isotherm_files)]

#         isotherm_pressures = []
#         isotherm_uptakes = []
#         for i in isotherm_files:
#             l1 = []
#             l2 = []
#             for row in isotherm_df.iterrows():
#                 if row[1]['filename'] == i:
#                     l1.append(row[1]['pressure'])
#                     l2.append(row[1]['vol_uptake'])
#             isotherm_pressures.append(l1)
#             isotherm_uptakes.append(l2)
    return train_df, test_df 

def alter_dtype(train_df, test_df, property_used, n_core, algo, features):
    train_fp = train_df[features].to_numpy().astype('float32')
    train_label = train_df[property_used]
    test_fp = test_df[features].to_numpy().astype('float32')
    test_label = test_df[property_used]
    
    if algo == 'xgb':
        train_d = xgb.DMatrix(data=train_fp, label=train_label, nthread=n_core)
        test_d = xgb.DMatrix(data=test_fp, label=test_label, nthread=n_core)
        return train_d, test_d, train_label, test_label
    if algo == 'nn':
        return (train_fp, train_label.to_numpy()), (test_fp, test_label.to_numpy()), train_label, test_label

def build_model(n_features, lr, h_units, ACTIVATION):
    #x = tf.placeholder('float', shape = [None, n_features])
    model = keras.Sequential([
        layers.Dense(h_units, activation='relu', input_shape=[n_features]), #default is 100
        #layers.Dense(100, activation='relu'), #default is not exist
        #layers.Dropout(.1), #default is not exist
        #layers.Dense(100, activation='relu'), #default is not exist
        #layers.Dropout(.1), #default is not exist
        layers.Dense(h_units, activation='relu'), #default is 100
        #layers.Dropout(.1), #default is not exist
        layers.Dense(h_units, activation='relu'), #default is 100
        #layers.Dropout(.1), #default is not exist
        #layers.Dense(1)
        layers.Dense(1, activation='linear') #default activation is None
    ])

#     model = keras.Sequential([
#         layers.Dense(400, activation='relu', input_shape=[len(train_fp.keys())]),
#         layers.Dense(400, activation='relu'),
#         layers.Dense(100, activation='relu'),
#         #layers.Dense(1)
#         layers.Dense(1, activation='linear')
#     ])

    opt = keras.optimizers.Adam(learning_rate=lr) #default is .001
    
    model.compile(loss='mse',
        optimizer=opt,
        metrics=['mae', 'mse'])
    return model
    
def run_model(algo, train_d, n_trees, params=None, n_core=None, chkpt_name='model_checkpoint'):
    
    start = time.time()
    if algo == 'xgb':
        print("Number of trees: %s" %n_trees)
        if params==None:
            params = default_params
        else:
            colsample_bytree = params[0]
            learning_rate = params[1]
            max_depth = params[2]
            alpha = params[3]
            params = {'colsample_bytree': colsample_bytree,
                     'learning_rate': learning_rate,
                     'max_depth': max_depth,
                     'alpha': alpha}
        
        model = xgb.train(params, train_d, n_trees)
    if algo == 'nn':
        try:
            lr = params[1]
        except:
            lr = .001 #default
        try:
            h_units = params[0]
        except:
            h_units = 100 #default
        try:
            patience = params[2]
        except:
            patience = 10 #default
        try:
            BS = params[3]
        except:
            BS = 32 #default
        try:
            VAL_SPLIT = params[4]
        except:
            VAL_SPLIT = .2 #default

        ACTIVATION = 'relu'    
        fp = train_d[0]
        label = train_d[1]
        EPOCHS = 1000
        n_features = fp.shape[1]
        model = build_model(n_features, lr, h_units, ACTIVATION)

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        checkpoint_callbacks = keras.callbacks.ModelCheckpoint(filepath='%s.h5' %chkpt_name, monitor='val_loss',\
                                                              verbose=1, save_best_only=True, mode='min')

        log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        if n_core != None:
            early_history = model.fit(fp, label, batch_size=BS,workers=n_core, use_multiprocessing=True,
                                epochs=EPOCHS, validation_split = VAL_SPLIT, verbose=1,\
                                      callbacks=[early_stop,checkpoint_callbacks,tfdocs.modeling.EpochDots(),tensorboard_callback])
        else:
            early_history = model.fit(fp, label, batch_size=BS,
                            epochs=EPOCHS, validation_split = VAL_SPLIT, verbose=1,\
                                  callbacks=[early_stop,checkpoint_callbacks,tfdocs.modeling.EpochDots(),tensorboard_callback])
        ##############################################        
    end = time.time()
    print("Elapsed time during model training: ", end-start)
    return model

def get_rmse(a, b):
    '''
    Compute rmse between a and b
    '''
    return math.sqrt(np.mean(np.square(np.subtract(a, b))))

def model_rmse(model, train_d, test_d, stacked, algo, target_mean, target_std, property_used, test_label, train_label, save=False, fname=None, subset_inds=None):
    if algo=='xgb':
        test_predictions = model.predict(test_d)
        train_predictions = model.predict(train_d)
    if algo=='nn':
        test_fp = test_d[0]
        train_fp = train_d[0]
        test_predictions = model.predict(test_fp).flatten()
        train_predictions = model.predict(train_fp).flatten()        
    res_test_predictions, res_test_label, res_train_label, res_train_predictions = unscale(property_used, 
                                                                                           test_predictions, 
                                                                                           train_predictions, 
                                                                                           test_label, train_label, 
                                                                                           target_mean, target_std)


    rmse = get_rmse(res_test_label, res_test_predictions)
    
    return rmse #lower is better

def unscale(property_name, test_predictions, train_predictions, test_label, train_label, target_mean, target_std):
    '''
    Undo the scaling on predictions of test set, labels of test set, labels of training set
    '''
    mean = target_mean
    std = target_std
    res_test_predictions = (test_predictions * std) + mean
    res_test_label = (test_label * std) + mean
    res_train_label = (train_label * std) + mean    
    res_train_predictions = (train_predictions * std) + mean   
    return res_test_predictions, res_test_label, res_train_label, res_train_predictions
                
def parity_plot(model, train_d, test_d, stacked, algo, target_mean, target_std, property_used, test_label, train_label, save=False, fname=None, subset_inds=None):
    '''
    Make parity plot and save with name equal to fname
    '''
    if algo=='xgb':
        test_predictions = model.predict(test_d)
        train_predictions = model.predict(train_d)
    if algo=='nn':
        test_fp = test_d[0]
        train_fp = train_d[0]
        test_predictions = model.predict(test_fp).flatten()
        train_predictions = model.predict(train_fp).flatten()        
    res_test_predictions, res_test_label, res_train_label, res_train_predictions = unscale(property_used, 
                                                                                           test_predictions, 
                                                                                           train_predictions, 
                                                                                           test_label, train_label, 
                                                                                           target_mean, target_std)
    fig1,ax1 = plt.subplots(figsize = (8,8))


    rmse = get_rmse(res_test_label, res_test_predictions)
    print("Test RMSE is %s" %rmse)

    tr_rmse = get_rmse(res_train_label, res_train_predictions)

    from sklearn.metrics import r2_score as r2

    r2_val = r2(y_true=res_test_label, y_pred=res_test_predictions)
    r2_tr = r2(y_true=res_train_label, y_pred=res_train_predictions)

    ax1.scatter(res_test_label, res_test_predictions, c='r',s=10, label='Test')
    if subset_inds != None:
        ax1.scatter([res_test_label[i] for i in subset_inds], [res_test_predictions[i] for i in subset_inds], c='b',s=10, label='Subset')
    if stacked:
        x_lab = 'True Volumetric Uptake'
        y_lab = 'Predicted Volumetric Uptake'
    else:
        x_lab = 'True Gravimetric Uptake'
        y_lab = 'Predicted Gravimetric Uptake'
        
    ax1.set_xlabel(x_lab,fontsize=labelfontsize)
    ax1.set_ylabel(y_lab,fontsize=labelfontsize)
    max_val = max([max(res_test_label),max(res_test_predictions)])+1
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)

    plot_x_min, plot_x_max = plt.xlim()
    plot_y_min, plot_y_max = plt.ylim()
    ax1.legend()
    ax1.plot(np.linspace(plot_x_min,plot_x_max,100),np.linspace(plot_y_min,plot_y_max,100),c='k',ls='--')
    text_position_x = plot_x_min + (plot_x_max - plot_x_min) * 0.05
    text_position_y = plot_y_max - (plot_y_max - plot_y_min) * 0.25

    ax1.text(text_position_x, text_position_y, "RMSE test=" + str("%.4f" % rmse) + '\n' + 
             "RMSE train=" + str("%.4f" % tr_rmse) + '\n' +
             "R2 test=" + str("%.4f" % r2_val) + '\n' +
             "R2 train=" + str("%.4f" % r2_tr), ha='left', fontsize=16)
    fig1.tight_layout()

    if save:
        plt.savefig('/data/rgur/efrc/ml/models/%s/plt_%s.png' %(fname, fname),dpi=200)

def getTime():
    return datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y")


from skopt import gp_minimize

from multiprocessing import Pool

class hmofMLdataset:
    def __init__(self, results_dir, SI_grav_data_path='/data/rgur/efrc/prep_data/all_v1/ml_data.csv', 
                 SD_grav_data_path='/data/rgur/efrc/prep_data/all_no_norm/ml_data.csv',SI_stacked_path=
                '/data/rgur/efrc/prep_data/all_v1/stacked.csv',
                 SD_stacked_path='/data/rgur/efrc/prep_data/all_no_norm/stacked.csv',
                 Y_DATA_PATH='/data/rgur/efrc/data_DONOTTOUCH/hMOF_allData_March25_2013.xlsx', n_core=8, skip=[]):
        self.results_dir = results_dir
        os.chdir(self.results_dir)
        self.SI_grav_data_path = SI_grav_data_path
        self.SD_grav_data_path = SD_grav_data_path
        self.SI_stacked_path = SI_stacked_path
        self.SD_stacked_path = SD_stacked_path
        self.n_core = n_core
        self.Y_DATA_PATH = Y_DATA_PATH
        self.PCA_COMPONENTS = 400
        self.del_defective_mofs = False
        self.cat_si_sd = True
        self.add_size_fp = True #make True if you want to add 20 feature columns, where each feature is the number of atoms in a linker
        self.srt_size_fp = True
        self.iso_start_str_sd = 'Density'
        self.iso_end_str_sd = 'norm_Dom._Pore_(ang.)'
        self.grav_start_str_sd = 'CH4_v/v_248_bar'
        self.grav_end_str_sd = 'norm_Dom._Pore_(ang.)'
        self.start_str_si = 'filename'
        self.end_str_si = 'valence_pa'
        self.cat_col_names = ['cat_1', 'cat_2', 'cat_3', 'cat_4']
        self.skip = skip
        self.feature_codes = ['10000', '11000', '01000', '10100', '11100', '01100',
                             '10010', '11010', '01010', '10110', '11110', '01110',
                             '10001', '11001', '01001', '10101', '11101', '01101',
                             '10011', '11011', '01011', '10111', '11111', '01111']
        self.feature_codes = [i for i in self.feature_codes if i not in self.skip]
        print("There are %s unique feature codes" %len(set(self.feature_codes)))
        print("now is %s" %now)
        
    
    def makeMasterDFs(self):
        #gravimetric
        self.grav, self.grav_prop, self.grav_target_mean, self.grav_target_std, self.grav_all_features = prepToSplit(
                                            'xgb', self.cat_si_sd, self.SD_grav_data_path, self.SI_grav_data_path, 
                                            self.grav_start_str_sd, self.grav_end_str_sd, self.start_str_si, 
                                            self.end_str_si, 1, self.del_defective_mofs, self.add_size_fp, 
                                            self.srt_size_fp, None, stacked=False, n_core=self.n_core, 
                                            del_geometric_fp=False, cat_col_names=self.cat_col_names, 
                                            Y_DATA_PATH=self.Y_DATA_PATH)
        size_cols = ["size_%s" %s for s in range(20)]
        self.LS_dict = {row[1]['filename']:row[1][size_cols] for row in self.grav.iterrows()} # map from filename 
                                                                                        #to linkersize-vector
        #stacked
        self.iso, self.iso_prop, self.iso_target_mean, self.iso_target_std, self.iso_all_features, self.pinfo = \
                                            prepToSplit(
                                            'nn', self.cat_si_sd, self.SD_stacked_path, self.SI_stacked_path, 
                                            self.iso_start_str_sd, self.iso_end_str_sd, self.start_str_si, 
                                            self.end_str_si, 1, self.del_defective_mofs, self.add_size_fp, 
                                            self.srt_size_fp, None, True, self.n_core, False, self.cat_col_names, 
                                            self.Y_DATA_PATH, self.LS_dict)

    def select_features(self, code, stacked):
        '''
        Should only be called after makeMasterDFs
        '''
        si = bool(int(code[0])) #True (=1) if size-independent features are included
        sd = bool(int(code[1])) #True (=1) if size-dependent features are included
        size_fp = bool(int(code[2])) #True (=1) if linker size features are included
        geo_fp = bool(int(code[3])) #True (=1) if geometric features are included
        non_pg = getNonPGcolNames(size_fp, stacked, not geo_fp, self.cat_col_names)
        pg = []
        if si:
            si_df = pd.read_csv(self.SI_grav_data_path)
            self.all_pg = [s for s in getPGcolNames(si_df, start_str=self.start_str_si, end_str=self.end_str_si)]
            pg += [s+'_si' for s in getPGcolNames(si_df, start_str=self.start_str_si, end_str=self.end_str_si) 
                   if s+'_si' in self.grav_all_features]
            del si_df
        if sd:
            sd_df = pd.read_csv(self.SD_grav_data_path)
            pg += [s for s in getPGcolNames(sd_df, self.grav_start_str_sd, self.grav_end_str_sd) if s in
                  self.grav_all_features]
        return non_pg + pg
    
    def makeResult(self, i):
        STACKED = bool(int(i[-1])) #True (=1) if stacked
        CODE = i[:-1]
        run_features = self.select_features(code=CODE, stacked=STACKED)
        if STACKED:
            print("Running code %s for isotherm model" %CODE)
            drop_features = [s for s in self.iso_all_features if s not in run_features]
            FpDataSet(self.iso.drop(drop_features, axis=1), run_features, self.iso_prop, 
                               self.iso_target_mean, self.iso_target_std, stacked=STACKED, fp_code=CODE).run()
        else:
            print("Running code %s for gravimetric uptake model" %CODE)
            drop_features = [s for s in self.grav_all_features if s not in run_features]
            FpDataSet(self.grav.drop(drop_features, axis=1), run_features, self.grav_prop, 
                               self.grav_target_mean, self.grav_target_std, stacked=STACKED, fp_code=CODE)        
    
    def makeAllResults(self):
        self.makeMasterDFs()
        print('\n')
        #Parallel(n_jobs=self.n_core)(delayed(self.makeResult)(j) for j in self.feature_codes)
        for i in self.feature_codes: #True if stacked
                STACKED = bool(int(i[-1])) #True (=1) if stacked
                CODE = i[:-1]
                run_features = self.select_features(code=CODE, stacked=STACKED)
                if STACKED:
                    print("Running code %s for isotherm model" %CODE)
                    drop_features = [s for s in self.iso_all_features if s not in run_features]
                    #l.append(self.iso.drop(drop_features, axis=1))
                    FpDataSet(self.iso.drop(drop_features, axis=1), run_features, self.iso_prop, 
                                       self.iso_target_mean, self.iso_target_std, stacked=STACKED, fp_code=CODE).run()
                else:
                    print("Running code %s for gravimetric uptake model" %CODE)
                    drop_features = [s for s in self.grav_all_features if s not in run_features]
                    FpDataSet(self.grav.drop(drop_features, axis=1), run_features, self.grav_prop, 
                                       self.grav_target_mean, self.grav_target_std, stacked=STACKED, fp_code=CODE).run()
       
class FpDataSet:
    def __init__(self, df, features, property_used, target_mean, target_std, stacked, fp_code, PCA_DIM=400, 
                 rand_seeds=[0, 10, 20], train_grid = [.5, .6, .7, .8, .9]):
        self.df = df
        self.rand_seeds = rand_seeds
        self.fp_code = fp_code
        self.property_used = property_used
        self.target_mean = target_mean
        self.target_std = target_std
        self.n_samples = len(self.df)
        self.PCA_DIM = min(PCA_DIM, self.n_samples)
        self.features = features
        self.stacked = stacked
        if self.stacked:
            self.algo = 'nn'
            self.model_tag = 'iso'
        else:
            self.algo = 'xgb'
            self.model_tag = 'grav'
        self.train_grid = train_grid
    
    def sortRemoteInds(self):
        '''
        Create list of most remote indices
        '''
        n_core = 1
        use_pca = True
        nn_tree, filenames = NNtree(self.df, self.stacked, n_core, self.features, use_pca, self.PCA_DIM)
        dt = nn_tree[0] #distance tree
        nt = nn_tree[1] #neighbor tree
        to_sort = list(zip(nt[:, 0],dt[:, 1]))

        srt_d = {}
        for i in to_sort:
            try:
                srt_d[i[0]] += i[1]
            except:
                srt_d[i[0]] = [i[1]]

        srt_ind_set = [(k, max(v)) for k,v in zip(srt_d.keys(), srt_d.values())]

        srt = sorted(srt_ind_set, key=lambda x: x[1], reverse=True)

        srt_inds = list([x[0] for x in srt])
        self.remote_info = [(x, filenames[x]) for x in srt_inds]
        
    
    def helper(self, data):
        train_pct = data[0]
        seed = data[1]
        TRAIN_PCT = int(round(train_pct*100))
        results_df, MODEL = trainTestSplit(self.df, train_pct, self.features, self.property_used,
                                                   self.target_mean, self.target_std, seed, self.remote_info,
                                                   self.stacked).makeResults()
        save_fragment = '%s_code_%s_train_%s_seed_%s_%s' %(self.model_tag, self.fp_code, TRAIN_PCT, seed, now)
        print("Save Results using Fragment %s" %save_fragment)
        results_df.to_csv('results_%s.csv' %save_fragment)
        try:
            MODEL.save_model('%s.xgb' %save_fragment)
        except:
            MODEL.save('%s.h5' %save_fragment,save_format='h5')
    def run(self):
        self.sortRemoteInds()
        self.results = []
        self.parallel_data = []
        for train_pct in self.train_grid:
            for seed in self.rand_seeds:
                self.parallel_data.append((train_pct, seed))
        Parallel(n_jobs=15)(delayed(self.helper)(j) for j in self.parallel_data)

class trainTestSplit:
    def __init__(self, df, train_pct, features, property_used, target_mean, target_std, seed, remote_info, stacked, 
                 hp_frac=.1, n_trees=5000):
        self.df = df
        self.filenames = df['filename'].unique().tolist()
        self.n_samples = len(self.filenames)
        self.train_pct = train_pct
        self.seed = seed
        self.target_mean = target_mean
        self.target_std = target_std
        self.remote_info = remote_info
        self.pct_remote = self.train_pct - .5
        self.n_remote = round(self.n_samples*self.pct_remote)
        self.n_train = round(self.n_samples*self.train_pct)
        self.n_random = self.n_train - self.n_remote
        self.stacked = stacked
        self.features = features
        self.property_used = property_used
        self.hp_frac = hp_frac
        self.n_trees = n_trees

        if self.stacked:
            self.algo = 'nn'
            self.model_tag = 'iso'
        else:
            self.algo = 'xgb'
            self.model_tag = 'grav'
        
    
    def split(self):
        '''
        Split into train and test set
        '''

        self.train_fn = [x[1] for x in self.remote_info[:self.n_remote]]
        
        self.remaining = list(set(self.filenames) - set(self.train_fn))

        random.Random(self.seed).shuffle(self.remaining)

        self.train_fn += self.remaining[:self.n_random]

        self.test_fn = self.remaining[self.n_random:]
        train_df = self.df[self.df['filename'].isin(self.train_fn)].reset_index().drop('index', axis=1)
        test_df = self.df[self.df['filename'].isin(self.test_fn)].reset_index().drop('index', axis=1)
        print("Total len of test_df + train_df: %s" %(len(train_df) + len(test_df)))
        train_fp = train_df[self.features].to_numpy().astype('float32')
        train_label = train_df[self.property_used]
        test_fp = test_df[self.features].to_numpy().astype('float32')
        test_label = test_df[self.property_used]
    
        if self.algo == 'xgb':
            train_d = xgb.DMatrix(data=train_fp, label=train_label)
            test_d = xgb.DMatrix(data=test_fp, label=test_label)
            return train_d, test_d, train_label, test_label
        if self.algo == 'nn':
            train_files = train_df['filename'].tolist()
            test_files = test_df['filename'].tolist()
            train_pressures = train_df['pressure'].tolist()
            test_pressures = test_df['pressure'].tolist()
            return (train_fp, train_label.to_numpy(), train_files, train_pressures), (test_fp, test_label.to_numpy(), \
                    test_files, test_pressures), train_label, test_label
        
    def hp_opt(self):
        if self.algo == 'nn':
            self.max_batch = 512
            if self.max_batch > self.n_train:
                self.max_batch = self.n_train // 2
            self.space = [(100, 400), #n_units
                    (.0005, .003),#learning rate
                    (2, 15), #patience
                    (12, self.max_batch), #batch size
                    (.01, .6)] #validation split
            
        else:
            self.max_depth_ub = 15
            if self.n_train < 200:
                self.max_depth_ub = 4
            self.space = [(.3, .95), #colsample_bytree
                            (.01, .5),#learning_rate
                            (2, 15), #max_depth
                            (1, 20)] #alpha
            
        hp_df = self.df.sample(frac=self.hp_frac, random_state=self.seed)
        hp_remote_info = [x for x in self.remote_info if x[1] in hp_df['filename'].tolist()]
        return HPOpt(hp_df, self.train_pct, self.features, self.property_used, self.target_mean, 
                            self.target_std, self.seed, hp_remote_info, self.stacked, self.space).get_params()
    
    def run_model(self):
        self.params = self.hp_opt()
        self.train_d, self.test_d, self.train_label, self.test_label = self.split()
        self.MODEL = run_model(self.algo, self.train_d, self.n_trees, self.params)
        
        
    def makeResults(self):
        self.run_model()
        if self.algo=='xgb':
            test_predictions = self.MODEL.predict(self.test_d)
            train_predictions = self.MODEL.predict(self.train_d)
            pressures = [35]*(self.n_samples)
            files = self.train_fn + self.test_fn
        if self.algo=='nn':
            test_fp = self.test_d[0]
            train_fp = self.train_d[0]
            files = self.train_d[2] + self.test_d[2]
            pressures = self.train_d[3] + self.test_d[3]
            test_predictions = self.MODEL.predict(test_fp).flatten()
            train_predictions = self.MODEL.predict(train_fp).flatten()        
        
        res_test_predictions, res_test_label, res_train_label, res_train_predictions = unscale(self.property_used, 
                                                                                       test_predictions, 
                                                                                       train_predictions, 
                                                                                       self.test_label, 
                                                                                       self.train_label, 
                                                                                    self.target_mean, 
                                                                                    self.target_std)
        preds = res_train_predictions.tolist() + res_test_predictions.tolist()
        sample_class = ['Train']*len(res_train_predictions) + ['Test']*len(res_test_predictions)
#        truth = res_train_label.tolist() + res_test_label.tolist()
#         results_df = pd.DataFrame({"Filename": files, "Pressure": pressures, "Class": sample_class,
#                                   "Prediction": preds, "Truth": truth})
        results_df = pd.DataFrame({"Filename": files, "Pressure": pressures, "Class": sample_class,
                                  "Prediction": preds})
        return results_df, self.MODEL

#class(HPtrainTestSplit(trainTestSplit):

class HPOpt:
    def __init__(self, df, train_pct, features, property_used, target_mean, target_std, seed, remote_info, stacked, 
                 space, n_trees=50):
        self.df = df
        self.seed = seed
        self.space = space
        self.remote_info = remote_info
        self.stacked = stacked
        self.property_used = property_used
        self.features = features
        self.target_mean = target_mean
        self.target_std = target_std
        self.n_trees = n_trees
        self.N_CALLS = 75
        
        
        if self.stacked:
            self.algo = 'nn'
        else:
            self.algo = 'xgb'
        self.train_pct = train_pct
    
    def objective(self, params):
        MODEL = run_model(self.algo, self.train_d, self.n_trees, params)
        return model_rmse(MODEL, self.train_d, self.test_d, self.stacked, self.algo, self.target_mean, 
                             self.target_std, self.property_used, self.test_label, self.train_label, save=False, 
                             fname=None, subset_inds=None)
    
    def get_params(self):
        HP_TTS = trainTestSplit(self.df, self.train_pct, 
                                                        self.features, self.property_used, self.target_mean,
                                                        self.target_std, self.seed, self.remote_info, 
                                                        self.stacked)
        self.train_d, self.test_d, self.train_label, self.test_label = HP_TTS.split()
        defects = []
        start = time.time()
        r = gp_minimize(self.objective, self.space, n_calls=self.N_CALLS, random_state=self.seed)
        end = time.time()
        self.params = r.x
        return self.params