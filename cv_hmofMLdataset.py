from joblib import Parallel, delayed
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
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow_docs as tfdocs
#import tensorflow_docs.plots
#import tensorflow_docs.modeling
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
#from rdkit import Chem
from sklearn.decomposition import PCA
import rishi_utils as ru

import importlib
import efrc_ml_production as ml
importlib.reload(ml)
importlib.reload(ru)
from skopt import gp_minimize

from multiprocessing import Pool

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.enable_eager_execution(config=config)
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from sklearn.model_selection import LeaveOneOut,KFold

class hmofMLdataset:
    def __init__(self, results_dir, now, SI_grav_data_path='/data/rgur/efrc/prep_data/all_v1/ml_data.csv', 
                 SD_grav_data_path='/data/rgur/efrc/prep_data/all_no_norm/ml_data.csv',SI_stacked_path=
                '/data/rgur/efrc/prep_data/all_v1/stacked.csv',
                 SD_stacked_path='/data/rgur/efrc/prep_data/all_no_norm/stacked.csv',
                 Y_DATA_PATH='/data/rgur/efrc/data_DONOTTOUCH/hMOF_allData_March25_2013.xlsx', n_core=15, skip=None, 
                 do=None, nn_space=None, grav_algo='xgb'):
        self.results_dir = results_dir 
        os.chdir(self.results_dir)
        self.SI_grav_data_path = SI_grav_data_path
        self.SD_grav_data_path = SD_grav_data_path
        self.SI_stacked_path = SI_stacked_path
        self.SD_stacked_path = SD_stacked_path
        self.n_core = n_core
        self.Y_DATA_PATH = Y_DATA_PATH
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
        self.do = do
        if self.do != None:
            self.feature_codes = self.do
        elif skip != None:
            self.feature_codes = [i for i in self.feature_codes if i not in self.skip]   
        print("There are %s unique feature codes" %len(set(self.feature_codes)))
        self.any_stacked = any([item[-1]=='1' for item in self.feature_codes]) #are any codes for stacked models?
        self.now = now
        print("now is %s" %self.now)
        self.nn_space = nn_space
        self.grav_algo = grav_algo
    def makeMasterDFs(self):
        #gravimetric
        self.grav, self.grav_prop, self.grav_target_mean, self.grav_target_std, self.grav_all_features = \
                                            ml.prepToSplit(
                                            self.grav_algo, self.cat_si_sd, self.SD_grav_data_path, self.SI_grav_data_path, 
                                            self.grav_start_str_sd, self.grav_end_str_sd, self.start_str_si, 
                                            self.end_str_si, 1, self.del_defective_mofs, self.add_size_fp, 
                                            self.srt_size_fp, None, stacked=False, n_core=self.n_core, 
                                            del_geometric_fp=False, cat_col_names=self.cat_col_names, 
                                            Y_DATA_PATH=self.Y_DATA_PATH
                                            )
        size_cols = ["size_%s" %s for s in range(20)]
        self.LS_dict = {row[1]['filename']:row[1][size_cols] for row in self.grav.iterrows()} # map from filename 
                                                                                        #to linkersize-vector
        #stacked
        if self.any_stacked:
            self.iso, self.iso_prop, self.iso_target_mean, self.iso_target_std, self.iso_all_features, self.pinfo = \
                                                ml.prepToSplit(
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
        non_pg = ml.getNonPGcolNames(size_fp, stacked, not geo_fp, self.cat_col_names)
        pg = []
        if si:
            try:
                si_df = pd.read_csv(self.SI_grav_data_path)
            except:
                si_df = pd.read_csv(self.SI_grav_data_path, compression='gzip')
            self.all_pg = [s for s in ml.getPGcolNames(si_df, start_str=self.start_str_si, end_str=self.end_str_si)]
            pg += [s+'_si' for s in ml.getPGcolNames(si_df, start_str=self.start_str_si, end_str=self.end_str_si) 
                   if s+'_si' in self.grav_all_features]
            del si_df
        if sd:
            try:
                sd_df = pd.read_csv(self.SD_grav_data_path)
            except:
                sd_df = pd.read_csv(self.SD_grav_data_path, compression='gzip')
            pg += [s for s in ml.getPGcolNames(sd_df, self.grav_start_str_sd, self.grav_end_str_sd) if s in
                  self.grav_all_features]
        return non_pg + pg
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
                    algo = 'nn'
                else:
                    print("Running code %s for gravimetric uptake model" %CODE)
                    algo = self.grav_algo
                    drop_features = [s for s in self.grav_all_features if s not in run_features]
                    #l.append(self.iso.drop(drop_features, axis=1))
                if algo == 'nn':
                    N_CORE=1
                else:
                    N_CORE=self.n_core
                if STACKED:
                    FpDataSet(self.iso.drop(drop_features, axis=1), run_features, self.iso_prop, 
                              self.iso_target_mean, self.iso_target_std, now=self.now, nn_space=self.nn_space, 
                              stacked=STACKED, fp_code=CODE, n_core=N_CORE, grav_algo=self.grav_algo).train()
                else:
                    FpDataSet(self.grav.drop(drop_features, axis=1), run_features, self.grav_prop, 
                              self.grav_target_mean, self.grav_target_std, now=self.now, nn_space=self.nn_space,
                              stacked=STACKED, fp_code=CODE, n_core=N_CORE, grav_algo=self.grav_algo).train()
class FpDataSet:
    def __init__(self, df, features, property_used, target_mean, target_std, stacked, now, nn_space, fp_code='0', 
                    n_core=15, grav_algo='xgb', track=True, chkpt_name='model_checkpoint',n_folds=15):
        self.n_folds = n_folds #for master run
        self.now = now
        self.df = df
        self.fp_code = fp_code
        self.property_used = property_used
        self.target_mean = target_mean
        self.target_std = target_std
        self.n_samples = len(self.df)
        self.features = features
        self.stacked = stacked
        self.n_core = n_core
        self.grav_algo = grav_algo
        self.track=track
        self.chkpt_name = chkpt_name
        self.file_tracker = {'train':[],'test':[]}
        self.pressure_tracker = {'train':[],'test':[]}
        if self.stacked:
            self.algo = 'nn'
            self.model_tag = 'iso'
        else:
            self.algo = self.grav_algo
            self.model_tag = 'grav'
        self.nn_space = nn_space
        self.hp_frac = .05
        self.fn = self.df['filename'].unique() #filenames
        
    def make_splits(self):
        def gen():
            for train_index, test_index in KFold(self.n_folds).split(self.fn):
                train_fn = self.fn[train_index]
                test_fn = self.fn[test_index]
                train_df = self.df[self.df['filename'].isin(train_fn)].reset_index().drop('index', axis=1)
                test_df = self.df[self.df['filename'].isin(test_fn)].reset_index().drop('index', axis=1)
                if self.track:
                    self.file_tracker['train'].append(train_df['filename'].tolist())
                    self.file_tracker['test'].append(test_df['filename'].tolist())
                    try:
                        self.pressure_tracker['train'].append(train_df['pressure'].tolist())
                        self.pressure_tracker['test'].append(test_df['pressure'].tolist())
                    except:
                        self.pressure_tracker['train'].append(['na']*len(train_df))
                        self.pressure_tracker['test'].append(['na']*len(test_df))
                X_train, X_test = train_df[self.features].to_numpy(), test_df[self.features].to_numpy()
                y_train, y_test = train_df[self.property_used].to_numpy(), test_df[self.property_used].to_numpy()
                yield X_train,y_train,X_test,y_test

        return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))    
    def CV_objective(self, params):
        try:
            lr = params[1]
        except:
            lr = .001 #default
        try:
            h_units = params[0]
        except:
            h_units = 100 #default
        patience = 15 #default
        try:
            BS = params[2]
        except:
            BS = 32 #default
        dataset = self.make_splits()
        start = time.time()
        import datetime
        rmses = []
        os.system('rm %s.h5' %self.chkpt_name)
        try:
            os.system('rm -rf logs')
        except:
            pass
        fold = 0
        for X_train,y_train,X_test,y_test in dataset:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
            checkpoint_callbacks = keras.callbacks.ModelCheckpoint(filepath='%s.h5' %self.chkpt_name, monitor='val_loss',\
                                                                  verbose=1, save_best_only=True, mode='min')
            model = ml.build_model(X_train.shape[1], lr, h_units, 'relu')

            model.fit(X_train, y_train, batch_size=BS,epochs=1000, verbose=1, 
                      callbacks=[checkpoint_callbacks, early_stop],
                        validation_split=None, validation_data=(X_test,y_test), shuffle=True, class_weight=None,
                        sample_weight=None, initial_epoch=0, steps_per_epoch=None,
                        validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
                        use_multiprocessing=False
                     ) 
            model.load_weights(filepath='%s.h5' %self.chkpt_name)
            preds = model.predict(X_test).flatten()
            rmse = ml.get_rmse(preds, y_test)
            print("RMSE of fold %s is %s" %( fold,rmse ))
            rmses.append(rmse)
            if self.track:
                n_train = len(self.file_tracker['train'][fold])
                n_test = len(self.file_tracker['test'][fold])
                res_test_predictions, res_test_label, res_train_label, res_train_predictions = \
                    ml.unscale(preds, model.predict(X_train).flatten(), y_test.numpy(), 
                           y_train.numpy(), self.target_mean, self.target_std)
                results_df = pd.DataFrame({"Filename": self.file_tracker['train'][fold]+ \
                                       self.file_tracker['test'][fold], 
                                       "Pressure": self.pressure_tracker['train'][fold]+ \
                                       self.pressure_tracker['test'][fold], 
                                       "Class": ['Train']*n_train+['Test']*n_test,
                                    "Prediction": res_train_predictions.tolist()+res_test_predictions.tolist(),
                                      "Truth": res_train_label.tolist()+res_test_label.tolist()}
                                     )
                save_fragment = '%s_code_%s_fold_%s_%s' %(self.model_tag, self.fp_code, fold, self.now)
                results_df.to_csv('results_%s.csv' %save_fragment, compression='gzip')
                print("Save Results using Fragment %s" %save_fragment)
                try:
                    model.save_model('%s.xgb' %save_fragment)
                except:
                    model.save('%s.h5' %save_fragment,save_format='h5')
            
            fold+=1
        print("\nBest fold is %s" %np.array(rmses).argmax())
        print("Average RMSEs of best epochs in each fold: %s" %np.mean(rmses))
        end = time.time()
        print('Set of Folds Done in %s' %(end-start))        
        if self.track:
            save_fragment = '%s_code_%s_%s' %(self.model_tag, self.fp_code, self.now)
            with open('file_tracker_%s.pkl' %save_fragment, 'wb') as f:
                pickle.dump(self.file_tracker, f)
        return np.mean(rmses)
    def train(self):
        hp_files = np.random.choice(self.fn, size=round(len(self.fn)*self.hp_frac), replace=False)
        hp_df = self.df[self.df['filename'].isin(hp_files)].reset_index().drop('index', axis=1)
        hp_df.to_csv('hp_df.csv', compression='gzip')
        print("Saved hp_df to disk")
        params = HPOpt(hp_df, self.features, self.property_used, self.target_mean, self.target_std, 
                      self.stacked, now=self.now, space=self.nn_space, grav_algo=self.grav_algo).get_params()
        print("Optimzed Hyperparameters found: %s" %params)
        mean_rmse = self.CV_objective(params)

class HPOpt:
    def __init__(self, df, features, property_used, target_mean, target_std, stacked, 
                 space, now, n_trees=50, grav_algo='xgb'):
        self.df = df
        self.space = space
        self.stacked = stacked
        self.property_used = property_used
        self.features = features
        self.target_mean = target_mean
        self.target_std = target_std
        self.n_trees = n_trees
        self.grav_algo = grav_algo
        self.now = now
        if stacked:
            self.N_CALLS = 20
            self.algo = 'nn'
        else:
            self.N_CALLS = 30
            self.algo = self.grav_algo
        
        print("Using %s calls for HPOpt" %self.N_CALLS)
    def get_params(self):
        HP_Inst = FpDataSet(self.df, self.features, self.property_used, self.target_mean, self.target_std, 
                            self.stacked, self.now, self.space, '0',1, self.grav_algo, track=False, 
                            chkpt_name='hp_model_checkpoint', n_folds=5)
        self.start = time.time()
        r = gp_minimize(HP_Inst.CV_objective, self.space, n_calls=self.N_CALLS)
        self.end = time.time()
        print("Finished HPOpt in %s" %(self.end-self.start))
        self.params = r.x
        return self.params