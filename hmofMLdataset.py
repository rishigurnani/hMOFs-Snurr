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

from skopt import gp_minimize

from multiprocessing import Pool

class hmofMLdataset:
    def __init__(self, results_dir, now, SI_grav_data_path='/data/rgur/efrc/prep_data/all_v1/ml_data.csv', 
                 SD_grav_data_path='/data/rgur/efrc/prep_data/all_no_norm/ml_data.csv',SI_stacked_path=
                '/data/rgur/efrc/prep_data/all_v1/stacked.csv',
                 SD_stacked_path='/data/rgur/efrc/prep_data/all_no_norm/stacked.csv',
                 Y_DATA_PATH='/data/rgur/efrc/data_DONOTTOUCH/hMOF_allData_March25_2013.xlsx', n_core=15, skip=[], 
                 job_dict=None, nn_space=None, grav_algo='xgb'):
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
        self.job_dict = job_dict
        if self.job_dict != None:
            self.feature_codes = list(job_dict.keys())
        print("There are %s unique feature codes" %len(set(self.feature_codes)))
        self.any_stacked = any([item[-1]=='1' for item in self.feature_codes]) #are any codes for stacked models?
        self.now = now
        print("now is %s" %self.now)
        self.nn_space = nn_space
        self.grav_algo = grav_algo
        
    
    def makeMasterDFs(self):
        #gravimetric
        self.grav, self.grav_prop, self.grav_target_mean, self.grav_target_std, self.grav_all_features = ml.prepToSplit(
                                            self.grav_algo, self.cat_si_sd, self.SD_grav_data_path, self.SI_grav_data_path, 
                                            self.grav_start_str_sd, self.grav_end_str_sd, self.start_str_si, 
                                            self.end_str_si, 1, self.del_defective_mofs, self.add_size_fp, 
                                            self.srt_size_fp, None, stacked=False, n_core=self.n_core, 
                                            del_geometric_fp=False, cat_col_names=self.cat_col_names, 
                                            Y_DATA_PATH=self.Y_DATA_PATH)
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
                try:
                    TRAIN_GRID, SEEDS = self.job_dict[i]
                except:
                    print('Failed to Find Code in job_dict')
                    TRAIN_GRID = [.5, .6, .7, .8, .9]
                    SEEDS = [0, 10, 20]
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
                    N_CORE=len(TRAIN_GRID)*len(SEEDS)
                if STACKED:
                    FpDataSet(self.iso.drop(drop_features, axis=1), run_features, self.iso_prop, 
                              self.iso_target_mean, self.iso_target_std, now=self.now, nn_space=self.nn_space, stacked=STACKED,
                              fp_code=CODE, n_core=N_CORE, grav_algo=self.grav_algo, 
                              rand_seeds=SEEDS, train_grid=TRAIN_GRID).run()
                else:
                    FpDataSet(self.grav.drop(drop_features, axis=1), run_features, self.grav_prop, 
                              self.grav_target_mean, self.grav_target_std, now=self.now, nn_space=self.nn_space,
                              stacked=STACKED, fp_code=CODE, n_core=N_CORE, grav_algo=self.grav_algo, 
                              rand_seeds=SEEDS, train_grid=TRAIN_GRID).run()

class FpDataSet:
    def __init__(self, df, features, property_used, target_mean, target_std, stacked, fp_code, now, nn_space, PCA_DIM=400, 
                 rand_seeds=[0, 10, 20], train_grid = [.5, .6, .7, .8, .9], n_core=15, grav_algo='xgb'):
        self.now = now
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
        self.n_core = n_core
        self.grav_algo = grav_algo
        if self.stacked:
            self.algo = 'nn'
            self.model_tag = 'iso'
        else:
            self.algo = self.grav_algo
            self.model_tag = 'grav'
        self.train_grid = train_grid
        self.nn_space = nn_space
    
    def sortRemoteInds(self):
        '''
        Create list of most remote indices
        '''
        n_core = 1
        use_pca = True
        nn_tree, filenames = ml.NNtree(self.df, self.stacked, n_core, self.features, use_pca, self.PCA_DIM)
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
                                                   self.stacked, self.nn_space, self.now, 
                                                   grav_algo=self.grav_algo).makeResults()
        save_fragment = '%s_code_%s_train_%s_seed_%s_%s' %(self.model_tag, self.fp_code, TRAIN_PCT, seed, self.now)
        print("Save Results using Fragment %s" %save_fragment)
        results_df.to_csv('results_%s.csv' %save_fragment, compression='gzip')
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
        Parallel(n_jobs=self.n_core)(delayed(self.helper)(j) for j in self.parallel_data)

class trainTestSplit:
    def __init__(self, df, train_pct, features, property_used, target_mean, target_std, seed, remote_info, stacked, nn_space,
                 now, hp_frac=.1, n_trees=5000, grav_algo='xgb'):
        self.df = df
        self.filenames = df['filename'].unique().tolist()
        self.n_samples = len(self.filenames)
        self.train_pct = train_pct
        #print('train_pct %s' %self.train_pct)
        self.seed = seed
        self.target_mean = target_mean
        self.target_std = target_std
        self.remote_info = remote_info
        #self.pct_remote = self.train_pct - .5
        self.pct_remote = 0
        print('pct_remote %s' %self.pct_remote)
        self.n_remote = round(self.n_samples*self.pct_remote)
        #print('n_remote %s' %self.n_remote)
        self.n_train = round(self.n_samples*self.train_pct)
        #print('n_train %s' %self.n_train)
        self.n_random = self.n_train - self.n_remote
        #print('n_random %s' %self.n_random)
        self.stacked = stacked
        self.features = features
        self.property_used = property_used
        self.hp_frac = hp_frac
        self.grav_algo = grav_algo
        self.now = now
        if stacked:
            self.hp_frac = .05
        print('hp_frac %s' %self.hp_frac)
        self.n_trees = n_trees
        self.nn_space = nn_space

        if self.stacked:
            self.algo = 'nn'
            self.model_tag = 'iso'
        else:
            self.algo = self.grav_algo
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
        self.train_fn_order = train_df['filename'].tolist()
        self.test_fn_order = test_df['filename'].tolist()
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
            if self.stacked:
                train_pressures = train_df['pressure'].tolist()
                test_pressures = test_df['pressure'].tolist()
            else:
                train_pressures = None
                test_pressures = None
            return (train_fp, train_label.to_numpy(), train_files, train_pressures), (test_fp, test_label.to_numpy(), \
                    test_files, test_pressures), train_label, test_label
        
    def hp_opt(self):
        start = time.time()
        if self.algo == 'nn':
            self.max_batch = 512
            if self.max_batch > self.n_train:
                self.max_batch = self.n_train // 2
            if self.nn_space == None:
                self.space = [(100, 400), #n_units
                (.0005, .003),#learning rate
                (2, 15), #patience
                (12, self.max_batch), #batch size
                (.01, .6)] #validation split
            else:
                self.space = self.nn_space #0:n_units, 1:learning rate, 2:patience, 3:batch size, 4:validation split
                self.min_batch = self.space[3][0]
                if self.min_batch > self.max_batch:
                    self.min_batch = self.max_batch // 2
                self.space[3] = (self.min_batch, self.max_batch)
        else:
            self.max_depth_ub = 15 #max depth upper bound
            if self.n_train < 200:
                self.max_depth_ub = 4
            self.space = [(.3, .95), #colsample_bytree
                            (.01, .5),#learning_rate
                            (2, 15), #max_depth
                            (1, 20)] #alpha
            
        print("Space is %s" %self.space)
        hp_df = self.df.sample(frac=self.hp_frac, random_state=self.seed)
        hp_remote_info = [x for x in self.remote_info if x[1] in hp_df['filename'].tolist()]
        opt_hps = HPOpt(hp_df, self.train_pct, self.features, self.property_used, self.target_mean, 
                            self.target_std, self.seed, hp_remote_info, self.stacked, self.space, self.now,
                            grav_algo=self.grav_algo).get_params()
        end = time.time()
        print("Time Elapsed during HPOpt: %s" %(end-start) )
        return opt_hps
    
    def run_model(self):
        self.params = self.hp_opt()
        print("Optimzed Hyperparameters found: %s" %self.params)
        self.train_d, self.test_d, self.train_label, self.test_label = self.split()
        self.MODEL = ml.run_model(self.algo, self.train_d, self.n_trees, self.params, 
                                  chkpt_name='model_checkpoint_%s' %self.now)
        
        
    def makeResults(self):
        self.run_model()
        if self.algo=='xgb':
            test_predictions = self.MODEL.predict(self.test_d)
            train_predictions = self.MODEL.predict(self.train_d)
            pressures = [35]*(self.n_samples)
            files = self.train_fn_order + self.test_fn_order
        if self.algo=='nn':
            test_fp = self.test_d[0]
            train_fp = self.train_d[0]
            files = self.train_d[2] + self.test_d[2]
            test_predictions = self.MODEL.predict(test_fp).flatten()
            train_predictions = self.MODEL.predict(train_fp).flatten()
            if self.stacked:
                pressures = self.train_d[3] + self.test_d[3]
            else:
                pressures = [35]*(len(test_predictions)+len(train_predictions))
            
        res_test_predictions, res_test_label, res_train_label, res_train_predictions = ml.unscale(self.property_used, 
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
                 space, now, n_trees=50, grav_algo='xgb'):
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
        self.grav_algo = grav_algo
        self.now = now
        if stacked:
            self.N_CALLS = 40
            self.algo = 'nn'
        else:
            self.N_CALLS = 125
            self.algo = self.grav_algo
        
        print("Using %s calls for HPOpt" %self.N_CALLS)
        self.train_pct = train_pct
    
    def objective(self, params):
        MODEL = ml.run_model(self.algo, self.train_d, self.n_trees, params, chkpt_name='hp_model_checkpoint_%s' %self.now)
        return ml.model_rmse(MODEL, self.train_d, self.test_d, self.stacked, self.algo, self.target_mean, 
                             self.target_std, self.property_used, self.test_label, self.train_label, save=False, 
                             fname=None, subset_inds=None)
    def get_params(self):
        HP_TTS = trainTestSplit(self.df, self.train_pct, 
                                                        self.features, self.property_used, self.target_mean,
                                                        self.target_std, self.seed, self.remote_info, 
                                                        self.stacked, nn_space=None, now=self.now, grav_algo=self.grav_algo)
        self.train_d, self.test_d, self.train_label, self.test_label = HP_TTS.split()
        start = time.time()
        r = gp_minimize(self.cv_objective, self.space, n_calls=self.N_CALLS, random_state=self.seed)
        end = time.time()
        self.params = r.x
        return self.params

# Run

# global now
# now = datetime.datetime.now().strftime("%I_%M%p_on_%B_%d_%Y")
# hmofMLdataset('/data/rgur/efrc/ml/results/', 
#                    SI_grav_data_path='/data/rgur/efrc/prep_data/all_v1/ml_data.csv', 
#                  SD_grav_data_path='/data/rgur/efrc/prep_data/all_no_norm/ml_data.csv',SI_stacked_path=
#                 '/data/rgur/efrc/prep_data/all_v1/stacked.csv',
#                  SD_stacked_path='/data/rgur/efrc/prep_data/all_no_norm/stacked.csv', skip=['10000', '11000', '01000', '10100', '11100', '01100', '10010', '11010', '01010', '10110', '11110', '01110', '10001'], n_core=15).makeAllResults()