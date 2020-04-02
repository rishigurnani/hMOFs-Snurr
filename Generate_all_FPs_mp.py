import pandas as pd

import numpy as np

import re

from os import listdir
from os.path import isfile, join, exists

from multiprocessing import Pool

import sys

import time

args = sys.argv

df_path = args[1]

fp_by_linker_path = args[2]

write_path = args[3]

string = args[4] #string which each Linker column contains, should be 'Smiles'

flag = args[5] #automated

n_core = int(args[6])

start = time.time()
#id:atom frequency
m_comp_d = {0: {"Zn": 4, "O": 1}, 1: {"Zn": 2}, 2: {"Cu": 2}, 3: {"V": 3, "O": 3}, 4: {"Zr": 6, "O": 6}}

#id:coordination #
m_coord_d = {0: 6, 1: 4, 2: 4, 3: 6, 4: 12}

#atomic radii (angstroms), affinity (eV), ionization potential (eV), electronegativity
element_d = {"Zn": [1.42, -.600, 9.39, 1.65], "O": [0.48, 1.461, 13.62, 3.44], "Cu": [1.45, 1.236, 7.73, 1.90], 
                "V": [1.71, .528, 6.75, 1.63], "Zr": [2.06, .433, 6.63, 1.33]}

def check6b(file):
    '''
    This function checks whether or not a file is in OutputGroup6b
    '''
    
    return exists('/home/rishi/Dropbox (GaTech)/EFRC_RamprasadGroup/hmof1/CIFs/outputGroup6b/' + file)





def whichNan(df):
    '''
    This function checks to see which columns of df contain a Nan value
    '''
    allowed_cols = ['CH4 v/v 5.8 bar',
 '5.8 bar err.',
 'CH4 v/v 65 bar',
 '65 bar err.',
 'CH4 v/v 248 bar',
 '248 bar err.']
    
    return [col for col in df.columns[df.isnull().any()].tolist() if col not in allowed_cols]





def getNAtoms(m_id):
    '''
    This function returns the number of atoms present in a metal ion
    '''
    
    return sum(m_comp_d[m_id].values())





def getMfp(m_id):
    '''
    This function takes in the metal id of an ion and returns the metallic fingerprint
    '''
    atom_freq = m_comp_d[m_id]
    
    n_a = getNAtoms(m_id)
    
    l = []
    
    for k in atom_freq.keys():
        l = l + [element_d[k] for i in range(atom_freq[k])]
    
    return [(m_coord_d[m_id] / n_a)] + list(np.mean(l, axis=0)) #valency descriptor plus others





def getMatches(df):
    '''
    This function returns a data frame containing all matches
    '''
    
    return df[df['Match'] == True]





def checkLinkerInd(i):
    
    d = {19: 30, 20: 31, 21: 32, 22: 33, 23: 34, 24: 35, 25: 36, 27: 37, 28: 38} #Maps Output Group 6b ID's with corresponding ID in 'fps'
    
    if i in [19, 20, 21, 22, 23, 24, 25, 27, 28]:
        return d[i]
    return False





def removeNASMiles(ls):
    '''
    This function removes 'NA' from ls
    '''
    
    return [s for s in ls if not pd.isna(s)]





def getLinkerFPs(row, linker_cols=None, fp_by_linker=None, flag='automated'):
    '''
    This function returns a list of numpy arrays containing [Linker A Fingerprint, Linker B Fingerprint] for the given row
    '''
    
    f = str(row['filename'])
    
    if flag != 'automated':
        la_id = row['Linker A ID']
        lb_id = row['Linker B ID']
        lc_id = row['Linker C ID']
        if check6b(f):
            if checkLinkerInd(la_id) != False:
                la_id = checkLinkerInd(la_id)


            if checkLinkerInd(lb_id) != False:
                lb_id = checkLinkerInd(lb_id)

        la_fp = fps.loc[la_id].drop(['Supp_ID', 'Smiles', 'clean_smiles'])
        lb_fp = fps.loc[lb_id].drop(['Supp_ID', 'Smiles', 'clean_smiles'])
        lc_fp = fps.loc[lc_id].drop(['Supp_ID', 'Smiles', 'clean_smiles'])
        linker_fps = [la_fp, lb_fp, lc_fp]
    else:
        smiles = [row[col] for col in linker_cols]
        smiles = removeNASMiles(smiles)
        
        linker_fps = [fp_by_linker[fp_by_linker['SMILES'] == s].drop(columns=['ID', 'SMILES']) for s in smiles]
        
    return [np.array(fp) for fp in linker_fps]





def getLinkerFP(row, linker_cols=None, fp_by_linker=None, flag='automated'):
    '''
    This function takes in a row from 'matches' and outputs its weighted FP
    '''
    
    linker_fps = getLinkerFPs(row, linker_cols, fp_by_linker, flag)
    if flag != "automated":
        n_a = row['# of Linker A']
        n_b = row['# of Linker B']
        n_c = row['# of Linker C']
        la_fp = fps[0] 
        lb_fp = fps[1]
        lc_fp = fps[2]
    
        return ((n_a*la_fp + n_b*lb_fp + n_c*lc_fp) / (n_a + n_b + n_c))
    else:
        n_linker = len(linker_fps)
        try:
            return (np.sum(np.stack(linker_fps, axis=0), axis=0) / n_linker).tolist()[0]
        except:
            print(row)
            print(linker_fps)
            raise ValueError





def getTotalFP(tup):

    i = tup[0]
    row = tup[1]
    linker_cols = tup[2]
    fp_by_linker = tup[3]
    flag = tup[4]
    
    m_id = int(row['Metal ID'])

    if i % 10 == 0:
        print("Completed MOF # ", i)
    return (i, np.concatenate((getLinkerFP(row, linker_cols, fp_by_linker, flag), getMfp(m_id))))





def mColNames():
    '''
    This function returns the names of the features related to the metal ions as a list
    '''
    
    return ['valence_pa', 'atomic_rad_pa_(angstroms)', 'affinity_pa_(eV)', 'ionization_potential_pa_(eV)', 'electronegativity_pa']





def colNames(level="metal", fps_of_linkers=None, flag='automated'):
    '''
    This function outputs all the feature names
    'pa' denotes that each metallic fp quantity is normalized by # of atoms in the ion
    '''
    if flag != 'automated':
        pg_cols = list(fps.columns)[3:]
    else:
        pg_cols = list(fps_of_linkers.columns)[2:]
    if level=='linker':
        return pg_cols
    else:
        return pg_cols + mColNames()





def makeFeatureDF(df, fps_of_linkers, flag='automated'):
    '''
    This function returns a copy of 'matches' with several blank columns for each of the features
    '''
    if flag != 'automated':
        f_df = getMatches(df)
    else:
        f_df = df
    for feature in colNames(fps_of_linkers=fps_of_linkers, flag=flag):
        f_df[feature] = np.nan
    
    return f_df





def getNullRows(df):
    '''
    This function returns the entire row of a df containing a null value 
    '''
    return df[df.isnull().any(axis=1)]





def getAutoLinkerCols(df, string):
    return [col for col in df.columns if string in col]





def splitBySuccess(fp_map, mofs, string):
    '''
    This function will return a split fps into successess and failures and MOFs into successess and failures.
    'string' is the substring of all columns containing SMILES strings in 'mofs'
    '''


    null_fps = fp_map[fp_map.isnull().any(axis=1)]


    null_smiles = list(null_fps['SMILES'])

    successful_fps = fp_map.dropna(thresh=1) 


    linker_cols = getAutoLinkerCols(mofs, string)


    failed_mofs = mofs[(mofs[linker_cols].isin(null_smiles).sum(axis=1) > 0) | (mofs['L0_Smiles'].isna())]


    successful_mofs = mofs[(mofs[linker_cols].isin(null_smiles).sum(axis=1) == 0) & (mofs['L0_Smiles'].notnull())]


    return null_fps, successful_fps, failed_mofs, successful_mofs.reset_index()





def main():
    '''
    This function outputs a df containing the weighted linker features for all matches
    '''
    matches = getMatches()
    f_df = makeFeatureDF()
    n_row = len(matches)
    
    for i in range(n_row):
        
        f_df.at[f_df.index == f_df.index[i], 7:] = getFP(f_df[f_df.index == f_df.index[i]].T.squeeze())
    
    return f_df





def main1(df, linker_cols, fp_by_linker, flag):
    '''
    This function outputs a df containing the weighted linker + metal features for all matches
    '''
    #matches = getMatches()
    try:
        df_drop_cols = [col for col in df.columns if 'Unnamed' in col]
        fp_drop_cols = [col for col in fp_by_linker.columns if 'Unnamed' in col]
        
        df = df[df.drop(df_drop_cols)]
        fp_by_linker = fp_by_linker[fp_by_linker.drop(fp_drop_cols)]
        
    except:
        pass

    f_df = makeFeatureDF(df, fp_by_linker, flag)
    n_row = len(f_df)
    if flag != 'automated':
        start_ind = 9
    else:
        start_ind = len(linker_cols) + 5
    
    repl_cols = list(f_df.columns)[start_ind:]
    
    data = [(i, f_df[f_df.index == f_df.index[i]].T.squeeze(), linker_cols, fp_by_linker, flag) for i in range(n_row)]
        
    pool = Pool(n_core)
    result = pool.map(getTotalFP, data)
    pool.close()
    pool.join()
    
    for j in range(n_row):
    
        data_point = result[j]
        i = data_point[0]
        tmp = data_point[1]
        try:
            f_df.loc[i, repl_cols] = tmp
        except:
            print("i: ", i)
            print("j: ", j)
            print("RHS len: ", len(tmp))
            print("LHS len: ", len(repl_cols))
            print("This is tmp: ", tmp)
    
    f_df.to_csv(write_path)  

def main2(df, linker_cols, fp_by_linker, flag):
    '''
    This function outputs a df containing the weighted linker + metal features for all matches Uses faster method for creating df
    '''
    #matches = getMatches()
    #fp_by_linker = fp_by_linker.dropna() #drop SMILES where fingerprinting failed
    try:
        df_drop_cols = [col for col in df.columns if 'Unnamed' in col]
        fp_drop_cols = [col for col in fp_by_linker.columns if 'Unnamed' in col]
        
        df = df.drop(df_drop_cols, axis=1)
        fp_by_linker = fp_by_linker.drop(fp_drop_cols, axis=1)
        
    except:
        pass
    
    null_fps, fp_by_linker, failed_mofs, df = splitBySuccess(fp_by_linker, df, string)

    f_df = makeFeatureDF(df, fp_by_linker, flag)
    n_row = len(f_df)
    if flag != 'automated':
        start_ind = 9
    else:
        start_ind = len(linker_cols) + 4
    
    repl_cols = list(f_df.columns)[start_ind:]
    
    data = [(i, f_df[f_df.index == f_df.index[i]].T.squeeze(), linker_cols, fp_by_linker, flag) for i in range(n_row)] #create data for pools
        
    #spread and collect jobs    
    pool = Pool(n_core)
    result = pool.map(getTotalFP, data)
    pool.close()
    pool.join()
    
    ll = [[0]*n_row for i in repl_cols]
    
    for j in range(n_row):
        data_point = result[j]
        i = data_point[0]
        tmp = data_point[1]
        for ind, k in enumerate(tmp):
            #ll[ind][i] = float(k)
            ll[ind][i] = k
    
    f_df = f_df.drop(repl_cols, axis=1)
    d = f_df.to_dict()
    
    for ind, k in enumerate(ll):
        d[repl_cols[ind]] = pd.Series(k)
    
    new_df = pd.DataFrame(d)
    new_df.to_csv(write_path)         
    

    
df = pd.read_csv(df_path)
fp_by_linker = pd.read_csv(fp_by_linker_path)            
     
linker_cols = getAutoLinkerCols(df, string)

main2(df, linker_cols, fp_by_linker, flag)

end = time.time()

print("Elapsed time: ", end-start)


