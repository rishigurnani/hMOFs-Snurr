import pandas as pd
import sys
import argparse
import gzip
import os
from os import listdir
from os.path import isfile, join
import itertools
import numpy as np
import datetime
import random

def pd_load(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, compression='gzip')

def compress_gzip(compress):
    '''
    Compress all files in the list 'compress'
    '''
    for path in compress:
        print('Compressing %s' %path)
        df = pd.read_csv(path)

        df.to_csv(path, compression='gzip')
     
def count_lines(filename):
    '''
    Count lines in file for gzip or not gzip
    '''
    try: #normal
        with open(filename, 'r') as handle:
            n_lines = sum(1 for row in handle)
    except: #gzip
        with gzip.open(filename, 'rb') as handle:
            n_lines = sum(1 for row in handle)
          
    return n_lines

def write_list_to_file(l, path):
    with open(path, "w") as outfile: 
        outfile.write("\n".join(l))
        
def mkdir_existOk(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def pass_argparse_list(l):
    l = ['"%s"' %t for t in l]
    return ' '.join(l)

def set_repl(s, sub, repl, ns):
    '''
    Replace a set of specific instances
    '''
    if any([i > s.count(sub) for i in ns]):
        raise ValueError('Too many replacement indices given')
    ns = sorted(ns)
    for ind, n in enumerate(ns):
        n -= ind #adjust n to reflect fact that ind # of sub have already been replaced
        find = s.find(sub)
        # If find is not -1 we have found at least one match for the substring
        i = find != -1
        # loop util we find the nth or we find no match
        while find != -1 and i != n:
            # find + 1 means we start searching from after the last match
            find = s.find(sub, find + 1)
            i += 1
        # If i is equal to n we found nth match so replace
        if i == n:
            s = s[:find] + repl + s[find+len(sub):]
    return s

def all_positions(len_list, n_ones):
    '''
    Return list of all unique combinations where n ones can arranged with len_list positions
    '''
    tup_zero_ind = list(itertools.combinations(range(len_list), n_ones))
    return [[i+1 for i in tup] for tup in tup_zero_ind]

def polymerize(s,n=None):
    '''
    Return n random SMILES strings of molecule with 2 Hs replaced by *
    '''
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(s)
        mol_h = Chem.AddHs(mol)
        h_smiles = Chem.MolToSmiles(mol_h)
        n_h = h_smiles.count('[H]')
        position_list = all_positions(n_h, 2) #only want 2 H's replace by *
        if n != None and len(position_list)>n:
            random.shuffle(position_list)
            position_list = position_list[:n]
        poly_smiles = [set_repl(h_smiles, '[H]', "[*]", pos) for pos in position_list]
        return poly_smiles
    except:
        return None

def depolymerize(s):
    return s.replace('[*]', '[H]')

def drawFromSmiles(s, img_size=(300,300), addH=False):
    from rdkit import Chem
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles(s, sanitize=True)
    if addH:
        mol = Chem.AddHs(mol)
    return Draw.MolToImage(mol, size=img_size)

def avg_max_n_pct_error(true, pred, frac=.01):
    '''
    Take in two lists and percent error
    '''
    srt_abs_error = np.sort(np.abs(np.subtract(true, pred)))[::-1]
    n_keep = round(frac*len(true))
    return (sum(srt_abs_error.tolist()[:n_keep]) / n_keep)

def getIndexPositions(listOfElements, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    indexPosList = []
    indexPos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            indexPos = listOfElements.index(element, indexPos)
            # Add the index position in list
            indexPosList.append(indexPos)
            indexPos += 1
        except ValueError as e:
            break
 
    return indexPosList

def sanitize_dir_path(p):
    '''
    Add '/' to path if it doesn't already exist
    '''
    if p[-1] != '/':
        p += '/'
    return p

def add_timestamp(s):
    now = datetime.datetime.now().strftime("%I_%M%p_on_%B_%d_%Y")
    return s+now

def str2bool(s):
    if s == 'False':
        return False
    elif s == 'True':
        return True
    else:
        raise ValueError("cannot convert string")


# Tanimoto similarity function over Morgan fingerprint
def similarity(a, b):

    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    if a is None or b is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.

    a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
    for i != j.
    """
    return a + a.T - np.diag(a.diagonal())

def struct_uniqueness(l):
    '''
    Find the structural uniqueness of each polymer in a list of polymers
    '''
    mat = np.zeros((len(l), len(l)))
    for ind1,i in enumerate(l):
        for ind2,j in enumerate(l):
            if i<j:
                mat[ind1][ind2] = similarity(i,j)
    mat = symmetrize(mat)
    return np.subtract(np.ones(len(l)), np.max(mat, axis=0))

def tf_make_dataset(X_data,y_data,n_splits):
    import tensorflow as tf
    def gen():
        for train_index, test_index in KFold(n_splits).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train,y_train,X_test,y_test

    return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))

def CIRconvert(ids):
    from urllib.request import urlopen
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + ids + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'

def smiles_rxn1(s):
    '''
    Turn molecules smile into polymers!
    '''
    monomers = s.split('.')
    m1 = monomers[1].replace('Cl', '', 1)
    m1 = m1.replace('Cl', '[*]')
    m0 = monomers[0]
    Os = ru.getIndexPositions(m0, 'O')
    if Os[0] == 0:
        add = 0
    else:
        add = 1
    return m0[:Os[0]+add]+'[*]'+m0[Os[0]+add:Os[1]+1]+m1+m0[Os[1]+1:]

def getAllFilenames(parent_dir):
    return [parent_dir+f for f in listdir(parent_dir) if isfile(join(parent_dir, f))]

def eq_space(x, y, n, force_int=False):
    '''
    Return n equally-spaced values between x and y
    '''
    step = (y - x) / (n - 1)
    if force_int:
        return [int(x + step * i) for i in range(n)]
    return [x + step * i for i in range(n)]

def df_split(a, n_chunks):
    '''
    Split a df, a, into n_chunks
    '''
    return np.array_split(a,n_chunks)

def worker_number(Process):
    s = str(Process)
    return int(s.split('-')[1].split(',')[0])

def flatten_ll(l):
    '''
    Flatten list of list
    '''
    return [item for sublist in l for item in sublist]

def alphabetize(df):
    '''
    Return df with columns arranged alphabetically
    '''
    return df.reindex(sorted(df.columns), axis=1)