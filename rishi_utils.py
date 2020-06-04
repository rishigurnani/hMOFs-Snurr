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
        df=pd.read_csv(path)
    except:
        df=pd.read_csv(path, compression='gzip')
    drop_cols = [col for col in df.columns.tolist() if 'Unnamed' in col]
    return df.drop(drop_cols,axis=1)

def nan_cols(df):
    '''
    Return list of columns in df which contain NaN
    '''
    return df.columns[df.isnull().any()].tolist()
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
    l = list(l)
    l = ['"%s"'.replace('[','').replace(']','') %t for t in l]
    return ' '.join(l)

def pass_argparse_bool(val, flag):
    if val:
        return flag
    else:
        return ''

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

def all_positions(len_list, n_ones, zero_index=False):
    '''
    Return list of all unique combinations where n ones can arranged with len_list positions
    '''
    tup_zero_ind = list(itertools.combinations(range(len_list), n_ones))
    if zero_index:
        return [list(tup) for tup in tup_zero_ind]
    else:
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

def drawFromSmiles(s,molSize=(600,300),kekulize=True):
    from rdkit import Chem
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.Draw import DrawingOptions
    DrawingOptions.atomLabelFontSize = 100
    mol = MolFromSmiles(s)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

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
def similarity(a, b, n_bits=2048):

    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    if a is None or b is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=n_bits, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=n_bits, useChirality=False)
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

def struct_uniqueness(l,n_core=1):
    '''
    Find the structural uniqueness of each polymer in a list of polymers
    '''
    from joblib import Parallel, delayed
    mat = np.zeros((len(l), len(l)))
    data=[(ind1,ind2,i,j) for ind1,i in enumerate(l) for ind2,j in enumerate(l) if i<j]
    max_ops = (3500*3499)/2 #largest number of similarity calculations which will be performed
    if len(data) > max_ops:
        print('**Warning!** Large number of input polymers has triggered approximate, not exact, uniqueness computation')
        random.shuffle(data)
        data = data[:max_ops]
    def helper(x):
        ind1=x[0]
        ind2=x[1]
        i=x[2]
        j=x[3]
        return (ind1,ind2,similarity(i,j,1024))
    vals=Parallel(n_jobs=n_core)(delayed(helper)(x) for x in data)
    for ind1,ind2,sim in vals:
        mat[ind1][ind2] = sim
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
    Os = getIndexPositions(m0, 'O')
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

def fp_intersection(df, ref, non_fp_cols=[]):
    '''
    Return intersection of fingerprint between two dataframes
    '''
    return set([i for i in df.keys().tolist() if i not in non_fp_cols]).intersection(ref.keys().tolist())

def dropWithToleranceFromReference(df, ref, tolerance=.001,non_fp_cols=[]):
    fp_intersect=fp_intersection(df, ref, non_fp_cols)
    print("%s fingerprint columns intersect" %len(fp_intersect)) 
    np_df = df[list(fp_intersect)].to_numpy()

    tol = [tolerance]*len(fp_intersect)

    np_reduced = ref[list(fp_intersect)].to_numpy()
    keep_inds = []
    fp_vals = []
    for ind, row in enumerate(np_df):
        diffs = np.abs(np.asarray(row[None, :]) - np_reduced)
        matching_inds = np.nonzero((diffs <= tol).all(1))[0].tolist()
        if len(matching_inds) == 0:
            keep_inds.append(ind)
            fp_vals.append(row)
    print("%s Unique (in fingerprint space) rows have survived" %len(keep_inds))
    #new_polymers_df = pd.DataFrame({'SMILES': new_polymers, "Band gap": pvs})
    new_polymers_df = df.iloc[keep_inds]
    return new_polymers_df

def chunk_div(i, n):
    '''
    Divide i into n chunks which sum to i
    '''
    k, m = divmod(i, n)
    return [((i + 1) * k + min(i + 1, m)) - ((i) * k + min(i, m))  for i in range(n)]