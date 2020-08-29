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
try:
    from rdkit import Chem
except:
    pass
try:
    from pylab import cm
except:
    pass
try:
    from matplotlib.cm import ScalarMappable
    import matplotlib as mpl
except:
    pass

sys.path.append('/home/appls/machine_learning/PolymerGenome/src/common_lib')
import auxFunctions as aF

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
        sys.stdout.flush()
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
    '''
    Get names of all files in parent_dir
    '''
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
    Return intersection of PG fingerprint between two dataframes
    '''
    ref_cols = [x for x in ref.keys().tolist() if x not in non_fp_cols]
    return set(df.keys().tolist()).intersection(ref_cols)

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

def dropWithTolerancePandas(df, cols_to_consider, tolerance=.001, KEEP='first'):
    '''
    Drop all duplicates from data frame
    '''
    #drop duplicates with tolerance
    DECIMALS = len(str(tolerance)) - 2
    df = df.round(decimals=DECIMALS)
    if KEEP == 'first':
        df = df.sort_values(by='Epoch') #sort so that lower epoch will be kept
    return df.drop_duplicates(subset=cols_to_consider,keep=KEEP)

def dropWithToleranceFromReferencePandas(df,ref,non_fp_cols=[],tolerance=.001,KEEP=False):
    #Keep=False if we don't want to keep ANY duplicates
    fp_intersect=fp_intersection(df, ref, non_fp_cols)
    print("%s fingerprint columns intersect" %len(fp_intersect))  
    df['Class'] = ['New']*len(df)
    ref['Class'] = ['Old']*len(ref)
    ref['ID'] = ['None']*len(ref)   
    
    keep_cols = fp_intersect.union(['Class','ID'])
    concat = pd.concat([df[keep_cols],ref[keep_cols]],ignore_index=True)
    concat_drop = dropWithTolerancePandas(concat,fp_intersect,tolerance,KEEP=KEEP)
    keep_ids = concat_drop[concat_drop['Class']=='New']['ID'].tolist()
    return df[df['ID'].isin(keep_ids)].drop('Class',axis=1) #keep new and drop Class col

def chunk_div(i, n):
    '''
    Divide i into n chunks which sum to i
    '''
    k, m = divmod(i, n)
    return [((i + 1) * k + min(i + 1, m)) - ((i) * k + min(i, m))  for i in range(n)]

def canonicalizeAndRemoveH(s):
    '''
    Convert SMILES string, s, into a canonicalized, PG valid, SMILES with all H removed
    '''
    try:
        tmp = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        return tmp.replace('*','[*]')
    except:
        print('***Warning*** Canonicalization failed')
        return None

def multiplySmiles(s,n):
    '''
    Make Chiho's code robust to n=1 case
    '''
    if n==1:
        return s.replace('*','H')
    else:
        return aF.v2_multiply_smiles_star(s, number_of_multiplications = n, debug=0)['extended_smiles']

def monomer_join(a,b):
    '''
    ***NOT YET RELIABLE*** use with extreme caution.
    Helper function for copolymerize
    '''
    if a.count('([*])') == 1:
        mod_mol = Chem.ReplaceSubstructs(Chem.MolFromSmiles(a.replace('[*]','*').replace('(*)','(**)')), 
                                     Chem.MolFromSmiles('**'), 
                                     Chem.MolFromSmiles(b),
                                     replaceAll=True)
    elif a.count('([*])') == 2:
        a_repl = a.replace('([*])','(*)',1).replace('([*])','(**)')
        mod_mol = Chem.ReplaceSubstructs(Chem.MolFromSmiles(a_repl), 
                                     Chem.MolFromSmiles('**'), 
                                     Chem.MolFromSmiles(b),
                                     replaceAll=True)         
    else:
        mod_mol = Chem.ReplaceSubstructs(Chem.MolFromSmiles(set_repl(a.replace('[*]','*'),'*','**',[2])), 
                                     Chem.MolFromSmiles('**'), 
                                     Chem.MolFromSmiles(b),
                                     replaceAll=True)        
    return set_repl(Chem.MolToSmiles(mod_mol[0]),'*','',[2]).replace('*','[*]').replace('()','')

def monomer_join2(a,b):
    '''
    ***NOT YET RELIABLE*** use with extreme caution.
    Helper function for copolymerize. Clean version. Yields better results than monomer_join.
    '''
    a_repl = set_repl(a.replace('[*]','*'),'*','**',[2])
    mod_mol = Chem.ReplaceSubstructs(Chem.MolFromSmiles(a_repl), 
                                     Chem.MolFromSmiles('**'), 
                                     Chem.MolFromSmiles(b),
                                     replaceAll=True)     
    return set_repl(Chem.MolToSmiles(mod_mol[0]),'*','',[2]).replace('()','').replace('==','=').replace('*','[*]')

def copolymerize(a,b,sequence,joiner=monomer_join2):
    '''
    ***NOT YET RELIABLE*** use with extreme caution.
    Return the valid PG polymer SMILES of copolymer of a and b with the given sequence.
    a must be valid PG polymer SMILES
    b must be valid PG polymer SMILES
    sequence must be list of characters containing only 'a' and 'b' strings
    '''
    print('Number of units in sequence: %s'%len(sequence))
    cmd = 'joiner(%s,%s)' %(sequence[0],sequence[1])
    rest_sequence = sequence[2:]
    for monomer in rest_sequence:
        cmd = 'joiner(' + ','.join([cmd,monomer]) + ')'
    out=eval(cmd)
    return out


def rishi_multiplySmiles(s,n,molecule=False):
    '''
    ***NOT YET RELIABLE*** use with extreme caution.
    '''
    if s.count('*') > 2:
        raise ValueError('Too many connection points!')
    elif s.count('*') == 0:
        s = '[*]'+s+'[*]'
    poly_repeat = copolymerize(s,s,sequence=['a']*n)
    if molecule:
        return poly_repeat.replace('*','H')
    else:
        return poly_repeat

def nAtoms(ls):
    '''
    Return number of atoms in SMILES string (w Hydrogen)
    '''
    if type(ls)==str:
        ls = list(ls)
    return [Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms() - s.count('*') for s in ls]
   
def mol_with_atom_index(mol):
    '''
    Label each atom in mol with index
    '''
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def frequency_dict(my_list): 
    '''
    Return frequency dictionary from list
    '''
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1

    return freq

def _main_chain_len(s):
    mol = Chem.MolFromSmiles(s)
    star_inds = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            star_inds.append(atom.GetIdx())
    return Chem.GetDistanceMatrix(mol)[star_inds[0]][star_inds[1]]

def main_chain_len(ls):
    '''
    Return length of main chain from list of polymer SMILES
    '''
    if type(ls)==str:
        ls = list(ls)    
    return [_main_chain_len(s) for s in ls]

def pltDensity(property_X,property_Y,N_BINS=100,cmapName='plasma_r',order='random',count_thresh=None):
    '''
    N_BINS: Number of bins per dimension
    cmapName: Matplotlib name of color map
    '''
    min_x = .9*min(property_X)
    max_x = 1.1*max(property_X)

    min_y = .9*min(property_Y)
    max_y = 1.1*max(property_Y)

    x_bins = np.linspace(min_x,max_x,N_BINS)
    x_bins

    y_bins = np.linspace(min_y,max_y,N_BINS)

    def find_index(x,y,x_bins,y_bins,N_BINS):
        x = float(x)
        y = float(y)
        for ind,i in enumerate(x_bins):
            if x < i:
                x_bin = ind-1
                break
        for ind,i in enumerate(y_bins):
            if y < i:
                y_bin = ind-1
                break
        return x_bin + N_BINS*y_bin

    bin_nums = [find_index(x,y,x_bins,y_bins,N_BINS) for (x,y) in zip(property_X,property_Y)] #associate each data point with a bin

    bin_frequency = frequency_dict(bin_nums) #number of data points in each bin

    count_data = np.array([bin_frequency[x] for x in bin_nums]) #associate each data point with the number of other points occupying its bin
    if count_thresh != None:
        count_data = np.minimum(count_data,count_thresh)
    if order != 'random':
        if order == 'dense_top':
            REVERSE = False
        else:
            REVERSE = True

        inds_order = [y[0] for y in sorted(enumerate(count_data),key=lambda x: x[1],reverse=REVERSE)]
        count_data = [count_data[i] for i in inds_order]
        property_X = [property_X[i] for i in inds_order]
        property_Y = [property_Y[i] for i in inds_order]
        
    max_val = max(count_data) - 1

    colors2 = cm.get_cmap(cmapName,max_val)

    c=np.array([colors2(x-1) for x in count_data]) #color of each data point. Count=1 has 'lowest' color.
    #sm = ScalarMappable(cmap=colors2,norm=mpl.colors.Normalize(vmin=min(count_data), vmax=max_val))
    sm = ScalarMappable(cmap=colors2,norm=mpl.colors.Normalize(vmin=0, vmax=max_val))
    sm.set_array([])
    tick_vals = np.round(np.linspace(0,max_val,6))
    tick_labels = tick_vals.astype('int').astype('str')
    if count_thresh != None:
        tick_labels[-1] = '>%s' %count_thresh
    
    return property_X, property_Y, c, sm, count_data

def checkSubstrings(l,s):
    '''
    Check if any strings in list l are substrings of s
    '''
    return any([i in s for i in l])

def getPGCols(df):
    '''
    Get PG cols from DataFrame
    '''
    cols = df.columns.tolist()
    return [col for col in cols if checkSubstrings(['afp','mfp','efp','bfp'],col)]