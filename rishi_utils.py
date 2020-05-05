import pandas as pd
import sys
import argparse
import gzip
import os
import itertools
import numpy as np
import datetime

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
    Return list of all unique positions where n
    '''
    tup_zero_ind = list(itertools.combinations(range(len_list), n_ones))
    return [[i+1 for i in tup] for tup in tup_zero_ind]

def polymerize(s):
    '''
    Return all SMILES strings of molecule with 2 Hs replaced by *
    '''
    from rdkit import Chem
    mol = Chem.MolFromSmiles(s)
    mol_h = Chem.AddHs(mol)
    h_smiles = Chem.MolToSmiles(mol_h)
    n_h = h_smiles.count('[H]')
    position_list = all_positions(n_h, 2) #only want 2 H's replace by *
    poly_smiles = [set_repl(h_smiles, '[H]', "[*]", pos) for pos in position_list]
    return poly_smiles

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