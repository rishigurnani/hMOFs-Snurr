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

try:
    from sklearn.externals import joblib
except:
    pass

sys.path.append('/home/appls/machine_learning/PolymerGenome/src/common_lib')
import auxFunctions as aF
from rdkit.Chem import rdmolfiles
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600

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

def multiplySmiles(s,n,mol=True):
    '''
    Make Chiho's code robust to n=1 case. mol=True means the connection points will be replaced by [*]
    '''
    if n==1:
        if mol==True:
            return s.replace('*','H')
        else:
            return s
    else:
        if mol==True:
            return aF.v2_multiply_smiles_star(s, number_of_multiplications = n, debug=0)['extended_smiles']
        else:
            return aF.v2_multiply_smiles_star(s, number_of_multiplications = n, debug=0)['At_extended_smiles_At'].replace('Bi','*')

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

def mol_without_atom_index(mol):
    '''
    Label each atom in mol with index
    '''
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def mol_with_partial_charge(mol,supress_output=False):
    '''
    Label each atom in mol with partial charge
    '''
    mol = mol_with_atom_index(mol)
    Chem.AllChem.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    if not supress_output:
        for ind,i in enumerate(contribs):
            print('Charge of atom %s is %s' %(ind,i))
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='jet', contourLines=10)
    #return fig

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


def get_star_inds(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    star_inds = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            star_inds.append(atom.GetIdx())
    return star_inds

def get_connector_inds(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    star_inds = get_star_inds(mol)
    connector_inds = []
    for ind in star_inds:
        star=mol.GetAtoms()[ind]
        connector=star.GetNeighbors()[0]
        connector_inds.append(connector.GetIdx())
    return connector_inds
    

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

class PeriodicMol(Chem.rdchem.Mol):
    def __init__(self, mol, star_inds, connector_inds):
        self.mol = mol
        self.connector_inds = []
        for i in connector_inds:
            srt=np.sort(star_inds + [i])[::-1] #sort in reverse order
            new_ind = i - int(np.argwhere(srt==i))
            self.connector_inds.append(new_ind)
    def HasSubstructMatch(self,match_mol):
        try:
            return self.mol.HasSubstructMatch(match_mol)
        except:
            self.GetSSSR()
            return self.mol.HasSubstructMatch(match_mol)

    def GetSubstructMatches(self,match_mol):
        try:
            return self.mol.GetSubstructMatches(match_mol)
        except:
            self.GetSSSR()
            return self.mol.GetSubstructMatches(match_mol)

    def GetSubstructMatch(self,match_mol):
        try:
            return self.mol.GetSubstructMatch(match_mol)
        except:
            self.GetSSSR()
            return self.mol.GetSubstructMatch(match_mol)     
    def GetSSSR(self):
        Chem.GetSSSR(self.mol)

    def GetRingInfo(self):
        return self.mol.GetRingInfo()


class LinearPol(Chem.rdchem.Mol):
    def __init__(self, mol,SMILES=None):
        '''
        SMILES = the smiles string of the parent polymer
        ''' 
        if type(mol) == str:
            self.mol = Chem.MolFromSmiles(mol)
            self.SMILES = mol
        else:
            self.mol = mol
            if SMILES == None:
                self.SMILES = Chem.MolToSmiles(self.mol)
            else:
                self.SMILES = SMILES
        self.star_inds = get_star_inds(self.mol) #these are always sorted from smallest index to largest index
        self.connector_inds = self.get_connector_inds() #these are always sorted from smallest index to largest index
        #self.main_chain_atoms, self.side_chain_atoms = self.get_main_chain()
        self.main_chain_atoms, self.side_chain_atoms = None,None
        
        self.alpha_atoms = None
        self.beta_atoms = None #need to implement
        
    
    def get_main_chain(self):
        '''return (x,y) where x are the main chain atoms and y are the rest (i.e. side chain atoms)'''
        main_inds_no_ring=set([x for x in Chem.GetShortestPath(self.mol,self.star_inds[0],self.star_inds[1])])
        inds = main_inds_no_ring.copy()
        ri = self.mol.GetRingInfo()
        self.main_chain_rings = []
        self.side_chain_rings = []
        side_chain_atom_inds = set()
        for ind,ring in enumerate(ri.AtomRings()):
            common = inds.intersection(ring)
            if len(common) > 0:
                self.main_chain_rings.append(ind)
            else:
                self.side_chain_rings.append(ind)   
        for i in self.main_chain_rings:
            inds = inds.union(ri.AtomRings()[i])
        for i in self.side_chain_rings:
            side_chain_atom_inds = side_chain_atom_inds.union(ri.AtomRings()[i])
        side_chain_atom_inds = side_chain_atom_inds.union([x for x in range(self.mol.GetNumAtoms()) if x not in inds])
        self.main_chain_atoms, self.side_chain_atoms = np.array([self.mol.GetAtomWithIdx(i) for i in inds]),np.array([self.mol.GetAtomWithIdx(i) for i in side_chain_atom_inds])
        #inds=[x for x in Chem.GetShortestPath(self.mol,self.star_inds[0],self.star_inds[1]) if x not in self.star_inds]
        #inds=[x for x in Chem.GetShortestPath(self.mol,self.star_inds[0],self.star_inds[1])]
        
        #return [x for ind,x in enumerate(list(self.mol.GetAtoms())) if ind in inds], [x for ind,x in enumerate(list(self.mol.GetAtoms())) if ind not in inds]
    
    def get_connector_inds(self):
            connector_inds = []
            for ind in self.star_inds:
                star=self.mol.GetAtoms()[ind]
                connector=star.GetNeighbors()[0]
                connector_inds.append(connector.GetIdx())
            return connector_inds
    
    def PeriodicMol(self,repeat_unit_on_fail=False):
        em = Chem.EditableMol(self.mol)
        try:
            em.AddBond(self.connector_inds[0],self.connector_inds[1],Chem.BondType.SINGLE)
            em.RemoveAtom(self.star_inds[1])
            em.RemoveAtom(self.star_inds[0])
            return PeriodicMol( em.GetMol(),self.star_inds,self.connector_inds )
        except:
            print('!!!Periodization of %s Failed!!!' %self.SMILES)
            if repeat_unit_on_fail == False:
                return None
            else:
                em.RemoveAtom(self.star_inds[1])
                em.RemoveAtom(self.star_inds[0])                
                return PeriodicMol( em.GetMol(),self.star_inds,self.connector_inds )

    def SubChainMol(self,mol,keep_atoms):
        em = Chem.EditableMol(mol)
        keep_atoms_idx = [atom.GetIdx() for atom in keep_atoms]
        for i in reversed(range(len(mol.GetAtoms()))):
            if i not in keep_atoms_idx:
                em.RemoveAtom(i)
        m = em.GetMol()
        try:
            Chem.SanitizeMol(m)
            return m
        except:
            try: 
                #reset numHs 
                mol_atoms_in_m = np.sort(keep_atoms_idx)
                discard_atoms_idx = set(range(self.mol.GetNumAtoms())).difference(keep_atoms_idx)
                for i in discard_atoms_idx:
                    neighs_idx = [x.GetIdx() for x in self.mol.GetAtomWithIdx(i).GetNeighbors()]
                    fragment_ind = set(neighs_idx).intersection(keep_atoms_idx)
                    for j in fragment_ind:
                        num_h = int(self.mol.GetBondBetweenAtoms(i,j).GetBondTypeAsDouble())
                        m_ind = int(np.argwhere(mol_atoms_in_m==j))
                        m_atom = m.GetAtomWithIdx(m_ind)
                        m_atom.SetNumExplicitHs(num_h)
                
                #force all rings to be aromatic
                aromatic_atom_inds = [x.GetIdx() for x in m.GetAromaticAtoms()]
                ri = m.GetRingInfo()
                ar = ri.AtomRings()
                br = ri.BondRings()
                for i in range(len(br)):
                    if ar[i][0] in aromatic_atom_inds:
                        for b in br[i]:
                            bond = m.GetBondWithIdx(b)
                            bond.SetBondType(Chem.BondType.AROMATIC)
                Chem.SanitizeMol(m)
                return m    
            except:
                return None

    def SubChainMol2(self,mol,keep_atoms_idx):
        em = Chem.EditableMol(mol)
        for i in reversed(range(len(mol.GetAtoms()))):
            if i not in keep_atoms_idx:
                em.RemoveAtom(i)
        try:
            m = em.GetMol()
            Chem.SanitizeMol(m)
            return m
        except:
            return None
    
    def MainChainMol(self):
        if self.main_chain_atoms is None:
            self.get_main_chain()
        mol = self.SubChainMol(self.mol,self.main_chain_atoms)
        return LinearPol(mol,self.SMILES)
        #return mol
    
    def get_alpha_atoms(self):
        if self.main_chain_atoms is None:
            self.main_chain_atoms, self.side_chain_atoms = self.get_main_chain()
        self.alpha_atoms = set(flatten_ll([list(x.GetNeighbors()) for x in self.main_chain_atoms]))
    
    def AlphaMol(self):
        if self.alpha_atoms is None:
            self.get_alpha_atoms()
        mol = self.SubChainMol(self.mol,self.alpha_atoms)
        return LinearPol(mol,self.SMILES)
    
    def get_beta_atoms(self):
        if self.alpha_atoms is None:
            self.get_alpha_atoms()
        self.beta_atoms = set(flatten_ll([list(x.GetNeighbors()) for x in self.alpha_atoms]))
    
    def BetaMol(self):
        if self.beta_atoms is None:
            self.get_beta_atoms()
        mol = self.SubChainMol(self.mol,self.beta_atoms)
        return LinearPol(mol,self.SMILES)

    def SideChainMol(self):
        if self.main_chain_atoms is None:
            self.get_main_chain()
        mol = self.SubChainMol(self.mol,self.side_chain_atoms)
        return mol
    
    def _HasSubstructMatch(self,mol):
        pm = self.PeriodicMol(repeat_unit_on_fail=True) #must use periodic mol to account for periodicity
        return pm.HasSubstructMatch(mol)
    
    def _GetSubstructMatches(self,mol):
        pm = self.PeriodicMol(repeat_unit_on_fail=True) #must use periodic mol to account for periodicity
        return pm.GetSubstructMatches(mol)        
    
    def delStarMol(self):
        new_mol_connector_inds = [0,0]
        for i,c in enumerate(self.connector_inds):
            n_lower = 0
            for s in self.star_inds:
                if s < c:
                    n_lower += 1
            new_mol_connector_inds[i] = c-n_lower 
        self.delStarMolInds = new_mol_connector_inds
        em = Chem.EditableMol(self.mol)
        em.RemoveAtom(max(self.star_inds)) #remove connection point
        em.RemoveAtom(min(self.star_inds)) #remove connection point
        try:
            new_mol = em.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except:
            return None

    def multiply(self,n):
        '''
        Return a LinearPol which is n times repeated from itself
        '''   
        add_lp = LinearPol(self.mol)
        for i in range(n-1):
            add_lp = LinearPol( bind_frags(self.mol,self.star_inds[1],add_lp.mol,add_lp.star_inds[0],self.connector_inds[1],add_lp.connector_inds[0]) )
        return add_lp
            
    
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

def is_symmetric_chem(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)    
    z=list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    return len(z) != len(set(z))

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

def side_chain_large_abs_rishi(s):
    '''
    Absolute length of the longest side chain without ring. Rishi implementation.
    '''
    try:
        mol = Chem.MolFromSmiles(s)
        lp=ru.LinearPol(mol)

        sc=lp.SideChainMol()

        frags = Chem.GetMolFrags(sc, asMols=True)

        sc_lens = np.zeros((len(frags,)))

        for i,m in enumerate(frags):
            ri=m.GetRingInfo()
            m_len = m.GetNumAtoms() - len(set(ru.flatten_ll(ri.AtomRings())))
            sc_lens[i] = m_len

        return max(sc_lens)    
    except:
        print('!!!Failed on %s. The polymer may not have a side chain!!!' %s)
        return 0

def n_acrylate(s):
    '''
    Return the number of acrylate groups present in the SMILES string s
    '''
    lp = LinearPol(s)
    matches = lp._GetSubstructMatches(Chem.MolFromSmarts('CC(=O)O*'))
    return len(matches)

def n_amide(s):
    '''
    Return the number of amide groups present in the SMILES string s
    '''
    lp = LinearPol(s)
    matches = lp._GetSubstructMatches(Chem.MolFromSmarts('C(=O)N'))
    return len(matches)

def n_spiro(mol):
    '''
    Return number of spiro centers in a MOLECULE(!!!). Works for either molecule objects or molecule SMILES.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    ri = mol.GetRingInfo()
    ar_sets = [set(ring) for ring in ri.AtomRings()]
    num_spiro = 0
    for i,ring in enumerate(ar_sets):
        for j,other_ring in enumerate(ar_sets):
            if i>j:
                if len(ring.intersection(other_ring)) == 1:
                    num_spiro += 1
    return num_spiro

def gprFeatureOrder(model_path,fp_file_path):
    '''
    Return feature order of GPR model
    '''
    file_model = model_path
    model_obj = joblib.load(file_model)
    fingerprint_df = pd.read_csv(fp_file_path)
    fingerprint = fingerprint_df.iloc[0,:].to_dict()

    # From ml_prediction

    debug=1
    gp = joblib.load(file_model)

    X_heads = gp.X_heads

    X = []
    names = []
    for key in X_heads:
        if key in fingerprint:
            X.append(fingerprint[key])
            names.append(key)
        else:
            if 'afp_' in key or 'bfp_' in key or 'mfp_' in key or 'efp_' in key:
                X.append(0)
                names.append(key)
                if debug:
                    print('    !Warning: Column \'%s\'... polymer fingerprint is missing in fingerprint file. Assumed %s = 0' % (key, key))
            else:
                if debug:
                    print('    !Error: Column \'%s\' is missing in fingerprint file. Prediction cannot be made.' % key)
                    print('            Required columns: ', X_heads)
                sys.exit()

    for key in fingerprint:
        if not key in X_heads:
            if debug:
                print('    !Warning: Column \'%s\' is not a member of training dataset. This column is ignored for the prediction.' % key)

    X = np.array(X)
    X = X.reshape(1,-1)

    ML_fp_scale = gp.X_scaled

    if ML_fp_scale == 1:
        # Get X scale from model.pkl
        X_scale = gp.X_scale
        X = X_scale.transform(X)

    return names
    
def equal_Canon(mol1,mol2):
    '''
    Check if two molecules are the same based on their canonical smiles
    '''
    return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)

def arg_unique_ordered(seq):
    '''
    Remove duplicates from list, seq, in order. Return argument number.
    '''
    seen = set()
    seen_add = seen.add
    return [i for i,x in enumerate(seq) if not (x in seen or seen_add(x))]    

def n_spiro_vol(s):
    '''
    Return the number of spiro rings per volume of s. If error return 0.
    '''
    s = s.replace('*','H')
    mol = Chem.MolFromSmiles(s)
    num_spiro = n_spiro(mol)
    try:
        Chem.AllChem.EmbedMolecule(mol)
        return num_spiro/Chem.AllChem.ComputeMolVolume(mol)
    except:
        return 0   

def MolsToGridImage(mol_ls,labels=None,molsPerRow=2,ImgSize=(3, 3),title=''):
    mol_ims = [Chem.Draw.MolToImage(mol,size=(300,300)) for mol in mol_ls]
    n_mols = len(mol_ls)
    n_cols = molsPerRow
    n_rows = int(np.ceil( n_mols / 2 ))

    if molsPerRow == 1: 
        ImgSize=(1.5, 1.5)
    FONT_FACTOR = ImgSize[0] / 6
    
    if labels is None:
        labels = ['' for x in mol_ls]
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.set_size_inches(ImgSize[0], ImgSize[1], forward=True)
    #fig.set_size_inches(20, 20, forward=True)
    for i in range(n_rows):
        for j in range(n_cols):
            try:
                ax = axes[i][j]
            except: 
                try: #in case there is only one row
                    ax = axes[j]
                except: # in case these is only one row and one column
                    ax = axes
            n = n_cols*i + j
            try:
                ax.imshow( mol_ims[n] )
                ax.set_xlabel( labels[n], fontsize=14*FONT_FACTOR )
            except: 
                pass
            #clean each axis
            for s in ax.spines.keys():
                ax.spines[s].set_visible(False)    
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    
    
    
    fig.suptitle(title, fontsize=16*FONT_FACTOR)
    
    downshift = title.count('\n')*.15
    #plt.subplots_adjust( top=(1-downshift) )
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0, 1, 1 - downshift])
    fig.show()
    #return fig

import io
#import cv2
# def get_img_from_fig(fig, dpi=180):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=dpi)
#     buf.seek(0)
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()
#     img = cv2.imdecode(img_arr, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     return img

def bind_frags(m1,m1_tail,m2,m2_head,m1_connector=None,m2_connector=None):
    '''
    Bind MOLECULES OBJECTS, m1 and m2, together at connection points m1_tail and m2_head
    '''
    if m1_connector is None:
        out = m1.GetAtoms()[m1_tail].GetNeighbors()
        if len(out) == 1:
            m1_connector = out[0].GetIdx()
        else:
            raise(ValueError, 'Too many or too few atoms bound to p1_tail')
    if m2_connector is None:
        out = m2.GetAtoms()[m2_head].GetNeighbors()
        if len(out) == 1:
            m2_connector = out[0].GetIdx()
        else:
            raise(ValueError, 'Too many or too few atoms bound to p2_head')    
    
    combo_mol = Chem.rdmolops.CombineMols(m1,m2)
    em = Chem.EditableMol(combo_mol)
    em.AddBond(m1_connector, m2_connector + m1.GetNumAtoms(),Chem.BondType.SINGLE)
    em.RemoveAtom(m1_tail)
    em.RemoveAtom(m2_head + m1.GetNumAtoms() - 1)
    new_mol = em.GetMol()
    try:
        #Chem.SanitizeMol(new_mol)
        return new_mol
    except:
        return None

def pd_load_pg_json(path_to_living_json):
    lines = ['{']
    with open(path_to_living_json, 'r') as f:
        for ind,line in enumerate(f):
            lines.append(line.strip())
    lines.append('}')
    spl_s = path_to_living_json.split('.csv')
    new_file_name = spl_s[0] + '_intermed' + spl_s[1]
    write_list_to_file(lines,new_file_name)
    return pd.read_json(new_file_name).transpose().fillna(0)

def pg_can(sm):
    '''
    Return the polymer genome canonicalized version of the SMILES string, sm
    '''
    return sm.replace('*','[*]')

def is_soluble(pol):
    '''
    Return whether or not the polymer, pol, is soluble
    False: if all atoms in repeat unit belong to at least one ring
    True: Else
    '''
    if type(pol) == str or type(pol) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        pol = LinearPol(pol)
    n_connectors = len(pol.connector_inds)
    try:
        non_ring = set(flatten_ll(pol.mol.GetSubstructMatches(Chem.MolFromSmarts('[R0]'))))
    except:
        Chem.GetSSSR(pol.mol)
        non_ring = set(flatten_ll(pol.mol.GetSubstructMatches(Chem.MolFromSmarts('[R0]'))))
    if len(non_ring) == n_connectors:
        return False
    else:
        return True