import rishi_utils as ru
from rdkit import Chem
import itertools
from rdkit.Chem import rdmolfiles

def frp_possible(mol):
    '''
    Determine if this polymer, mol (either SMILES string or rdkit mol object), is a candidate for free-radical polymerization.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    main_chain_patt = Chem.MolFromSmiles('*CC*')
    side_chain_patt = Chem.MolFromSmiles('C=C')
    
    return mol.HasSubstructMatch(main_chain_patt) & ( not mol.HasSubstructMatch(side_chain_patt) )

def hydrogenate_chain(mol):
    '''
    Return a list of pol-mols which each contain a hydrogenated version of mol
    '''
    
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    patt = Chem.MolFromSmiles('CC')
    
    matches = mol.GetSubstructMatches(patt)
    
    new = []
    for bonds in matches:
        em = Chem.EditableMol(mol)
        em.RemoveBond(bonds[0],bonds[1])
        em.AddBond(bonds[0],bonds[1],Chem.BondType.DOUBLE)
        try:
            Chem.SanitizeMol(em.GetMol())
            new.append(em.GetMol())
        except:
            pass
    return new

def admet_depolymerize(lp):
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    patt = Chem.MolFromSmiles('C=C')    
    if not frp_possible(lp.mol):
        pm = lp.PeriodicMol()
        mc = lp.MainChainMol()
        pm_match = pm.GetSubstructMatches(patt)
        if mc.HasSubstructMatch(patt) and len(pm_match)==1:
            em = Chem.EditableMol(pm)
            em.RemoveBond(pm_match[0][0],pm_match[0][1])
            C1_ind = em.AddAtom(Chem.AtomFromSmiles('C'))
            C2_ind = em.AddAtom(Chem.AtomFromSmiles('C'))
            em.AddBond(pm_match[0][0],C1_ind,Chem.BondType.DOUBLE)
            em.AddBond(pm_match[0][1],C2_ind,Chem.BondType.DOUBLE)
            try:
                Chem.SanitizeMol(em.GetMol())
                return em.GetMol()
            except:
                return None
            

def ro_depolymerize(lp): #Need to implement
    '''
    Return (polymer,monomer) if ring-opening polymerization is possible
    '''
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    
    out=[]
    am = lp.AlphaMol()
    patterns = ['CNC(=O)','COC(=O)']
    for patt in patterns:
        try:
            if len(am.GetSubstructMatches(patt))==1:
                out.append((lp,patt))
        except:
            pass
    return out

def condensation_possible(lp):
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    possible_patt_smiles = ['CO']
    has_patt = []
    try:
        pm = lp.PeriodicMol()
    except:
        return (False,has_patt)
    for s in possible_patt_smiles:
        if len(pm.GetSubstructMatches(Chem.MolFromSmiles(s))) > 1:
            has_patt.append(s)
    if len(has_patt) > 0:
        return (True,has_patt)
    else:
        return (False,has_patt)

def depolymerize(lp,patt):
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    pm = lp.PeriodicMol()    
    matches=pm.GetSubstructMatches(Chem.MolFromSmiles(patt)) #need to fix this to only look at main chain mol
    bond_pairs = list(itertools.combinations(matches, 2))
    monomer_sets = []
    def fragment(bonds):
        '''
        Fragment PeriodicMol according to bonds
        '''
        b1 = bonds[0]
        b2 = bonds[1]
        
        carbon_idx_1 = [x for x in b1 if pm.GetAtoms()[x].GetSymbol() == 'C'][0]
        carbon_idx_2 = [x for x in b2 if pm.GetAtoms()[x].GetSymbol() == 'C'][0]
        n_atoms = len(pm.GetAtoms())
        
        em = Chem.EditableMol(pm)

        em.RemoveBond(b1[0],b1[1])
        em.RemoveBond(b2[0],b2[1])
        m=em.GetMol()
        em.AddAtom(Chem.AtomFromSmiles('[Cl]'))
        em.AddAtom(Chem.AtomFromSmiles('[Cl]'))
        em.AddBond(carbon_idx_1,n_atoms,Chem.BondType.SINGLE)
        em.AddBond(carbon_idx_2,n_atoms+1,Chem.BondType.SINGLE)
        m=em.GetMol()
        try:
            Chem.SanitizeMol(m)
            s=Chem.MolToSmiles(m) 
            ls_s = s.split('.') #split dual molecule object into two molecule objects
            if (len(ls_s) == 2) and ru.is_symmetric_chem(Chem.MolFromSmiles(ls_s[0])) and ru.is_symmetric_chem(Chem.MolFromSmiles(ls_s[1])): #the fragments need to be symmetric
                return m
            else:
                return None
        except:
            return None
        
        
    for pair in bond_pairs:
        if (pair[0][0] in lp.main_chain_atoms) and (pair[0][1] in lp.main_chain_atoms) and (pair[1][0] in lp.main_chain_atoms) and (pair[1][1] in lp.main_chain_atoms):
            out = fragment(pair)
            if out != None:
                monomer_sets.append(out)
    return monomer_sets