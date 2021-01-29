
import rishi_utils as ru
from rdkit import Chem
import itertools
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('/data/rgur/retrosynthesis/scscore')
import re
from joblib import Parallel, delayed
import fall20_mse_8803 as retro

def ppf_poly_carboxylic_acid_plus_carbodiimide(mol):
    if type(mol) == str:
        mol = ru.LinearPol(Chem.MolFromSmiles(mol))
    try:
        mod_mol = Chem.ReplaceSubstructs(mol, 
                                         Chem.MolFromSmarts('C(=O)[NH]'), 
                                         Chem.MolFromSmiles('C(=O)[OH]'),
                                         replaceAll=True)

        mod_mol_str = Chem.MolToSmiles(mod_mol[0])

        ls=mod_mol_str.split('.')

        non_hc = []
        for s in ls:
            if not isHydrocarbon(s):
                non_hc.append(s)

        if len(non_hc) == 1:
            return Chem.MolFromSmiles(non_hc[0]) 
        else:
            return None
    except:
        print(mol.SMILES)
        return None

def hydrogenate_chain(lp,max_replacements=1):
    '''
    Return a list of pol-mols which each contain a hydrogenated version of mol. We set max_replacements to 1 since only one double-bond is typically needed for ADMET and ROMP. BEWARE: setting max_replacements > 1 will result in considerable computational expense.
    '''
    
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
        
    ri = lp.mol.GetRingInfo()
    br = ru.flatten_ll(ri.BondRings())
    
    single_bonds = set([b.GetIdx() for b in lp.mol.GetBonds() if (b.GetBondType() == Chem.BondType.SINGLE) and (b.GetIdx() not in br)]).difference([b.GetIdx() for b in lp.mol.GetAtomWithIdx(lp.star_inds[0]).GetBonds()]+[b.GetIdx() for b in lp.mol.GetAtomWithIdx(lp.star_inds[1]).GetBonds()])

    n_single_bonds = min(len(single_bonds),max_replacements) #cap the number of single_bonds that can be replaced for performance reasons
    mols = []
    for L in range(1,n_single_bonds+1):
        for combo in itertools.combinations(single_bonds,L):
            mol_copy = Chem.MolFromSmiles(lp.SMILES)
            try:
                clean_edit = False
                for bond in combo:
                    if (mol_copy.GetAtomWithIdx(mol_copy.GetBondWithIdx(bond).GetBeginAtomIdx()).GetNumImplicitHs() > 0) and (mol_copy.GetAtomWithIdx(mol_copy.GetBondWithIdx(bond).GetEndAtomIdx()).GetNumImplicitHs() > 0):
                        mol_copy.GetBondWithIdx(bond).SetBondType(Chem.BondType.DOUBLE)
                        clean_edit = True
                    else:
                        clean_edit = False
                        break
                if clean_edit:
                    Chem.SanitizeMol(mol_copy)
                    mols.append(mol_copy)
                mol_copy = None
            except:
                pass
    return mols

func_chain_rxns = {
    'nitro_base':Chem.MolFromSmarts('n[*R0]'), #test SMILES = [*]C1=CC(=CC=C1)C6=NC2=C(C=C(C=C2)C3=CC4=C(C=C3)N=C([*])[N]4CC5=CC=CC=C5)[N]6CC7=CC=CC=C7
    'SO2_oxidation':Chem.MolFromSmarts('S(=O)(=O)'), #test SMILEs = [*]Oc1ccc(S(=O)(=O)c2ccc(Oc3ccc(C(C)(C)c4ccc([*])cc4)cc3)cc2)cc1
    'aldean': #aldean = ALDehydes + Alpha-Effect Nucleophiles
                #Catalyst = aniline
                #Source: Dr. Finn Lecture Slides, p. 18
        ['[*;R0,R1,R2:0][NHR0:1][NH0R0:2]=[CR0:3]([*;R0,R1,R2:4])[*;R0,R1,R2:5]>>[*:4][C:3](=O)[*:5].[NH2:2][NH:1][*:0]', #when R != H
        '[*;R0,R1,R2:0][NHR0:1][NH0R0:2]=[CR0:3][*;R0,R1,R2:5]>>[C:3](=O)[*:5].[NH2:2][NH:1][*:0]'], #when R = H
        #first item in list is checked first, last item is checked last
    'thiol-ene': #catalyst = radical initiator
                #Source: Dr. Finn Lecture Slides, p. 27
        ['[*;R0:1][Sv2:2][CH2:3][CH2:4][*;R0,R1,R2:5]>>[CH2:3]=[CH:4][*:5].[*:1][SH:2]'],
    'azal': #azal = Azide + Alkyne
            #Catalyst = Cu(1)
            #Source:  Dr. Finn Lecture Slides, p. 29-36
        ['[*:6][#7:1]1:[#6:4]:[#6:5]:[#7:2]:[#7:3]:1>>[C:4]#[C:5].[*:6][N:1]=[NX2+:2]=[NX1-:3]']
}

def func_chain_retro(lp,rxn):
    '''
    Functions to retrosynthetically functionalize chains. The majority of reactions only transform at most one match.
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)


    if rxn in ['aldean','thiol-ene','azal']:
        ls = func_chain_rxns[rxn]
        for smart in ls:
            RD_rxn = Chem.AllChem.ReactionFromSmarts(smart) #RDKit reaction object
            rxn_out = RD_rxn.RunReactants((lp.mol,))
            if len(rxn_out) != 0:
                for mol in rxn_out[0]: #only look at first set of reactants
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[#0]')):
                        return [mol]
        return []

    replace_group = func_chain_rxns[rxn]
    if rxn == 'nitro_base':
        sc = lp.SideChainMol()
        if sc is None:
            return []
        sc_matches = sc.GetSubstructMatches(replace_group)
        if len(sc_matches) != 0:
            return []
    matches=lp.mol.GetSubstructMatches(replace_group)
    if len(matches) == 0:
        return []
    flat_matches = ru.flatten_ll(matches)
    if lp.star_inds[0] in flat_matches or lp.star_inds[1] in flat_matches: #make sure the connector inds aren't being matched
        return []
    else:
        mols = []
        if rxn == 'nitro_base': #this branch is for reactions where arbitrary groups must be removed
            frag_mol=Chem.FragmentOnBonds(lp.mol,[lp.mol.GetBondBetweenAtoms(i,j).GetIdx() for i,j in matches])
            frag_mols = Chem.GetMolFrags(frag_mol)[1:]
            frag_ls = Chem.MolToSmiles(ru.mol_without_atom_index(frag_mol)).split('.')[1:]
            frag_ls = [re.sub('\[[0-9]+\*\]','*',s) for s in frag_ls]
            unique_frags = {}#SMILES - list of atom matches
            for i,s in enumerate(frag_ls):
                if s not in unique_frags:
                    unique_frags[s] = list(frag_mols[i])
                else:
                    unique_frags[s].extend(frag_mols[i])
            #for L in range( 1,len(unique_frags.keys())+1 ):
            for combo_matches in unique_frags.values():
                all_atoms = set(range(lp.mol.GetNumAtoms()))
                keep_atoms = all_atoms.difference(combo_matches)
                mols.append( lp.SubChainMol(lp.mol,[lp.mol.GetAtomWithIdx(x) for x in keep_atoms]) )
        else:
            n_matches = len(matches)
            for L in range(1,n_matches+1):
                for match_combo in itertools.combinations(matches,L):
                    o_inds = []
                    for match in match_combo:
                        o_inds.append(match[1])
                        o_inds.append(match[2])
                    o_inds = sorted(o_inds,reverse=True)
                    em = Chem.EditableMol(lp.mol)
                    [em.RemoveAtom(x) for x in o_inds]
                    new_mol = em.GetMol()
                    try:
                        Chem.SanitizeMol(new_mol)
                        mols.append(new_mol)
                    except:
                        pass
    return mols
    
def edit_RCO2H(em,pm,match_combo,L=None):
    '''
    Should return a list of mols
    Source: Reynolds Class Notes, 9-24-20, p.3
    Test SMILES: *c1ccc(*)cc1
    '''
    o_inds = [em.AddAtom(Chem.AtomFromSmiles('O')) for _ in range(L)]
    c_inds = [em.AddAtom(Chem.AtomFromSmiles('C')) for _ in range(L)]
    dbl_o_inds = [em.AddAtom(Chem.AtomFromSmiles('O')) for _ in range(L)]
    methyl_c_inds = [em.AddAtom(Chem.AtomFromSmiles('C')) for _ in range(L)]
    for i in range(L):
        em.AddBond(o_inds[i],match_combo[i][1],Chem.BondType.SINGLE)
        em.AddBond(c_inds[i],o_inds[i],Chem.BondType.SINGLE)
        em.AddBond(c_inds[i],dbl_o_inds[i],Chem.BondType.DOUBLE)
        em.AddBond(c_inds[i],methyl_c_inds[i],Chem.BondType.SINGLE)    
    em = retro.pm_to_lp_em(em,pm)
    new_mol = em.GetMol()
    for x in match_combo:
        new_mol.GetAtomWithIdx(x[1]).SetNumExplicitHs( pm.mol.GetAtomWithIdx(x[1]).GetNumImplicitHs() )
        new_mol.GetAtomWithIdx(x[0]).SetNumExplicitHs( pm.mol.GetAtomWithIdx(x[0]).GetNumImplicitHs() + 1 )
    return [new_mol]

def edit_HCl(em,pm,match_combo,L=None):
    '''
    Should return a list of mols
    Source: Reynolds Class Notes, 9-24-20, p.5
    Test SMILES: '*/C(C#N)=C/C1C=CC(*)CC1'
    '''
    match_arr = np.array(match_combo)
    new_mols = []
    for i in range(2):
        em = Chem.EditableMol(pm.mol)
        cl_inds = [em.AddAtom(Chem.AtomFromSmiles('Cl')) for _ in range(L)]
        c_inds = match_arr[:,i]
        for i in range(L):
            em.RemoveBond(match_combo[i][0],match_combo[i][1])
            em.AddBond(match_combo[i][0],match_combo[i][1],Chem.BondType.SINGLE)
            em.AddBond(int(c_inds[i]),cl_inds[i],Chem.BondType.SINGLE)
        em = retro.pm_to_lp_em(em,pm)
        new_mols.append( em.GetMol() )
    return new_mols

elim_rxns = {
    'RCO2H': {'replace_patt': '[c,C;R][c,C;R;!H0]', 'edit_fxn': edit_RCO2H},
    'HCl': {'replace_patt': 'C=C', 'edit_fxn': edit_HCl},
    }

def elim_retro(lp, elim_group, pm=None, max_sites=2, max_matches=30):
    '''
    Reverse of eliminations. They generally will occur after addition of heat.
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    if pm is None:
        pm = lp.PeriodicMol()
    if pm is None:
        return []

    rxn_info = elim_rxns[elim_group]
    replace_patt = Chem.MolFromSmarts(rxn_info['replace_patt'])
    matches = pm.GetSubstructMatches(replace_patt)
    mols = []
    if len(matches) == 0:
        return []
    max_sites = min(len(matches),max_sites)
    for L in range(1, max_sites+1):
        for match_combo in itertools.combinations(matches,L):
            match_combo_flat = ru.flatten_ll(match_combo)
            if len(mols) == max_matches: #limit the number of combinations that are tried
                pass
            elif len(set(match_combo_flat)) != len(match_combo_flat): #no atoms should overlap
                pass
            else:
                em = Chem.EditableMol(pm.mol)
                #print('elim_group:',elim_group)
                try:
                    new_mols = rxn_info['edit_fxn'](em,pm,match_combo,L)
                except:
                    new_mols = []
                for new_mol in new_mols:
                    try:
                        Chem.SanitizeMol(new_mol)
                        mols.append(new_mol)
                    except:
                        pass
    return mols

def ring_close_retro(lp,pm=None):
    '''
    Reverse of Ring-closing which will occur after addition of heat
    Source: Reynolds Class Notes, 8-25-20, p.2
    Test SMILES: '*c5ccc(Oc4ccc(n3c(=O)c2cc1c(=O)n(*)c(=O)c1cc2c3=O)cc4)cc5'
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    
    start_match = Chem.MolFromSmarts('[c,C;!R0;!R1](=O)[n,N;!R0;!R1]')
    end_match = Chem.MolFromSmarts('[#6R0](=O)([OH])[C,c][C,c][CR1](=O)[NR1]')
    if pm is None:
        pm = lp.PeriodicMol()
        if pm is not None:
            pm.GetSSSR()
    
    if pm is None:
        return []
    
    
    mols = []
    if pm.HasSubstructMatch(start_match) and not pm.HasSubstructMatch(end_match): 
        lp_no_connect_inds = np.array([x for x in range(lp.mol.GetNumAtoms()) if x not in lp.star_inds])
        def lp_to_pm_ind(lp_ind):
            return int(np.argwhere(lp_no_connect_inds==lp_ind))
        ar_atom_idx = [a.GetIdx() for a in lp.mol.GetAromaticAtoms()]
        if len(ar_atom_idx) != 0: #only execute below for aromatic polymers. There for speed.
            ri = lp.mol.GetRingInfo()
            ar = ri.AtomRings()
            atom_aromaticity = {a:0 for a in ar_atom_idx}

            for ring in ar:
                ar_ring = 1
                for a in ring:
                    if a not in atom_aromaticity.keys():
                        ar_ring = 0
                if ar_ring == 1:
                    for a in ring:   
                        atom_aromaticity[a] += 1

        all_matches = pm.mol.GetSubstructMatches(start_match)
        seen = set()
        matches = [] #unique matches
        for match in all_matches:
            ms = set(match) #match set
            #print(seen.difference(ms))
            #print(ms.difference(seen))
            if len( ms.intersection(seen) ) == 0:
                seen = seen.union(ms)
                matches.append(match)
        #matches = pm.mol.GetSubstructMatches(start_match)
        for L in range(1, len(matches)+1):
        #for L in range(2,3):
            for match_combo in itertools.combinations(matches,L):
                em = Chem.EditableMol(pm.mol)
                #print('Match combo:', match_combo)
                for i_c,i_o,i_n in match_combo: #indices of atoms in pm
                    
                    #print('Matches: %s %s %s' %(i_c,i_o,i_n) )
                    fix_aromaticity = False
                    if pm.mol.GetBondBetweenAtoms(i_c,i_n).GetBondType() == Chem.BondType.AROMATIC:
                        fix_aromaticity = True
                        ring_atoms = None
                        ring_size = 100
                        for i in range(len(ar)):
                            ring = ar[i]
                            if lp_no_connect_inds[i_c] in ring and lp_no_connect_inds[i_n] in ring and len(ring) < ring_size: #assume correct ring is the smallest one
                                ring_atoms = set(ring)
                                ring_size = len(ring)

                    o=em.AddAtom(Chem.AtomFromSmiles('O'))
                    em.AddBond(i_c,o,Chem.BondType.SINGLE)
                    #print('bond between %s and %s' %(i_c,o))
                    em.RemoveBond(i_c,i_n)
                    #print('Bond removed between %s and %s' %(i_c,i_n))

                    med_mol = em.GetMol()
                    if fix_aromaticity:
                        try:
                            i_n_aromaticity = atom_aromaticity[ lp_to_pm_ind(i_n) ]
                        except:
                            i_n_aromaticity = 0
                        for i in ring_atoms:
                            if atom_aromaticity[ i ] == i_n_aromaticity: #if an atom was part of same number of aromatic rings as the N atom, it shouldn't be aromatic
                                #print('Ring atom lp:',i)
                                pm_i = lp_to_pm_ind(i)
                                #print('Ring atom pm:',pm_i)
                                med_mol.GetAtomWithIdx( pm_i ).SetIsAromatic(False)
                                #remove all aromatic bonds
                                neighs = [x.GetIdx() for x in med_mol.GetAtoms()[ pm_i ].GetNeighbors()]
                                aromatic_neighs = [x for x in neighs if med_mol.GetBondBetweenAtoms(pm_i,x).GetBondType()==Chem.BondType.AROMATIC]
                                #print('Aromatic neighs of %s: %s' %(pm_i,aromatic_neighs))
                                em = Chem.EditableMol(med_mol)
                                for x in aromatic_neighs:
                                    em.RemoveBond( x, pm_i )
                                    em.AddBond(x,pm_i, Chem.BondType.SINGLE)
                                med_mol = em.GetMol()     
                
                em = Chem.EditableMol(med_mol)
                star1 = em.AddAtom(Chem.AtomFromSmiles('*'))
                star2 = em.AddAtom(Chem.AtomFromSmiles('*'))
                em.RemoveBond(pm.connector_inds[0],pm.connector_inds[1])
                em.AddBond(pm.connector_inds[0],star1,Chem.BondType.SINGLE)
                em.AddBond(pm.connector_inds[1],star2,Chem.BondType.SINGLE)

                new_mol=em.GetMol()
                try:
                    Chem.SanitizeMol(new_mol)
                    mols.append( ru.mol_without_atom_index(new_mol) )
                except:
                    return []
        return mols
    else:
        return []

post_polymerization_rxns = [ring_close_retro, func_chain_retro, hydrogenate_chain, elim_retro] #each return type should be ***list of mols*** or ***empty*** list
