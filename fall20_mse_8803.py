import rishi_utils as ru
from rdkit import Chem
import itertools
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem
import numpy as np

def frp_possible(mol):
    '''
    Determine if this polymer, mol (either SMILES string or rdkit mol object), is a candidate for free-radical polymerization.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    main_chain_patt = Chem.MolFromSmarts('[#0]C[CH2][#0]') #ensure at least one carbon is unsubstituted
    side_chain_patt = Chem.MolFromSmiles('C=C')
    
    return mol.HasSubstructMatch(main_chain_patt) & ( not mol.HasSubstructMatch(side_chain_patt) )

def frp_depolymerize(mol):
    '''
    Determine if this polymer, mol (either SMILES string or rdkit mol object), is a candidate for free-radical polymerization.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    #main_chain_patt1 = Chem.MolFromSmarts('[#0]C[CH2][#0]') #ensure at least one carbon is unsubstituted
    #main_chain_patt2 = Chem.MolFromSmarts('[#0]C=[CH][#0]') #ensure at least one carbon is unsubstituted
    main_chain_patt1 = Chem.MolFromSmarts('*CC*')
    main_chain_patt2 = Chem.MolFromSmarts('*C=C*')

    side_chain_patt1 = Chem.MolFromSmiles('C=C')
    side_chain_patt2 = Chem.MolFromSmiles('C#C')
    side_chain_patt3 = Chem.MolFromSmarts('[OH]')
    mc_match1 = mol.HasSubstructMatch(main_chain_patt1)
    mc_match2 = mol.HasSubstructMatch(main_chain_patt2)
    sc_matches = mol.HasSubstructMatch(side_chain_patt1) or mol.HasSubstructMatch(side_chain_patt2) or side_chain_patt3
    
    if not sc_matches:
        lp = ru.LinearPol(mol)
        n_connectors = len(set(lp.connector_inds))
        if mc_match1 and n_connectors == 2:
            em = Chem.EditableMol(mol)
            em.RemoveBond(lp.connector_inds[0],lp.connector_inds[1])
            em.AddBond(lp.connector_inds[0],lp.connector_inds[1],Chem.BondType.DOUBLE) #replace single bond w/ double
            em.RemoveAtom(lp.star_inds[1])
            em.RemoveAtom(lp.star_inds[0]) #remove connection points
            
            try:
                new_mol = em.GetMol()
                Chem.SanitizeMol(new_mol)
                em = Chem.EditableMol(new_mol)
                #lp.delStarMol()
                inds = new_mol.GetSubstructMatch(Chem.MolFromSmiles('C=C'))
                em.RemoveBond(inds[0],inds[1])
                m=em.GetMol()
                frags = Chem.GetMolFrags(m, asMols=True)
                Chem.AllChem.EmbedMolecule(frags[0])
                Chem.AllChem.EmbedMolecule(frags[1])
                volumes = np.array([Chem.AllChem.ComputeMolVolume(frags[0]),Chem.AllChem.ComputeMolVolume(frags[1])])
                if np.min(volumes) < 35.7: #35.7 is the volume of C(F)(F). 
                    return new_mol
                else:
                    #print(np.min(volumes))
                    return None
            except:
                return None
        elif mc_match2 and n_connectors == 2:
            em = Chem.EditableMol(mol)
            em.RemoveBond(lp.connector_inds[0],lp.connector_inds[1])
            em.AddBond(lp.connector_inds[0],lp.connector_inds[1],Chem.BondType.TRIPLE) #replace double bond w/ triple
        
            em.RemoveAtom(lp.star_inds[1])
            em.RemoveAtom(lp.star_inds[0]) #remove connection points
            try:
                new_mol = em.GetMol()
                Chem.SanitizeMol(new_mol)
                em = Chem.EditableMol(new_mol)
                #lp.delStarMol()
                inds = new_mol.GetSubstructMatch(Chem.MolFromSmiles('C#C'))
                em.RemoveBond(inds[0],inds[1])
                m=em.GetMol()
                frags = Chem.GetMolFrags(m, asMols=True)
                Chem.AllChem.EmbedMolecule(frags[0])
                Chem.AllChem.EmbedMolecule(frags[1])
                volumes = np.array([Chem.AllChem.ComputeMolVolume(frags[0]),Chem.AllChem.ComputeMolVolume(frags[1])])
                if np.min(volumes) < 20.43: #35.7 is the volume of C(F)(F). 20.43 is the volume of C(H)(H)
                    return new_mol
                else:
                    #print(np.min(volumes))
                    return None
            except:
                return None
        else:
            return None
    else:
        return None

def ox_depolymerize(mol):
    '''
    Retrosynthesis of an oxidative depolymerization reaction
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    lp = ru.LinearPol(mol)
    new_mol = lp.delStarMol() #make would-be monomer
    if is_symmetric2(new_mol,group=lp.delStarMolInds): #check symmetry 
        bonds = lp.mol.GetAtoms()[max(lp.connector_inds)].GetBonds() 
        bond_types = np.array([b.GetBondTypeAsDouble() for b in bonds]) #bond types of connector atom
        if any(bond_types > 1): #check if there is a pi-system
            return new_mol
        else:
            return None
    else:
        return None    

    def isHydrocarbon(smiles):
        if type(smiles) != str:
            smiles = Chem.MolToSmiles(smiles)
        smiles=smiles.lower()
        filtered = [c.lower() for c in smiles if c.isalpha()]
        if smiles.count('c') == len(filtered):
            return True
        else:
            return False

def getCharges(mol,atom_inds):
    atoms = mol.GetAtoms()
    return [atoms[ind].GetDoubleProp('_GasteigerCharge') for ind in atom_inds]

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

def dap_depolymerize(mol):
    '''
    Determine if this polymer, mol (either SMILES string or rdkit mol object), is a candidate for donor-acceptor polymerization.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    #main_chain_patt = Chem.MolFromSmarts('[#0][C;H0,H1][C;H0,H1][C;H0,H1][C;H0,H1][#0]') #does not allow unsubbed carbons
    main_chain_patt = Chem.MolFromSmarts('[#0][C;H0,H1,H2][C;H0,H1,H2][C;H0,H1,H2][C;H0,H1,H2][#0]') #allows unsubbed carbons
    side_chain_patt = Chem.MolFromSmiles('C=C')
    
    if mol.HasSubstructMatch(main_chain_patt) & ( not mol.HasSubstructMatch(side_chain_patt) ):
        match = mol.GetSubstructMatch(main_chain_patt)

        em = Chem.EditableMol(mol)

        em.RemoveBond(match[2],match[3]) #unjoin repeat unit

        em.RemoveBond(match[1],match[2])
        em.AddBond(match[1],match[2],Chem.BondType.DOUBLE) #replace single bond w/ double

        em.RemoveBond(match[3],match[4]) 
        em.AddBond(match[3],match[4],Chem.BondType.DOUBLE) #replace single bond w/double

        em.RemoveAtom(match[0])
        em.RemoveAtom(match[-1]-1) #remove connection points
        try:
            new_mol = em.GetMol()
            Chem.SanitizeMol(new_mol)
            if Chem.MolToSmiles(new_mol).count('.')==1: #make sure two monomers have been found
                new_mol.ComputeGasteigerCharges()
                mon_matches=new_mol.GetSubstructMatches(Chem.MolFromSmiles('C=C'))
                charges = getCharges(new_mol,[mon_matches[0][0],mon_matches[0][1],mon_matches[1][0], \
                     mon_matches[1][1]])
                print(charges)
                if charges[0]*charges[1] > 0 and charges[2]*charges[3] > 0 and charges[3]*charges[1] < 0:
                    return new_mol
                else:
                    return None
            else:
                return None
        except:
            return None
        
    else:
        return None

def hclify(mol):
    '''
    Return a list of mols which each contain a version of mol but with ONE HCl added. HCl can be removed with heat. Might want to add version where multiple HCl are added.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    
    new = []
    patt = Chem.MolFromSmarts('[CH2,CH1]=C')
    matches=mol.GetSubstructMatches(patt)
    for match in matches:
        for c_func_ind in [0,1]:
            em = Chem.EditableMol(mol)

            Cl=em.AddAtom(Chem.AtomFromSmiles('Cl'))

            em.RemoveBond(match[0],match[1])
            em.AddBond(match[0],match[1],Chem.BondType.SINGLE)
            em.AddBond(match[c_func_ind],Cl,Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(em.GetMol())
                new.append(em.GetMol())
            except:
                pass
    return new
 
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
            
def ro_depolymerize2(lp):
    '''
    Return (polymer,monomer) if ring-opening polymerization is possible
    '''
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    
    patt1=Chem.MolFromSmiles('CNC(=O)')
    patt2=Chem.MolFromSmiles('COC(=O)')
    patt3=Chem.MolFromSmiles('COC')
    patt4=Chem.MolFromSmiles('CNC')
    patt5=Chem.MolFromSmiles('COC=N')
    patt_oh=Chem.MolFromSmarts('[OX2H]')
    try:
        pm = lp.PeriodicMol()
        am = lp.AlphaMol()
        mcm = lp.MainChainMol()
        alpha_match_len = (len(am.GetSubstructMatches(patt1)),len(am.GetSubstructMatches(patt2)),
                          len(mcm.GetSubstructMatches(patt3)),
                          len(mcm.GetSubstructMatches(patt4)),
                          len(mcm.GetSubstructMatches(patt5)),
                          len(am.GetSubstructMatches(patt5))) #exo-imino
        if alpha_match_len[0] == 1 or alpha_match_len[1] == 1 or alpha_match_len[4] == 1 or alpha_match_len[5] == 1:
            alpha_match_len = list(alpha_match_len)
            alpha_match_len[2] = 0 #if we have an ester third group should be removed
            alpha_match_len[3] = 0
            alpha_match_len = tuple(alpha_match_len)
        if alpha_match_len[4] == 1:
            alpha_match_len[5] == 0 #endo-inimo overshadows exo-imino
        argmax = np.argmax(alpha_match_len)
        if sorted(alpha_match_len) == [0, 0, 0, 0, 0, 1]: #make sure groups exist on alpha_chain
            lactone = len(pm.GetSubstructMatches(patt2))
            cyc_ether = len(pm.GetSubstructMatches(patt3))
            endo_imino_cyc_ether = len(pm.GetSubstructMatches(patt5))
            if pm.HasSubstructMatch(patt_oh) == True: 
                lactone = 0 #OH for lactone is no good
                cyc_ether = 0 #OH for cyclic ether is no good
                endo_imino_cyc_ether = 0 #OH for endo_imino_cyc_ether is no good
            tot_match_len = (len(pm.GetSubstructMatches(patt1)),lactone,cyc_ether,
                            len(pm.GetSubstructMatches(patt4)),endo_imino_cyc_ether,endo_imino_cyc_ether)
            if tot_match_len[argmax] == alpha_match_len[argmax]: #make sure no groups are on side chains
                return (lp.mol,pm)
            else:
                return None
        else:
            return None
    except:
        return None   
            
def ro_depolymerize(lp):
    '''
    Return (polymer,monomer) if ring-opening polymerization is possible
    '''
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    
    patt1=Chem.MolFromSmiles('CNC(=O)')
    patt2=Chem.MolFromSmiles('COC(=O)')
    
    patt3=Chem.MolFromSmarts('[OX2H]')
    try:
        pm = lp.PeriodicMol()
        am = lp.AlphaMol()
        alpha_match_len = (len(am.GetSubstructMatches(patt1)),len(am.GetSubstructMatches(patt2)))
        if sorted(alpha_match_len) == [0, 1]: #make sure groups exist on alpha_chain
            lactone = len(pm.GetSubstructMatches(patt2))
            if pm.HasSubstructMatch(patt3) == True: 
                lactone = 0 #OH for lactone is no good
            tot_match_len = (len(pm.GetSubstructMatches(patt1)),lactone)
            if tot_match_len == alpha_match_len: #make sure no groups are on side chains
                return (lp.mol,pm)
            else:
                return None
        else:
            return None
    except:
        return None

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


def cyclodepolymerize(mol):
    
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    # check that connection point is on ring
    lp = ru.LinearPol(mol)
    ri = mol.GetRingInfo()
    
    non_aromatic_rings = []
    for ind,ring in enumerate(ri.BondRings()):
        for bond_ind in ring:
            if mol.GetBonds()[bond_ind].GetBondType() != Chem.rdchem.BondType.AROMATIC:
                non_aromatic_rings.append(ind)
    non_aromatic_rings = set(non_aromatic_rings)
    
    if len(non_aromatic_rings) != 0:
        non_aromatic_ring_atoms = []
        ring_connection_atom_ind = 'False'
        ring_connection_ring_ind = 'False'
        for ind,ring in enumerate(ri.AtomRings()):
            if ind in non_aromatic_rings:
                for atom_ind in ring:
                    if atom_ind in lp.connector_inds:
                        ring_connection_atom_ind  = atom_ind
                        ring_connection_ring_ind = ind

        if ring_connection_ring_ind != 'False':
            #cycle through all ring bonds of ring_connection_ring_ind
            monomers = []
            patt1 = Chem.MolFromSmiles('*CC*')
            patt2 = Chem.MolFromSmiles('*C=C*')
            for bond_ind in ri.BondRings()[ring_connection_ring_ind]:
                bond = mol.GetBondWithIdx(bond_ind)
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    em = Chem.EditableMol(mol)
                    atoms = (bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
                    em.RemoveBond(atoms[0],atoms[1])
                    ind1=em.AddAtom(Chem.AtomFromSmiles('*'))
                    ind2=em.AddAtom(Chem.AtomFromSmiles('*'))
                    em.AddBond(atoms[0],ind1,Chem.rdchem.BondType.SINGLE)
                    em.AddBond(atoms[1],ind2,Chem.rdchem.BondType.SINGLE)
                    new_mol = em.GetMol()
                    n_matches = len(new_mol.GetSubstructMatches(patt1)) + len(new_mol.GetSubstructMatches(patt2))
                    if n_matches == 2:
                        monomers.append(new_mol)

            monomers2 = []
            for a in monomers:
                success_rxns = 0
                rxn = AllChem.ReactionFromSmarts('[#0]C([*:1])=C[#0]>>[*:1]C#C')
                rxn2 = AllChem.ReactionFromSmarts('[#0]C([*:1])C[#0]>>[*:1]C=C')
                try:
                    ps = rxn.RunReactants((a,))
                    b=ps[0][0]
                    success_rxns += 1
                except:
                    b=a
                try:
                    c=rxn.RunReactants((b,))[0][0]
                    success_rxns += 1
                except:
                    c=b
                try:
                    d=rxn2.RunReactants((c,))[0][0]
                    success_rxns += 1
                except:
                    d=c
                try:  
                    e=rxn2.RunReactants((d,))[0][0]
                    success_rxns += 1
                except:
                    e=d
                if success_rxns == 2:
                    monomers2.append(e)
            return monomers2
        else:
            return []
    else:
        return []
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

def is_symmetric2(mol,group):
    ru.mol_with_partial_charge(mol,supress_output=True)
    if type(group) == str:
        group=Chem.MolFromSmiles(group)
    if type(mol) == str:
        mol=Chem.MolFromSmiles(mol)
    if type(group) == list:
        if mol.GetAtoms()[group[0]].GetProp('_GasteigerCharge') == mol.GetAtoms()[group[1]].GetProp('_GasteigerCharge'):
            return True
        else:
            return False
    matches = mol.GetSubstructMatches(group)
    if len(matches) == 2:
        n = group.GetNumAtoms()
        charges1 = []
        charges2 = []
        for i in range(n):
            charges1.append(mol.GetAtoms()[matches[0][i]].GetProp('_GasteigerCharge'))
            charges2.append(mol.GetAtoms()[matches[1][i]].GetProp('_GasteigerCharge'))
        if charges1 == charges2:
            return True
        else:
            return False
    else:
        return False