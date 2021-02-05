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

def frp_depolymerize(lp,sc=None,strict=True):
    '''
    Determine if this polymer, mol (either SMILES string or rdkit mol object), is a candidate for free-radical polymerization. If 'strict' is True, then some heuristics will be enforced based on the suggestions of Dr. Sotzing.
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
        
    if strict:
        main_chain_patt1 = Chem.MolFromSmarts('[#0]C[CH2][#0]')
        main_chain_patt2 = Chem.MolFromSmiles('*C=[CH]*')
        main_chain_patt3 = Chem.MolFromSmarts('[#0][CR][CHR]=[CHR][CR][#0]')
    else:
        main_chain_patt1 = Chem.MolFromSmiles('*CC*')
        main_chain_patt2 = Chem.MolFromSmiles('*C=C*')
        main_chain_patt3 = Chem.MolFromSmarts('[#0][CR][CR]=[CR][CR][#0]')

    
    side_chain_patt1 = Chem.MolFromSmiles('C=C')
    side_chain_patt2 = Chem.MolFromSmiles('C#C')
    side_chain_patt3 = Chem.MolFromSmarts('[OH]')
    side_chain_patt4 = Chem.MolFromSmarts('[CR]=[CR][CR]=[CR]')
    side_chain_patt5 = Chem.MolFromSmarts('[c]')

    if sc is None:
        sc = lp.SideChainMol()
    if sc is not None:
        sc_matches = sc.HasSubstructMatch(side_chain_patt1) or sc.HasSubstructMatch(side_chain_patt2) or sc.HasSubstructMatch(side_chain_patt3) or sc.HasSubstructMatch(side_chain_patt4) or sc.HasSubstructMatch(side_chain_patt5)
    else:
        sc_matches = False #if there is no side-chain mol then there can't be any matches
    
    if not sc_matches:
        mc_match1 = lp.mol.GetSubstructMatch(main_chain_patt1)
        mc_match2 = lp.mol.HasSubstructMatch(main_chain_patt2)
        mc_match3 = lp.mol.GetSubstructMatches(main_chain_patt3)
        
        if len(mc_match3) == 1:
            rxn = Chem.AllChem.ReactionFromSmarts('[*:1][CR:3]([#0:2])[CHR:4]=[CHR:5][CR:6]([#0:7])[*:9]>>[*:1][CR:3]=[CR:4][CR:5]=[CR:6][*:9]')
            prods = rxn.RunReactants((lp.mol,))
            if len(prods) == 0:
                return None
            new_mol = prods[0][0]
            try:
                Chem.SanitizeMol(new_mol)
                return [new_mol]
            except:
                return None

        if len(mc_match1) > 0:
            if strict:
                if lp.mol.GetAtoms()[mc_match1[1]].GetNumImplicitHs() < 1: #ceiling temperature AND ring consideration
                    return None
            
            em = Chem.EditableMol(lp.mol)
            em.RemoveBond(lp.connector_inds[0],lp.connector_inds[1])
            em.AddBond(lp.connector_inds[0],lp.connector_inds[1],Chem.BondType.DOUBLE) #replace single bond w/ double
            em.RemoveAtom(lp.star_inds[1])
            em.RemoveAtom(lp.star_inds[0]) #remove connection points
            
            try:
                new_mol = em.GetMol()
                Chem.SanitizeMol(new_mol)
                if strict: #if strict then presence of [CH2] has already been enforced
                    return [new_mol]
                else:
                    em = Chem.EditableMol(new_mol)
                    #lp.delStarMol()
                    inds = new_mol.GetSubstructMatch(Chem.MolFromSmiles('C=C'))
                    em.RemoveBond(inds[0],inds[1])
                    m=em.GetMol()
                    frags = Chem.GetMolFrags(m, asMols=True)
                    Chem.AllChem.EmbedMolecule(frags[0])
                    Chem.AllChem.EmbedMolecule(frags[1])
                    volumes = np.array([Chem.AllChem.ComputeMolVolume(frags[0]),Chem.AllChem.ComputeMolVolume(frags[1])])
                    if np.min(volumes) < 20.5: #35.7 is the volume of C(F)(F). 20.5 is the volume of C(H)(H)
                        return [new_mol]
                    else:
                        #print(np.min(volumes))
                        return None
            except:
                return None
        elif mc_match2:
            em = Chem.EditableMol(lp.mol)
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
                if strict: #CH2 already checked for if strict turned on
                    return [new_mol]
                else: 
                    frags = Chem.GetMolFrags(m, asMols=True)
                    Chem.AllChem.EmbedMolecule(frags[0])
                    Chem.AllChem.EmbedMolecule(frags[1])
                    volumes = np.array([Chem.AllChem.ComputeMolVolume(frags[0]),Chem.AllChem.ComputeMolVolume(frags[1])])
                    if np.min(volumes) < 20.5: #35.7 is the volume of C(F)(F). 20.43 is the volume of C(H)(H)
                        return [new_mol]
                    else:
                        return None
            except:
                return None
        else:
            return None
    else:
        return None

def ox_depolymerize(lp):
    '''
    Retrosynthesis of an oxidative depolymerization reaction
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)

    if lp.mol.HasSubstructMatch( Chem.MolFromSmarts('[#0][R]') ): #Most mol will fail. Necessary but not sufficient condition. Added for efficiency purposes
        pi_bonds = set([b.GetIdx() for b in lp.mol.GetBonds() if b.GetBondTypeAsDouble() > 1 and b.GetBeginAtom().GetSymbol() != 'O' and b.GetEndAtom().GetSymbol() != 'O'])
        c1_bonds = [b.GetIdx() for b in lp.mol.GetAtoms()[max(lp.connector_inds)].GetBonds()] #bonds of connector 1
        c2_bonds = [b.GetIdx() for b in lp.mol.GetAtoms()[min(lp.connector_inds)].GetBonds()] #bonds of connector 2    
        if not any([b in pi_bonds for b in c1_bonds]) or not any([b in pi_bonds for b in c2_bonds]): #if connectors don't have pi bonds return None
            return None
        ri = lp.mol.GetRingInfo()
        try:
            c1_ring = [ring for ring in ri.BondRings() if len(set(ring).intersection(c1_bonds)) > 0][0]
            c2_ring = [ring for ring in ri.BondRings() if len(set(ring).intersection(c2_bonds)) > 0][0]
        except: #if connectors are not in rings then return None
            return None
        if set(c1_ring).union(c2_ring) != pi_bonds: #if any pi-bonds exist outside of connector rings return None
            return None
        new_mol = lp.delStarMol() #make would-be monomer
        if retro.is_symmetric(new_mol,group=lp.delStarMolInds):
            return [ru.mol_without_atom_index(new_mol)]
    else:
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
                charges = retro.getCharges(new_mol,[mon_matches[0][0],mon_matches[0][1],mon_matches[1][0], \
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

def admet_depolymerize(lp):
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    patt = Chem.MolFromSmiles('C=C')    
    pm = lp.PeriodicMol()
    mc = lp.MainChainMol()
    pm_match = pm.GetSubstructMatches(patt)
    if mc.HasSubstructMatch(patt) and len(pm_match)==1:
        em = Chem.EditableMol(pm.mol)
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


ro_linkages = {
    'lactam':Chem.MolFromSmarts('[CR][NR][CR](=O)'), 
    'lactone':Chem.MolFromSmarts('[CR][OR][CR](=O)'),
    'cyclic_ether':Chem.MolFromSmarts('[CR][OR][CR]'),
    'imine':Chem.MolFromSmarts('[CR][NR][CR]'),
    'ncie':Chem.MolFromSmarts('[NR][CR0](=O)'), #ncie = eNdo Cyclic Imino Ether
    'cyclic_sulfide':Chem.MolFromSmarts('[CR][SR]')
}

def ro_depolymerize(lp, ro_linkage_key, pm=None, selectivity=False):
    '''
    Return (polymer,monomer) if ring-opening polymerization is possible. selectivity = True means a selectivity check will occur. selectivity appears to be violated in 10.1002/pola.10090, p.194, Scheme 2, 5 so default is to not check for selectivity at the moment.
    lactam test SMILES: [*]C(NC(=O)C)CCC(=O)N[*]
    lactone test SMILES: [*]CCCC(=O)O[*]
    cyclic_ether test SMILES: [*]COCOCO[*]
    imine test SMILES: [*]CCN([*])C
    ncie test SMILES: [*]N(C=O)CC(=O)[*]
    cyclic_sulfide test SMILES: [*]CCOCCSS[*]
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    
    try:
        linkage = ro_linkages[ro_linkage_key]
        
        #look for hydroxyl group
        if ro_linkage_key in ['lactone', 'cyclic_ether']:
            oh_mol = Chem.MolFromSmarts('[OH]')
            if lp.mol.HasSubstructMatch(oh_mol):
                return None
        
        if pm is None:
            pm = lp.PeriodicMol()
            if pm is not None:
                pm.GetSSSR()
        pm_matches = pm.GetSubstructMatches(linkage)
        ar = [set(ring) for ring in pm.GetRingInfo().AtomRings()]
        if max([len(r) for r in ar]) > 9: #don't allow large rings
            return None
        pm_match_set = [set(match) for match in pm_matches]
        if ro_linkage_key not in ['cyclic_ether', 'cyclic_sulfide']: #most rings will only polymerize w/ one linkage
            if len(pm_matches) != 1: #the pm should have just one match of linkage
                return None

        else: #few rings can polymerize w/ more than one linkage but all linkages must be in same ring.
            if len(pm_matches) == 0: 
                return None

        #check selectivity
        if selectivity:
            pm_match_set = set(ru.flatten_ll(pm_matches))
            other_linkage_keys = [k for k in ro_linkages.keys() if k != ro_linkage_key]
            for k in other_linkage_keys:
                k_matches = set(ru.flatten_ll(pm.GetSubstructMatches( ro_linkages[k] )))
                if len(  k_matches.difference(pm_match_set) ) > 0: 
                    return None

        reduced_pm = None
        if ro_linkage_key in ['lactam','lactone']:
            reduced_pm = lp.AlphaMol().PeriodicMol()
        elif ro_linkage_key in ['ncie']:
            reduced_pm = lp.BetaMol().PeriodicMol()
        else:
            reduced_pm = lp.MainChainMol().PeriodicMol()
        reduced_pm.GetSSSR()
        reduced_pm_matches = reduced_pm.GetSubstructMatches(linkage)

        #1,2- and 2,3-disubstituted aziridines do not polymerize; 1-and 2-substituted aziridines undergo polymerization
        #Source: Odian, Principles of Polymerization, 4E, p.587
        if ro_linkage_key == 'imine':
            aziridine = Chem.MolFromSmarts('[#6]1-[#7]-[#6]-1')
            if reduced_pm.HasSubstructMatch(aziridine):
                if pm.HasSubstructMatch( Chem.MolFromSmarts('[#6H2]1-[#7](*)-[#6H2]-1') ) or pm.HasSubstructMatch( Chem.MolFromSmarts('[#6]1(*)-[#7H]-[#6H2]-1') ): #1 and 2 substitution works but nothing else
                    pass
                else:
                    return None

        if len(reduced_pm_matches) == len(pm_matches): #the only matches should exist on the reduced_pm
            if ro_linkage_key == 'ncie': #ncie requires some edits to the pm. Source: 10.1002/pola.10090, p.194
                rxn = Chem.AllChem.ReactionFromSmarts('[CR:1][NR:2]([CR0:4]=[O:5])[CR:3]>>[CR:1][NR:2]=[CR:4][OR:5][CR:3]')
                ps=rxn.RunReactants((pm.mol,))
                valid_ps = []
                for p in ps:
                    try:
                        Chem.SanitizeMol(p[0])
                        valid_ps.append(p[0])
                    except:
                        pass
                if len(valid_ps) == 0:
                    return None
                else:
                    return valid_ps
            else:
                return [pm.mol]
        else:
            return None
    except:
        return None

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

def cooh_nh2_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ COOH and another monomer w/ OH
    '''   
    em = Chem.EditableMol(pm.mol)
    a_ir1,a_in,a_ic,a_io,a_ir2 = match_pair[0]
    b_ir1,b_in,b_ic,b_io,b_ir2 = match_pair[1]
    em.RemoveBond(a_ic,a_in)
    em.RemoveBond(b_ic,b_in)
    o1 = em.AddAtom(Chem.AtomFromSmiles('O'))
    o2 = em.AddAtom(Chem.AtomFromSmiles('O'))
    em.AddBond(o1,a_ic,Chem.BondType.SINGLE)
    em.AddBond(o2,b_ic,Chem.BondType.SINGLE)
    new_mol=em.GetMol()
    Chem.SanitizeMol(new_mol)
    frag_ids = Chem.GetMolFrags(new_mol)
    
    if len(frag_ids) == 2:
        frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
        if frag_mols[0].HasSubstructMatch(Chem.MolFromSmarts('[NH2]')):
            nh2_ind = 0
            cooh_ind = 1
        else:
            nh2_ind = 1
            cooh_ind = 0
        nh2_mol = frag_mols[nh2_ind]
        cooh_mol = frag_mols[cooh_ind]
        return [(new_mol, nh2_mol, cooh_mol)]
    else:
        return []

def cooh_oh_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ COOH and another monomer w/ OH
    '''   
    em = Chem.EditableMol(pm.mol)
    ai_r1,ai_c,ai_o_dbl,ai_o,ai_r2 = match_pair[0]
    bi_r1,bi_c,bi_o_dbl,bi_o,bi_r2 = match_pair[1]
    em.RemoveBond(ai_o,ai_r2)
    em.RemoveBond(bi_o,bi_r2)
    i_o1 = em.AddAtom(Chem.AtomFromSmiles('O'))
    i_o2 = em.AddAtom(Chem.AtomFromSmiles('O'))
    em.AddBond(ai_r2,i_o1,Chem.BondType.SINGLE)
    em.AddBond(bi_r2,i_o2,Chem.BondType.SINGLE)
    new_mol=em.GetMol()
    Chem.SanitizeMol(new_mol)
    frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
    if len(frag_mols) == 2:
        oh_mol1 = frag_mols[0]
        oh_mol2 = frag_mols[1]
        return [(new_mol, oh_mol1, oh_mol2)]
    else:
        return []

def oh_cl_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ OH and another monomer w/ Cl
    '''
    new_mols = []
    cl_mols = []
    oh_mols = []
    
    ### make the first monomer set ###
    em = Chem.EditableMol(pm.mol)
    ai_r,ai_o_right,ai_c,i_dbl0,ai_o_left = match_pair[0]
    bi_r,bi_o_right,bi_c,bi_dbl0,bi_o_left = match_pair[1]
    em.RemoveBond(ai_o_left,ai_c)
    em.RemoveBond(bi_o_left,bi_c)
    i_cl1 = em.AddAtom(Chem.AtomFromSmiles('Cl'))
    i_cl2 = em.AddAtom(Chem.AtomFromSmiles('Cl'))
    em.AddBond(ai_c,i_cl1,Chem.BondType.SINGLE)
    em.AddBond(bi_c,i_cl2,Chem.BondType.SINGLE)
    new_mol1=em.GetMol()
    Chem.SanitizeMol(new_mol1)

    ### make the second monomer set ###
    em2 = Chem.EditableMol(pm.mol)
    em2.RemoveBond(ai_o_right,ai_r)
    em2.RemoveBond(bi_o_right,bi_r)
    i_cl1 = em2.AddAtom(Chem.AtomFromSmiles('Cl'))
    i_cl2 = em2.AddAtom(Chem.AtomFromSmiles('Cl'))
    em2.AddBond(ai_r,i_cl1,Chem.BondType.SINGLE)
    em2.AddBond(bi_r,i_cl2,Chem.BondType.SINGLE)
    new_mol2 = em2.GetMol()
    Chem.SanitizeMol(new_mol2)   

    for new_mol in (new_mol1,new_mol2):
        frag_ids = Chem.GetMolFrags(new_mol, asMols=False)
        if len(frag_ids) == 2:
            frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
            if frag_mols[0].HasSubstructMatch(Chem.MolFromSmiles('Cl')):
                cl_ind = 0
                oh_ind = 1
            else:
                cl_ind = 1
                oh_ind = 0
            oh_mol = frag_mols[oh_ind]
            cl_mol = frag_mols[cl_ind]
            new_mols.append(new_mol)
            cl_mols.append(cl_mol)
            oh_mols.append(oh_mol)
    return list(zip(new_mols, cl_mols, oh_mols))

def nh2_nco_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create the possible constituent diisocyanates and a diamines
    '''
    new_mols = []
    nhx_mols = []
    nco_mols = []
    
    ### make the first monomer set ###
    em = Chem.EditableMol(pm.mol)
    a_ir1,a_inh,a_ic,a_io,a_in,a_ir = match_pair[0]
    b_ir1,b_inh,b_ic,b_io,b_in,b_ir = match_pair[1]
    em.RemoveBond(a_inh,a_ic)
    em.RemoveBond(b_inh,b_ic)
    #switch bond
    em.RemoveBond(a_in,a_ic)
    em.AddBond(a_in,a_ic,Chem.BondType.DOUBLE)
    #switch bond
    em.RemoveBond(b_in,b_ic)
    em.AddBond(b_in,b_ic,Chem.BondType.DOUBLE)

    new_mol1=em.GetMol()
    try:
        Chem.SanitizeMol(new_mol1)
    except:
        pass

    ### make the second monomer set ###
    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(a_in,a_ic)
    em.RemoveBond(b_in,b_ic)
    #switch bond
    em.RemoveBond(a_inh,a_ic)
    em.AddBond(a_inh,a_ic,Chem.BondType.DOUBLE)
    #switch bond
    em.RemoveBond(b_inh,b_ic)
    em.AddBond(b_inh,b_ic,Chem.BondType.DOUBLE)
    new_mol2 = em.GetMol()
    try:
        Chem.SanitizeMol(new_mol2) 
    except:
        pass  

    for new_mol in (new_mol1,new_mol2):
        try:
            frag_ids = Chem.GetMolFrags(new_mol, asMols=False)
            if len(frag_ids) == 2:
                frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
                if frag_mols[0].HasSubstructMatch(Chem.MolFromSmiles('N=C=O')):
                    nco_ind = 0
                    nhx_ind = 1
                else:
                    nco_ind = 1
                    nhx_ind = 0
                nhx_mol = frag_mols[nhx_ind]
                nco_mol = frag_mols[nco_ind]
                new_mols.append(new_mol)
                nhx_mols.append(nhx_mol)
                nco_mols.append(nco_mol)
        except:
            pass
    return list(zip(new_mols, nhx_mols, nco_mols))

def nh_nco_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create the possible constituent diisocyanates and a diamines
    '''
    new_mols = []
    nhx_mols = []
    nco_mols = []
    
    a_inh,a_ic,_,a_in = match_pair[0]
    b_inh,b_ic,_,b_in = match_pair[1]

    ### make the first monomer set ###
    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(a_inh,a_ic)
    em.RemoveBond(b_inh,b_ic)
    #switch bond
    em.RemoveBond(a_in,a_ic)
    em.AddBond(a_in,a_ic,Chem.BondType.DOUBLE)
    #switch bond
    em.RemoveBond(b_in,b_ic)
    em.AddBond(b_in,b_ic,Chem.BondType.DOUBLE)

    new_mol1=em.GetMol()
    try:
        Chem.SanitizeMol(new_mol1)
    except:
        pass

    ### make the second monomer set ###
    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(a_in,a_ic)
    em.RemoveBond(b_in,b_ic)
    #switch bond
    em.RemoveBond(a_inh,a_ic)
    em.AddBond(a_inh,a_ic,Chem.BondType.DOUBLE)
    #switch bond
    em.RemoveBond(b_inh,b_ic)
    em.AddBond(b_inh,b_ic,Chem.BondType.DOUBLE)
    new_mol2 = em.GetMol()
    try:
        Chem.SanitizeMol(new_mol2) 
    except:
        pass  

    for new_mol in (new_mol1,new_mol2):
        try:
            frag_ids = Chem.GetMolFrags(new_mol, asMols=False)
            if len(frag_ids) == 2:
                frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
                if frag_mols[0].HasSubstructMatch(Chem.MolFromSmiles('N=C=O')):
                    nco_ind = 0
                    nhx_ind = 1
                else:
                    nco_ind = 1
                    nhx_ind = 0
                nhx_mol = frag_mols[nhx_ind]
                nco_mol = frag_mols[nco_ind]
                new_mols.append(new_mol)
                nhx_mols.append(nhx_mol)
                nco_mols.append(nco_mol)
        except:
            pass
    return list(zip(new_mols, nhx_mols, nco_mols))    

def cl_NaO_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ COOH and another monomer w/ OH.
    Source: Odian, Principles of Polymerization, 4E, p.149
    '''   
    _,_,_,_,_,ai_c,ai_o,_,_ = match_pair[0]
    _,_,_,_,_,bi_c,bi_o,_,_ = match_pair[1]
    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(ai_c,ai_o)
    em.RemoveBond(bi_c,bi_o)
    #add atoms
    cl1=em.AddAtom(Chem.AtomFromSmiles('Cl'))
    cl2=em.AddAtom(Chem.AtomFromSmiles('Cl'))
    na1=em.AddAtom(Chem.AtomFromSmiles('[Na]'))
    na2=em.AddAtom(Chem.AtomFromSmiles('[Na]'))
    #add bonds
    em.AddBond(ai_o,na1,Chem.BondType.SINGLE)
    em.AddBond(bi_o,na2,Chem.BondType.SINGLE)
    em.AddBond(ai_c,cl1,Chem.BondType.SINGLE)
    em.AddBond(bi_c,cl2,Chem.BondType.SINGLE)
    #get mol
    new_mol=em.GetMol()
    Chem.SanitizeMol(new_mol)
    frag_ids = Chem.GetMolFrags(new_mol)
    if len(frag_ids) == 2:
        frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
        if frag_mols[0].HasSubstructMatch(Chem.MolFromSmarts('Cl')):
            cl_ind = 0
            na_ind = 1
        else:
            cl_ind = 1
            na_ind = 0
        cl_mol = frag_mols[cl_ind]
        na_mol = frag_mols[na_ind]
        return [(new_mol, cl_mol, na_mol)]
    else:
        return []

def cooh_nh2_oh_ar_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ COOH and another monomer w/ both OH and NH2. 'ar' denotes that the 5-membered ring in polymer is aromatic 
    Source: Odian, Principles of Polymerization, 4E, p.162
    '''   
    a_ic,a_in,_,_,a_io = match_pair[0]
    b_ic,b_in,_,_,b_io = match_pair[1]
    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(a_ic,a_io)
    em.RemoveBond(b_ic,b_io)
    em.RemoveBond(a_ic,a_in)
    em.RemoveBond(b_ic,b_in)

    #remove and replace aromatic bonds
    em.ReplaceAtom(a_ic,Chem.AtomFromSmiles('C'))
    em.ReplaceAtom(b_ic,Chem.AtomFromSmiles('C'))

    em.ReplaceAtom(a_in,Chem.AtomFromSmiles('N'))
    em.ReplaceAtom(b_in,Chem.AtomFromSmiles('N'))

    em.ReplaceAtom(a_io,Chem.AtomFromSmiles('O'))
    em.ReplaceAtom(b_io,Chem.AtomFromSmiles('O'))

    #add =O(OH) 
    dblO1 = em.AddAtom(Chem.AtomFromSmiles('O'))
    dblO2 = em.AddAtom(Chem.AtomFromSmiles('O'))
    o1 = em.AddAtom(Chem.AtomFromSmiles('O'))
    o2 = em.AddAtom(Chem.AtomFromSmiles('O'))
    em.AddBond(a_ic,dblO1,Chem.BondType.DOUBLE)
    em.AddBond(b_ic,dblO2,Chem.BondType.DOUBLE)
    em.AddBond(a_ic,o1,Chem.BondType.SINGLE)
    em.AddBond(b_ic,o2,Chem.BondType.SINGLE)

    new_mol=em.GetMol()
    Chem.SanitizeMol(new_mol)

    frag_ids = Chem.GetMolFrags(new_mol)
    if len(frag_ids) == 2:
        frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
        if frag_mols[0].HasSubstructMatch(Chem.MolFromSmarts('[NH2]')):
            nh2_ind = 0
            cooh_ind = 1
        else:
            nh2_ind = 1
            cooh_ind = 0
        nh2_mol = frag_mols[nh2_ind]
        cooh_mol = frag_mols[cooh_ind]
        return [(new_mol, cooh_mol, nh2_mol)]
    else:
        return []

def cooh_nh2_oh_al_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ COOH and another monomer w/ both OH and NH2. 'al' denotes that the 5-membered ring in polymer is aliphatic 
    Source: Odian, Principles of Polymerization, 4E, p.162
    '''   
    a_ic,a_in,_,_,a_io = match_pair[0]
    b_ic,b_in,_,_,b_io = match_pair[1]
    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(a_ic,a_io)
    em.RemoveBond(b_ic,b_io)
    em.RemoveBond(a_ic,a_in)
    em.RemoveBond(b_ic,b_in)

    #add =O(OH) 
    dblO1 = em.AddAtom(Chem.AtomFromSmiles('O'))
    dblO2 = em.AddAtom(Chem.AtomFromSmiles('O'))
    o1 = em.AddAtom(Chem.AtomFromSmiles('O'))
    o2 = em.AddAtom(Chem.AtomFromSmiles('O'))
    em.AddBond(a_ic,dblO1,Chem.BondType.DOUBLE)
    em.AddBond(b_ic,dblO2,Chem.BondType.DOUBLE)
    em.AddBond(a_ic,o1,Chem.BondType.SINGLE)
    em.AddBond(b_ic,o2,Chem.BondType.SINGLE)

    new_mol=em.GetMol()
    Chem.SanitizeMol(new_mol)

    frag_ids = Chem.GetMolFrags(new_mol)
    if len(frag_ids) == 2:
        frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
        if frag_mols[0].HasSubstructMatch(Chem.MolFromSmarts('[NH2]')):
            nh2_ind = 0
            cooh_ind = 1
        else:
            nh2_ind = 1
            cooh_ind = 0
        nh2_mol = frag_mols[nh2_ind]
        cooh_mol = frag_mols[cooh_ind]
        return [(new_mol, cooh_mol, nh2_mol)]
    else:
        return []

def oh_oh_xo_edit_6m(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ 2 -OH group on each side and another monomer w/ X=O. 
    6m denotes that the spiro-rings are 6-membered 
    Source: 10.1039/C4PY00178H (DOI), p. 3216
    '''   
    a_ilw, a_ilto, _, _, _, a_ilbo, _, a_irbo, a_irw, a_irto, _ = match_pair
    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(a_ilw,a_ilto)
    em.RemoveBond(a_ilw,a_ilbo)
    em.RemoveBond(a_irw,a_irto)
    em.RemoveBond(a_irw,a_irbo)
    o1 = em.AddAtom(Chem.AtomFromSmiles('O'))
    o2 = em.AddAtom(Chem.AtomFromSmiles('O'))
    em.AddBond(a_ilw,o1,Chem.BondType.DOUBLE)
    em.AddBond(a_irw,o2,Chem.BondType.DOUBLE)
    new_mol = em.GetMol()
    Chem.SanitizeMol(new_mol)
    frag_ids = Chem.GetMolFrags(new_mol)
    if len(frag_ids) == 2:
        frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
        if frag_mols[0].HasSubstructMatch(Chem.MolFromSmarts('[OH]')):
            oh_ind = 0
            xo_ind = 1
        else:
            oh_ind = 1
            xo_ind = 0
        oh_mol = frag_mols[oh_ind]
        xo_mol = frag_mols[xo_ind]
        return [(new_mol, oh_mol, xo_mol)]
    else:
        return []       

def oh_oh_f_f_edit(pm,match_pair):
    '''
    Take in an editable mol and match_pair and perform the bond breakage to create one monomer w/ 2 -OH group on each side and another monomer w/ 2 -F on each side. 
    Source: 10.1039/C9TA04844H, p. 22437
    '''   
    return []

def nh2_xox_edit(pm,match_pair):
    '''
    Source: Reynolds Class Notes, 8-25-20, p.2
    '''
    _,_,a_ioh,a_ic1,a_ic2,a_ico,_,a_in = match_pair[0]
    _,_,b_ioh,b_ic1,b_ic2,b_ico,_,b_in = match_pair[1]
    bond_type = pm.mol.GetBondBetweenAtoms(a_ic1,a_ic2).GetBondType()

    em = Chem.EditableMol(pm.mol)
    em.RemoveBond(a_ico,a_in)
    em.RemoveBond(b_ico,b_in)

    em.AddBond(a_ico,a_ioh,bond_type)
    em.AddBond(b_ico,b_ioh,bond_type)

    new_mol=em.GetMol()
    Chem.SanitizeMol(new_mol)
    
    frag_ids = Chem.GetMolFrags(new_mol)
    if len(frag_ids) == 2:
        frag_mols = Chem.GetMolFrags(new_mol, asMols=True)
        if frag_mols[0].HasSubstructMatch(Chem.MolFromSmarts('[NH2]')):
            nh2_ind = 0
            xox_ind = 1
        else:
            nh2_ind = 1
            xox_ind = 0
        nh2_mol = frag_mols[nh2_ind]
        xox_mol = frag_mols[xox_ind]
        return [(new_mol, nh2_mol, xox_mol)]
    else:
        return []        

sg_rxns = { #SMARTS of polymer linkage: [(g1,g2,edit_function),(g3,g4,edit_function)]. Order matters. Do not change!
    '*O[#6](=O)O': [(Chem.MolFromSmiles('Cl'),Chem.MolFromSmarts('[OH]'),oh_cl_edit)],
    '*[#6](=O)O*': [(Chem.MolFromSmarts('[OH]'),Chem.MolFromSmarts('[OH]'),cooh_oh_edit)],
    '*[NH][#6](=O)*': [(Chem.MolFromSmarts('[NH2]'),Chem.MolFromSmarts('C(=O)[OH]'),cooh_nh2_edit)],
    '*[NH][#6](=O)[NH]*': [(Chem.MolFromSmarts('[NH2]'),Chem.MolFromSmarts('N=C=O'),nh2_nco_edit)],
    '[NH][#6](=O)N': [(Chem.MolFromSmarts('[NH]'),Chem.MolFromSmarts('N=C=O'),nh_nco_edit)],
    'O=Cc1ccc(O)cc1': [(Chem.MolFromSmiles('Cl'),Chem.MolFromSmiles('O[Na]'),cl_NaO_edit)],
    'C1=NccO1': [([Chem.MolFromSmarts('C(=O)[OH]')],[Chem.MolFromSmarts('[NH2]'),Chem.MolFromSmarts('[OH]')],cooh_nh2_oh_ar_edit)],
    'C1=NCCO1': [([Chem.MolFromSmarts('C(=O)[OH]')],[Chem.MolFromSmarts('[NH2]'),Chem.MolFromSmarts('[OH]')],cooh_nh2_oh_al_edit)],
    '*1-[#8]-[#6]-[#6]2(-[#6]-[#8]-1)-[#6]-[#8]-*-[#8]-[#6]-2': [(Chem.MolFromSmarts('[OH]'),Chem.MolFromSmarts('*=O'),oh_oh_xo_edit_6m)],
    '[#6R0](=O)([OH])[C,c][C,c][CR1](=O)[NR1]': [(Chem.MolFromSmarts('[NH2,nH2]'),Chem.MolFromSmarts('*[OR,oR]*'),nh2_xox_edit)]
}


def sg_depolymerize(lp,polymer_linkage,rxn_info,pm=None,debug=False,greedy=True):
    '''
    Return the monomers (one w/ fxnl group g1 and the other w/ g2) that could undergo a step-growth polymerization to form mol. For now only works when input mol has only one repeat unit. Using Chris's code may help this.
    '''
    g1,g2,edit_function=rxn_info[0],rxn_info[1],rxn_info[2]
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)

    if pm is None:
        pm = lp.PeriodicMol()
    if pm is None: #periodization failed
        if debug:
            print('None1')
        return None
    try: #sometimes g1 and g2 are given as lists. If so they will fail below.
        if pm.HasSubstructMatch(g1) or pm.HasSubstructMatch(g2): #chain should not have same functional groups we want to react
            if edit_function not in [nh_nco_edit,nh2_xox_edit]: #but there are exceptions
                if debug:
                    print('None2')
                return None
        g1 = [g1] #do this so symmetry check will have an iterable
        g2 = [g2] #do this so symmetry check will have an iterable
    except:
        if any([pm.HasSubstructMatch(x) for x in g1] + [pm.HasSubstructMatch(x) for x in g2]): #chain should not have same functional groups we want to react
            if edit_function not in [nh_nco_edit,nh2_xox_edit]: #but there are exceptions
                if debug:
                    print('None3')
                return None        
    try:
        matches=pm.GetSubstructMatches(polymer_linkage)
    except:
        pm.GetSSSR()
        matches=pm.GetSubstructMatches(polymer_linkage)

    if 'oh_oh_xo_edit' in str(edit_function): 
        match_pairs = list(matches)
    else:
        match_pairs = list(itertools.combinations(matches, 2))
    new_mols = []
    for match_pair in match_pairs:
        try:
            new_mols_info = edit_function(pm,match_pair)
        except:
            if greedy:
                new_mols_info = []
            else:
                print('sg_failed for %s' %lp.SMILES)
                return None
        if new_mols_info is not []:
            for new_mol_info in new_mols_info:
                new_mol = new_mol_info[0]
                g1_mol = new_mol_info[1]
                g2_mol = new_mol_info[2]
                if debug:
                    print( Chem.MolToSmiles(g1_mol),Chem.MolToSmiles(g2_mol) )
                if 'oh_oh_xo_edit' in str(edit_function): #g1_mol is symmetric w.r.t [OH] by construction
                    if all([retro.is_symmetric(g2_mol,x) for x in g2]):
                        new_mols.append(new_mol)
                else:
                    if all([retro.is_symmetric(g1_mol,x) for x in g1] + [retro.is_symmetric(g2_mol,x) for x in g2]): #symmetry function includes a check to make sure there are only 2 matches
                        new_mols.append(new_mol)
    if new_mols == []:
        if debug:
            print('None4')
        return None
    else:
        return new_mols
    #return new_mols_info
