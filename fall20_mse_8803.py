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

### set up scscore ###
from scscore import standalone_model_numpy as sc
sc_model = sc.SCScorer()
sc_model.restore('/data/rgur/retrosynthesis/scscore/models/full_reaxys_model_1024uint8/model.ckpt-10654.as_numpy.json.gz')

def frp_possible(mol):
    '''
    Determine if this polymer, mol (either SMILES string or rdkit mol object), is a candidate for free-radical polymerization.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    main_chain_patt = Chem.MolFromSmarts('[#0]C[CH2][#0]') #ensure at least one carbon is unsubstituted
    side_chain_patt = Chem.MolFromSmiles('C=C')
    
    return mol.HasSubstructMatch(main_chain_patt) & ( not mol.HasSubstructMatch(side_chain_patt) )

def frp_depolymerize(mol,strict=True):
    '''
    Determine if this polymer, mol (either SMILES string or rdkit mol object), is a candidate for free-radical polymerization. If 'strict' is True, then some heuristics will be enforced based on the suggestions of Dr. Sotzing.
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
        
    #main_chain_patt1 = Chem.MolFromSmarts('[#0]C[CH2][#0]') #ensure at least one carbon is unsubstituted
    #main_chain_patt2 = Chem.MolFromSmarts('[#0]C=[CH][#0]') #ensure at least one carbon is unsubstituted
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

    sc_mol = ru.LinearPol(mol).SideChainMol()
    if sc_mol is not None:
        sc_matches = sc_mol.HasSubstructMatch(side_chain_patt1) or sc_mol.HasSubstructMatch(side_chain_patt2) or mol.HasSubstructMatch(side_chain_patt3) or sc_mol.HasSubstructMatch(side_chain_patt4) or sc_mol.HasSubstructMatch(side_chain_patt5)
    else:
        sc_matches = False #if there is no side-chain mol then there can't be any matches
    
    if not sc_matches:
        lp = ru.LinearPol(mol)
        
        mc_match1 = mol.GetSubstructMatch(main_chain_patt1)
        mc_match2 = mol.HasSubstructMatch(main_chain_patt2)
        mc_match3 = mol.GetSubstructMatches(main_chain_patt3)
        
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
                if mol.GetAtoms()[mc_match1[1]].GetNumImplicitHs() < 1: #cieling temperature AND ring consideration
                    return None
            
            em = Chem.EditableMol(mol)
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

def ox_depolymerize(mol):
    '''
    Retrosynthesis of an oxidative depolymerization reaction
    '''
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    if mol.HasSubstructMatch( Chem.MolFromSmarts('[#0][R]') ): #Most mol will fail. Necessary but not sufficient condition. Added for efficiency purposes
        lp = ru.LinearPol(mol)
        pi_bonds = set([b.GetIdx() for b in mol.GetBonds() if b.GetBondTypeAsDouble() > 1])
        c1_bonds = [b.GetIdx() for b in lp.mol.GetAtoms()[max(lp.connector_inds)].GetBonds()] #bonds of connector 1
        c2_bonds = [b.GetIdx() for b in lp.mol.GetAtoms()[min(lp.connector_inds)].GetBonds()] #bonds of connector 2    
        if not any([b in pi_bonds for b in c1_bonds]) or not any([b in pi_bonds for b in c2_bonds]): #if connectors don't have pi bonds return None
            return None
        ri = mol.GetRingInfo()
        try:
            c1_ring = [ring for ring in ri.BondRings() if len(set(ring).intersection(c1_bonds)) > 0][0]
            c2_ring = [ring for ring in ri.BondRings() if len(set(ring).intersection(c2_bonds)) > 0][0]
        except: #if connectors are not in rings then return None
            return None
        if set(c1_ring).union(c2_ring) != pi_bonds: #if any pi-bonds exist outside of connector rings return None
            return None
        new_mol = lp.delStarMol() #make would-be monomer
        if is_symmetric2(new_mol,group=lp.delStarMolInds):
            return [ru.mol_without_atom_index(new_mol)]
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

def admet_depolymerize(lp):
    if type(lp) == str: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    patt = Chem.MolFromSmiles('C=C')    
    if not frp_possible(lp.mol):
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
            
def ro_depolymerize2(lp):
    '''
    Return (polymer,monomer) if ring-opening polymerization is possible
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    
    patt1=Chem.MolFromSmiles('CNC(=O)')
    patt2=Chem.MolFromSmiles('COC(=O)')
    patt3=Chem.MolFromSmiles('COC')
    patt4=Chem.MolFromSmiles('CNC')
    patt5=Chem.MolFromSmiles('COC=N')
    patt_oh=Chem.MolFromSmarts('[OX2H]')
    try:
        #print('here1')
        pm = lp.PeriodicMol()
        #print('herea')
        am = lp.AlphaMol().PeriodicMol()
        #print('hereb')
        mcm = lp.MainChainMol().PeriodicMol()
        #print('herec')
        alpha_match_len = (len(am.GetSubstructMatches(patt1)),len(am.GetSubstructMatches(patt2)),
                          len(mcm.GetSubstructMatches(patt3)),
                          len(mcm.GetSubstructMatches(patt4)),
                          len(mcm.GetSubstructMatches(patt5)),
                          len(am.GetSubstructMatches(patt5))) #exo-imino
        #print('hered')
        if alpha_match_len[0] == 1 or alpha_match_len[1] == 1 or alpha_match_len[4] == 1 or alpha_match_len[5] == 1:
            #print('here2')
            alpha_match_len = list(alpha_match_len)
            alpha_match_len[2] = 0 #if we have an ester third group should be removed
            alpha_match_len[3] = 0
            alpha_match_len = tuple(alpha_match_len)
        if alpha_match_len[4] == 1:
            alpha_match_len[5] == 0 #endo-inimo overshadows exo-imino
        argmax = np.argmax(alpha_match_len)
        if sorted(alpha_match_len) == [0, 0, 0, 0, 0, 1]: #make sure groups exist on alpha_chain
            #print('here3')
            lactone = len(pm.GetSubstructMatches(patt2))
            cyc_ether = len(pm.GetSubstructMatches(patt3))
            endo_imino_cyc_ether = len(pm.GetSubstructMatches(patt5))
            if pm.HasSubstructMatch(patt_oh) == True: 
                #print('here4')
                lactone = 0 #OH for lactone is no good
                cyc_ether = 0 #OH for cyclic ether is no good
                endo_imino_cyc_ether = 0 #OH for endo_imino_cyc_ether is no good
            tot_match_len = (len(pm.GetSubstructMatches(patt1)),lactone,cyc_ether,
                            len(pm.GetSubstructMatches(patt4)),endo_imino_cyc_ether,endo_imino_cyc_ether)
            if tot_match_len[argmax] == alpha_match_len[argmax]: #make sure no groups are on side chains
                #print('here5')
                return [pm]
            else:
                return None
        else:
            return None
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

def ro_depolymerize(lp, ro_linkage_key, selectivity=False):
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
        
        pm = lp.PeriodicMol()
        pm.GetSSSR()
        pm_matches = pm.GetSubstructMatches(linkage)
        ar = [set(ring) for ring in pm.GetRingInfo().AtomRings()]
        if max([len(r) for r in ar]) > 9: #don't allow large rings
            return None
        pm_match_set = [set(match) for match in pm_matches]
        match_ring = None
        if ro_linkage_key not in ['cyclic_ether', 'cyclic_sulfide']: #most rings will only polymerize w/ one linkage
            if len(pm_matches) != 1: #the pm should have just one match of linkage
                return None
            for ring in ar:            
                if match_ring is not None:
                    break
                for match in pm_match_set:
                    if len( match.intersection(ring) ) > 0:
                        match_ring = ring
                        break            
        else: #few rings can polymerize w/ more than one linkage but all linkages must be in same ring.
            if len(pm_matches) == 0: 
                return None
            n_ring_matches = 0
            for ring in ar:            
                for match in pm_match_set:
                    if len( match.intersection(ring) ) > 0:
                        n_ring_matches += 1
                        match_ring = ring
                        break
                if n_ring_matches > 1:
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
        
        em = Chem.EditableMol(pm.mol)

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


def sg_depolymerize(mol,polymer_linkage,rxn_info,debug=False):
    '''
    Return the monomers (one w/ fxnl group g1 and the other w/ g2) that could undergo a step-growth polymerization to form mol. For now only works when input mol has only one repeat unit. Using Chris's code may help this.
    '''
    g1,g2,edit_function=rxn_info[0],rxn_info[1],rxn_info[2]
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    lp = ru.LinearPol(mol)
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
        new_mols_info = edit_function(pm,match_pair)
        if new_mols_info is not []:
            for new_mol_info in new_mols_info:
                new_mol = new_mol_info[0]
                g1_mol = new_mol_info[1]
                g2_mol = new_mol_info[2]
                if debug:
                    print( Chem.MolToSmiles(g1_mol),Chem.MolToSmiles(g2_mol) )
                if 'oh_oh_xo_edit' in str(edit_function): #g1_mol is symmetric w.r.t [OH] by construction
                    if all([is_symmetric2(g2_mol,x) for x in g2]):
                        new_mols.append(new_mol)
                else:
                    if all([is_symmetric2(g1_mol,x) for x in g1] + [is_symmetric2(g2_mol,x) for x in g2]): #symmetry function includes a check to make sure there are only 2 matches
                        new_mols.append(new_mol)
    if new_mols == []:
        if debug:
            print('None4')
        return None
    else:
        return new_mols
    #return new_mols_info

func_chain_rxns = {
    'nitro_base':Chem.MolFromSmarts('n[*R0]'),
    'SO2_oxidation':Chem.MolFromSmarts('S(=O)(=O)')
    }

def func_chain_retro(lp,rxn):
    '''
    Functions to retrosynthetically functionalize chains
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    # if rxn == 'nitro_base':
    #     r_mol = lp.MainChainMol()#reduced mol 
    # if rxn != 'nitro_base':
    #     r_mol = r_mol.PeriodicMol()
    replace_group = func_chain_rxns[rxn]
    if rxn == 'nitro_base':
        sc_matches = lp.SideChainMol().GetSubstructMatches(replace_group)
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
    em = pm_to_lp_em(em,pm)
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
        em = pm_to_lp_em(em,pm)
        new_mols.append( em.GetMol() )
    return new_mols

elim_rxns = {
    'RCO2H': {'replace_patt': '[c,C;R][c,C;R;!H0]', 'edit_fxn': edit_RCO2H},
    'HCl': {'replace_patt': 'C=C', 'edit_fxn': edit_HCl},
    }

def pm_to_lp_em(em,pm):
    '''
    Add '*' for connection points at appropriate place of em
    '''
    star1 = em.AddAtom(Chem.AtomFromSmiles('*'))
    star2 = em.AddAtom(Chem.AtomFromSmiles('*'))
    em.RemoveBond(pm.connector_inds[0],pm.connector_inds[1])
    em.AddBond(pm.connector_inds[0],star1,Chem.BondType.SINGLE)
    em.AddBond(pm.connector_inds[1],star2,Chem.BondType.SINGLE)
    return em

def elim_retro(lp, elim_group):
    '''
    Reverse of eliminations. They generally will occur after addition of heat.
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    pm = lp.PeriodicMol()
    rxn_info = elim_rxns[elim_group]
    replace_patt = Chem.MolFromSmarts(rxn_info['replace_patt'])
    matches = pm.GetSubstructMatches(replace_patt)
    mols = []
    if len(matches) == 0:
        return []
    for L in range(1, len(matches)+1):
        for match_combo in itertools.combinations(matches,L):
            match_combo_flat = ru.flatten_ll(match_combo)
            if len(set(match_combo_flat)) != len(match_combo_flat): #no atoms should overlap
                pass
            else:
                em = Chem.EditableMol(pm.mol)
                new_mols = rxn_info['edit_fxn'](em,pm,match_combo,L)
                for new_mol in new_mols:
                    try:
                        Chem.SanitizeMol(new_mol)
                        mols.append(new_mol)
                    except:
                        pass
    return mols

def ring_close_retro(lp):
    '''
    Reverse of Ring-closing which will occur after addition of heat
    Source: Reynolds Class Notes, 8-25-20, p.2
    Test SMILES: '*c5ccc(Oc4ccc(n3c(=O)c2cc1c(=O)n(*)c(=O)c1cc2c3=O)cc4)cc5'
    '''
    if type(lp) == str or type(lp) == Chem.rdchem.Mol: #only for convenience. Pass in LinearPol object when possible
        lp = ru.LinearPol(lp)
    
    start_match = Chem.MolFromSmarts('[c,C;!R0;!R1](=O)[n,N;!R0;!R1]')
    end_match = Chem.MolFromSmarts('[#6R0](=O)([OH])[C,c][C,c][CR1](=O)[NR1]')
    pm = lp.PeriodicMol()
    if pm is None:
        return []
    pm.GetSSSR()
    
    mols = []
    if pm.mol.HasSubstructMatch(start_match) and not pm.mol.HasSubstructMatch(end_match): 
        lp_no_connect_inds = np.array([x for x in range(lp.mol.GetNumAtoms()) if x not in lp.star_inds])
        def lp_to_pm_ind(lp_ind):
            return int(np.argwhere(lp_no_connect_inds==lp_ind))
        ar_atom_idx = [a.GetIdx() for a in lp.mol.GetAromaticAtoms()]
        if len(ar_atom_idx) != 0: #only execute below for aromatic polymers. There for speed.
            ri = lp.mol.GetRingInfo()
            ar = ri.AtomRings()
            atom_aromaticity = {a:0 for a in ar_atom_idx}

            for ring in ar:
                if ring[0] in ar_atom_idx:
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
                        i_n_aromaticity = atom_aromaticity[ lp_to_pm_ind(i_n) ]
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

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉") #convert numbers in str to subscript
fwd_rxn_labels = {
    'frp_depolymerize': 'radical/ionic polymerization',
    'sg_depolymerize': 'step growth polymerization',
    'ox_depolymerize': 'oxidative polymerization',
    'ro_depolymerize': 'ring-opening polymerization',
    'ring_close_retro': '\u0394 and ring-closure',
    'hydrogenate_chain': 'hydrogenation',
}

def LabelReaction(rxn_string):
    if 'elim_retro' in rxn_string:
        elim_group = rxn_string.split('.')[1]
        return str('\u0394 and %s elimination' %elim_group).translate(SUB)
    if rxn_string in fwd_rxn_labels.keys():
        return fwd_rxn_labels[rxn_string]
    return 'reaction'

def drawRxn(p_mol,r_mols=None,dp_func_ls=None,extra_arg1=None,extra_arg2=None,imgSize=(6,4),title='', legend_adds=['']):
    '''
    Return the single-step polymerization, reverse of dp_func, of a polymer, p_mol. extra_args are for compatability with step_growth. Legend adds are list of strings that will be added below each legend.
    '''
    if type(p_mol) == str:
        p_mol = Chem.MolFromSmiles(p_mol)
    if r_mols is None:
        try:
            r_mols = dp_func_ls[0](p_mol)
        except:
            dp_func = sg_depolymerize
            r_mols = dp_func_ls(p_mol,extra_arg1,extra_arg2)
    if type(r_mols) != list:
        r_mols = [r_mols]
    #all_mols = ru.flatten_ll([[r_mols[i],p_mol] for i in range(len(r_mols))])
    all_mols = r_mols + [p_mol]
    rxn_labels = [LabelReaction(dp_func) for dp_func in dp_func_ls]
    all_legends = ['0'] + ['%s: After %s of %s' %(i+1,rxn_labels[i],i) for i in range(len(r_mols))]
    return ru.MolsToGridImage(all_mols,labels=all_legends,molsPerRow=2,ImgSize=imgSize,title=title)

class ReactionStep:
    def __init__(self, reactant, product, rxn_fn_hash, addl_rxn_info=''): 
        self.reactant_mol = reactant
        self.reactant_frags = Chem.GetMolFrags(self.reactant_mol, asMols=True)
        self.reactant_frag_smiles = [Chem.MolToSmiles(mol) for mol in self.reactant_frags]
        self.n_reactants = len( self.reactant_frag_smiles)
        self.product_mol = product
        self.product_smiles = Chem.MolToSmiles(self.product_mol).replace('*','[*]') #replace is there for historical reasons
        self.rxn_fn_hash = rxn_fn_hash
        self.rxn_fn = str(rxn_fn_hash).split(' ')[1]
        if addl_rxn_info != '':
            self.rxn_fn = self.rxn_fn + '.' + addl_rxn_info
        self.catalog = None
        self.synthetic_scores = None
        self.poly_syn_score = None #synthetic complexity of polymer. TODO: change this name
        try:
            self.fwd_rxn_label = fwd_rxn_labels[self.rxn_fn]
        except:
            self.fwd_rxn_label = 'Reaction Unknown'
    
    def SearchReactants(self,mol_set):
        if self.catalog is None:
            self.catalog = np.array([x in mol_set for x in self.reactant_frag_smiles])
        return self.catalog
    
    def DrawStep(self,size=(6, 4),title_add=''):
        title = 'Reaction'
        title += title_add
        drawRxn(self.product_mol,self.reactant_mol,self.rxn_fn,imgSize=size,title=title)
    
    def DrawCatalog(self,mol_set=None):
        if self.catalog is None:
            if mol_set is None:
                raise LookupError('No catalog or mol_set specified')
            else:
                _ = self.SearchReactants(mol_set)
        else:
            labels = []
            for ind,i in enumerate(self.catalog):
                mol_index = chr(65+ind)
                if i == True:
                    labels.append('%s: In eMolecules set' %mol_index)
                else:
                    labels.append('%s: Not in eMolecules set' %mol_index)
            if self.synthetic_scores is not None:
                for ind in range(self.n_reactants):
                    label = labels[ind] + '\nSCScore: {:.2f}'.format( self.synthetic_scores[ind] )
                    labels[ind] = label
            return ru.MolsToGridImage(self.reactant_frags,labels=labels,molsPerRow=min(2,self.n_reactants),ImgSize=(6, 1.5*self.n_reactants),title='Reactants')
    
    def DrawDetail(self,mol_set=None):
        a = self.DrawStep()
        b = self.DrawCatalog(mol_set)
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow( ru.get_img_from_fig(a) )
        axes[1].imshow( ru.get_img_from_fig(b) )
        #clean each axis
        for ax in axes:
            for s in ax.spines.keys():
                ax.spines[s].set_visible(False)    
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        fig.show()

    
    def SyntheticScore(self):
        if self.catalog is None: #synthetic score cannot be computed without first running SearchReactants
            raise LookupError('Catalog is equal to None')
        elif self.poly_syn_score is None:
            def helper(ind):
                if self.catalog[ind]:
                    return 1
                else:
                    return sc_model.get_score_from_smi(self.reactant_frag_smiles[ind])[1] #return second argument, first is smiles
            self.synthetic_scores = np.array( list(map(lambda x: helper(x), range(self.n_reactants))) )
            self.poly_syn_score = float(np.product(self.synthetic_scores))
        return self.poly_syn_score
    
    def SetRepresentation(self):
        '''
        Return a representation of the reactionStep which can be used to remove duplicates
        '''
        return (self.product_smiles, ' '.join(sorted(self.reactant_frag_smiles)), self.rxn_fn)


def search_polymer(lp,polymer_set):
    if type(lp) == str or type(lp) == Chem.rdchem.Mol:
        lp = ru.LinearPol(lp)
    if type(lp) == ru.LinearPol:
        try:
            pm1 = lp.PeriodicMol()
            if pm1 is not None:
                return Chem.MolToSmiles(pm1) in polymer_set
            pm2 = lp.multiply(2).PeriodicMol()
            if pm2 is not None:
                return Chem.MolToSmiles(pm2) in polymer_set
            pm3 = lp.multiply(3).PeriodicMol()
            return Chem.MolToSmiles(pm3) in polymer_set
        except:
            raise(ValueError)
    else:
        raise(TypeError, 'Invalid type for input lp')
    

class ReactionPath:
    def __init__(self, reaction_step_ls):
        self.reaction_step_ls = reaction_step_ls
        self.lp_syn_score = None
        self.can_smiles = None
        self.lp_exists = None
        self.lp = None
        self.doi = None
        #below is a patch job that will need to changed when multi-step reactions occur
        self.catalog = None
        self.synthetic_scores = None
        self.syn_class = None
        self.r_mols = None
        self.depolymerization_step = None
    
    def GetLP(self): #must make a separate function for this to avoid pickling issues
        self.lp = ru.LinearPol(self.reaction_step_ls[-1].product_mol) #is a polymer
        self.lp_smiles = self.lp.SMILES

    def SearchPolymer(self,pol_dict):
        if self.lp is None:
            self.GetLP()
        for n in range(4):
            if self.lp_exists:
                break
            pm = self.lp.multiply(n).PeriodicMol()
            if pm is not None:
                try:
                    self.can_smiles = pol_dict[ Chem.MolToSmiles(pm) ]
                    self.lp_exists = True
                except:
                    self.lp_exists = False
                    pass
    
    def DrawSteps(self,size=(6, 4)):
        if self.lp_syn_score is None:
            title_add = ''
        else:
            title_add = title_add='\nSynthetic Complexity: {:.2f}'.format( self.lp_syn_score )
        
        # if self.GetNSteps() == 1:
        #     return self.reaction_step_ls[0].DrawStep(size=size,title_add=title_add)
        #else:
        title = 'Reaction'
        title += title_add
        if self.r_mols is None:
            self.r_mols = [x.reactant_mol for x in reversed(self.reaction_step_ls)]
        dp_func_ls = list(reversed([x.rxn_fn for x in self.reaction_step_ls]))
        drawRxn(self.lp.mol,self.r_mols,dp_func_ls,imgSize=size,title=title)            

    def DrawCatalog(self):
        if self.GetNSteps() == 1:
            return self.reaction_step_ls[0].DrawCatalog()

    def SearchReactants(self,mol_set):
        if self.depolymerization_step is None:
            self.GetPolymerizationStep()
        if self.catalog is None:
            self.catalog = self.depolymerization_step.SearchReactants(mol_set)
    
    def SyntheticScore(self,solubility=True):
        '''
        If solubulity=True, the solubility of the polymer pre-cursor is used in scoring
        '''
        if self.catalog is None: #synthetic score cannot be computed without first running SearchReactants
           raise ValueError('First call SeachReactants with a mol_set')
        if self.lp_syn_score is None:
            self.synthetic_scores = self.depolymerization_step.SyntheticScore()            
            solubility_score = 1
            if solubility:
                if not ru.is_soluble(self.depolymerization_step.product_mol):
                    solubility_score = 2.5
            n_steps = self.GetNSteps()
            self.lp_syn_score = float(np.product(self.synthetic_scores))*solubility_score*(1.25)**(n_steps-1)
        return self.lp_syn_score

    def SyntheticClass(self):
        if self.syn_class is None and self.lp_exists is not None and self.catalog is not None:
            if self.lp_exists:
                self.syn_class = 1 
            elif np.max(self.synthetic_scores) == 1:
                self.syn_class = 2
            elif np.min(self.synthetic_scores) == 1:
                self.syn_class = 3
            else:
                self.syn_class = 4
        return self.syn_class
    
    def GetNSteps(self):
        return len(self.reaction_step_ls)

    def GetPolymerizationStep(self): 
        self.depolymerization_step = list(filter(lambda x: 'depolymerize' in x.rxn_fn, self.reaction_step_ls))[0]
        return self.depolymerization_step
    

    # def SearchDoi(self,doi_dict):
    #     if self.doi is None and self.lp_exists:
    #         self.doi = doi_dict[self.can_smiles]

def retro_depolymerize(mol,radion=True,sg=True,ox=True,ro=True,debug=False):
    '''
    Input a list of smiles and return the synthesis pathways
    '''
    ReactionStepList = []
    #do free-radical depolymerization
    if radion:
        monomer_ls = frp_depolymerize(mol)
        if monomer_ls is not None:
            for monomer in monomer_ls:
                rs = ReactionStep(monomer,mol,frp_depolymerize)
                ReactionStepList.append(rs)
    #do step-growth depolymerization
    if sg:
        linkages = sg_rxns.keys()
        for linkage_smiles in linkages:
            if '#' not in linkage_smiles: #if string does not have '#' in it then it's a SMILES 
                linkage_mol = Chem.MolFromSmiles(linkage_smiles)
            else:
                linkage_mol = Chem.MolFromSmarts(linkage_smiles)
            for rxn_info in sg_rxns[linkage_smiles]:
                if debug:
                    print('##### %s #####' %linkage_smiles)
                monomers = sg_depolymerize(mol,linkage_mol,rxn_info,debug=debug)
                if monomers != None:
                    for monomer in monomers:
                        rs = ReactionStep(monomer,mol,sg_depolymerize)
                        ReactionStepList.append(rs)
    #do oxidative depolymerization
    if ox:
        monomer_ls = ox_depolymerize(mol)
        if monomer_ls is not None:
            for monomer in monomer_ls:
                rs = ReactionStep(monomer,mol,ox_depolymerize)
                ReactionStepList.append(rs)
    #do ring-opening depolymerization
    if ro:
        linkages = ro_linkages.keys()
        for l in linkages:
            monomers = ro_depolymerize(mol,l)
            if monomers != None:
                for m in monomers:
                    rs = ReactionStep(m,mol,ro_depolymerize)
                    ReactionStepList.append(rs)
    keep_inds = ru.arg_unique_ordered([x.SetRepresentation() for x in ReactionStepList])
    return [ReactionStepList[i] for i in keep_inds] 

post_polymerization_rxns = [ring_close_retro, func_chain_retro, hydrogenate_chain, elim_retro] #each return type should be ***list of mols*** or ***empty*** list

def retrosynthesize(smiles_ls,n_core=1,radion=True,sg=True,ox=True,ro=True,chain_reactions=True,dimerize=False,debug=False):
    '''
    Input a list of smiles and return the synthesis pathways
    '''
    if type(smiles_ls) == 'str': #handle case of one string passed in
        smiles_ls = [smiles_ls]

    def helper(sm):
        mol = Chem.MolFromSmiles(sm)
        lp = ru.LinearPol(mol)
        sm_RxnPaths = [ReactionPath([])]
        if dimerize:
            lp2 = lp.multiply(2)
            Chem.GetSSSR(lp2.mol)
            sm_RxnPaths += [ReactionPath([])]
        if debug:
            print('#######')
            print(sm)
        if chain_reactions:
            for rxn in post_polymerization_rxns:
                if debug:
                    print(str(rxn))
                inner_RxnPaths = []
                for RxnPath in sm_RxnPaths:
                    i = 1
                    if RxnPath.reaction_step_ls == []:
                        if i == 1:
                            curr_mol = lp.mol #the mol to depolymerize
                            i += 1
                        elif i == 2 and dimerize:
                            curr_mol = lp2.mol
                    else:
                        curr_mol = RxnPath.reaction_step_ls[-1].reactant_mol
                    if 'elim_retro' in str(rxn):
                        RxnSteps = []
                        for elim_group in elim_rxns.keys():
                           RxnSteps.extend( [ReactionStep(product=curr_mol,reactant=x,rxn_fn_hash=rxn,addl_rxn_info=elim_group) for x in rxn(curr_mol,elim_group)] ) 
                    elif 'func_chain' in str(rxn):
                        RxnSteps = []
                        for k in func_chain_rxns.keys():
                           RxnSteps.extend( [ReactionStep(product=curr_mol,reactant=x,rxn_fn_hash=rxn) for x in rxn(curr_mol,rxn=k)] )                        
                    else:
                        RxnSteps = [ReactionStep(product=curr_mol,reactant=x,rxn_fn_hash=rxn) for x in rxn(curr_mol)]
                    #print('RxnSteps len:', len(RxnSteps))
                    keep_inds = ru.arg_unique_ordered([x.SetRepresentation() for x in RxnSteps])
                    #print('Unique RxnSteps len:',len(keep_inds))
                    unique_RxnSteps = [RxnSteps[i] for i in keep_inds]
                    inner_RxnPaths.extend( [ ReactionPath(RxnPath.reaction_step_ls + [x]) for x in unique_RxnSteps] )
                sm_RxnPaths = sm_RxnPaths + inner_RxnPaths #update sm_RxnPaths
                if debug:
                    print('inner_RxnPaths len:', len(inner_RxnPaths))
            if debug:
                print('sm_RxnPaths len:', len(sm_RxnPaths))
        
        final_rxn_paths = []
        for RxnPath in sm_RxnPaths:
            i = 1
            if RxnPath.reaction_step_ls == []:
                if i == 1:
                    curr_mol = lp.mol #the mol to depolymerize
                    i += 1
                elif i == 2 and dimerize:
                    curr_mol = lp2.mol
            else:
                curr_mol = RxnPath.reaction_step_ls[-1].reactant_mol #the mol to depolymerize
            try:
                DepolymerizationSteps = retro_depolymerize(curr_mol,radion=radion,sg=sg,ox=ox,ro=ro)
                final_rxn_paths.extend( [ ReactionPath(RxnPath.reaction_step_ls + [x]) for x in DepolymerizationSteps] )
            except:
                print(sm)
                raise ValueError
        return final_rxn_paths


    if n_core == 1:
        all_RxnPaths = []
        for sm in smiles_ls:
            RxnPaths = helper(sm)
            all_RxnPaths.extend( RxnPaths )
    else:
        all_RxnPaths_ll = Parallel(n_jobs=n_core)(delayed(helper)(sm) for sm in smiles_ls)
        all_RxnPaths = ru.flatten_ll(all_RxnPaths_ll)
    return all_RxnPaths