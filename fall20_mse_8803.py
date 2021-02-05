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
from retro_prepolymerization import *
from retro_postpolymerization import *
import retro_prepolymerization as pre

### set up scscore ###
from scscore import standalone_model_numpy as sc
sc_model = sc.SCScorer()
sc_model.restore('/data/rgur/retrosynthesis/scscore/models/full_reaxys_model_1024uint8/model.ckpt-10654.as_numpy.json.gz')

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

def is_symmetric(mol,group):
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
    def __init__(self, reaction_step_ls, is_dimer=False):
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
        self.is_dimer = is_dimer
    
    def GetLP(self): #must make a separate function for this to avoid pickling issues
        self.lp = ru.LinearPol(self.reaction_step_ls[0].product_mol) #is a polymer
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
        if self.lp is None:
            self.GetLP()
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

def retro_depolymerize(lp,pm=None,radion=True,sg=True,ox=True,ro=True,debug=False):
    '''
    Input a list of smiles and return the synthesis pathways
    '''
    if pm is None:
        pm = lp.PeriodicMol()
    ReactionStepList = []
    #do free-radical depolymerization
    if radion:
        monomer_ls = frp_depolymerize(lp)
        if monomer_ls is not None:
            for monomer in monomer_ls:
                rs = ReactionStep(monomer,lp.mol,frp_depolymerize)
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
                monomers = sg_depolymerize(lp,linkage_mol,rxn_info,pm=pm,debug=debug)
                if monomers != None:
                    for monomer in monomers:
                        rs = ReactionStep(monomer,lp.mol,sg_depolymerize)
                        ReactionStepList.append(rs)
    #do oxidative depolymerization
    if ox:
        monomer_ls = ox_depolymerize(lp)
        if monomer_ls is not None:
            for monomer in monomer_ls:
                rs = ReactionStep(monomer,lp.mol,ox_depolymerize)
                ReactionStepList.append(rs)
    #do ring-opening depolymerization
    if ro:
        linkages = ro_linkages.keys()
        for l in linkages:
            monomers = pre.ro_depolymerize(lp,l,pm=pm)
            if monomers != None:
                for m in monomers:
                    rs = ReactionStep(m,lp.mol,ro_depolymerize)
                    ReactionStepList.append(rs)
    keep_inds = ru.arg_unique_ordered([x.SetRepresentation() for x in ReactionStepList])
    return [ReactionStepList[i] for i in keep_inds] 

def retrosynthesize(smiles_ls,n_core=1,radion=True,sg=True,ox=True,ro=True,chain_reactions=True,dimerize=False, hydrogenate_chain=True,debug=False,greedy=True):
    '''
    Input a list of smiles and return the synthesis pathways
    '''
    if type(smiles_ls) == 'str': #handle case of single string passed in
        smiles_ls = [smiles_ls]
    
    #determine from user input which chain reactions to perform
    if chain_reactions:
        exclude_chain_rxns = [] #skip these reactions
        if not hydrogenate_chain:
            exclude_chain_rxns.append('hydrogenate_chain')
        if debug:
            print('Exlude rxns:', exclude_chain_rxns)
        keep_chain_rxns = [rxn for rxn in post_polymerization_rxns if not ru.checkSubstrings(exclude_chain_rxns,str(rxn))]
    
    def helper(sm):
        mol = Chem.MolFromSmiles(sm)
        lp = ru.LinearPol(mol)
        pm = lp.PeriodicMol()
        sm_RxnPaths = [ReactionPath([])]
        if dimerize:
            lp2 = lp.multiply(2)
            Chem.GetSSSR(lp2.mol)
            pm2 = lp2.PeriodicMol()
            sm_RxnPaths += [ReactionPath([],is_dimer=True)]
        if debug:
            print('#######')
            print(sm)
        if chain_reactions:
            for rxn in keep_chain_rxns:
                if debug:
                    print(str(rxn))
                inner_RxnPaths = [] #the list where RxnPaths generated after rxn will be added
                for RxnPath in sm_RxnPaths:
                    #select the mol to react
                    if RxnPath.reaction_step_ls == []:
                        if RxnPath.is_dimer:
                            curr_lp = lp2 #the mol to depolymerize
                            curr_pm = pm2
                        else:
                            curr_lp = lp
                            curr_pm = pm
                    else:
                        curr_mol = RxnPath.reaction_step_ls[-1].reactant_mol
                        curr_lp = ru.LinearPol(curr_mol)
                        curr_pm = curr_lp.PeriodicMol()
                    reaction_string = str(rxn)
                    if 'elim_retro' in reaction_string:
                        if not ru.is_soluble(curr_lp): #elimination reactions only need to occur for polymers that are not soluble 
                            RxnSteps = []
                            for elim_group in elim_rxns.keys():
                                if debug:
                                    print('elim_group:',elim_group)
                                try:
                                    RxnSteps.extend( [ReactionStep(product=curr_lp.mol,reactant=x,rxn_fn_hash=rxn,addl_rxn_info=elim_group) for x in rxn(curr_lp,elim_group,pm=curr_pm)] ) 
                                except:
                                    if greedy: #greedy = ignore and skip errors 
                                        pass
                                    else:
                                        return curr_lp.mol
                    elif 'func_chain' in reaction_string:
                        RxnSteps = []
                        for k in func_chain_rxns.keys():
                           RxnSteps.extend( [ReactionStep(product=curr_lp.mol,reactant=x,rxn_fn_hash=rxn) for x in rxn(curr_lp,rxn=k)] )                        
                    elif 'ring_close' in reaction_string:
                        RxnSteps = [ReactionStep(product=curr_lp.mol,reactant=x,rxn_fn_hash=rxn) for x in ring_close_retro(curr_lp,pm=curr_pm)]
                    elif 'hydro' in reaction_string:
                        RxnSteps = [ReactionStep(product=curr_lp.mol,reactant=x,rxn_fn_hash=rxn) for x in hydrogenate_chain(curr_lp)]

                    #filter unique steps
                    keep_inds = ru.arg_unique_ordered([x.SetRepresentation() for x in RxnSteps])
                    unique_RxnSteps = [RxnSteps[i] for i in keep_inds]

                    inner_RxnPaths.extend( [ ReactionPath(RxnPath.reaction_step_ls + [x]) for x in unique_RxnSteps] )
                
                sm_RxnPaths = sm_RxnPaths + inner_RxnPaths #update sm_RxnPaths
                
                if debug:
                    print('inner_RxnPaths len:', len(inner_RxnPaths))
            
            if debug:
                print('sm_RxnPaths len:', len(sm_RxnPaths))
        
        final_rxn_paths = []
        for RxnPath in sm_RxnPaths:
            #select the mol to depolymerize
            if RxnPath.reaction_step_ls == []:
                if RxnPath.is_dimer:
                    curr_lp= lp2
                else:
                    curr_lp = lp
            else:
                curr_mol = RxnPath.reaction_step_ls[-1].reactant_mol
                curr_lp = ru.LinearPol(curr_mol)
            curr_pm = curr_lp.PeriodicMol()

            try:
                DepolymerizationSteps = retro_depolymerize(curr_lp,curr_pm,radion=radion,sg=sg,ox=ox,ro=ro)
                final_rxn_paths.extend( [ ReactionPath(RxnPath.reaction_step_ls + [x]) for x in DepolymerizationSteps] )
            except:
                print(sm)
                if greedy:
                    pass
                else:
                    raise ValueError
        return final_rxn_paths


    if n_core == 1:
        all_RxnPaths = []
        for sm in smiles_ls:
            RxnPaths = helper(sm)
            if type(RxnPaths) == Chem.rdchem.Mol: #this line to help with debugging code
                return RxnPaths
            else:
                all_RxnPaths.extend( RxnPaths )
    else:
        all_RxnPaths_ll = Parallel(n_jobs=n_core)(delayed(helper)(sm) for sm in smiles_ls)
        all_RxnPaths = ru.flatten_ll(all_RxnPaths_ll) #flatten list of lists of RxnPaths
    return all_RxnPaths