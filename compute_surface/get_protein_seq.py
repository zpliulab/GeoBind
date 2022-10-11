import os
from Bio.PDB import *
from Bio.SeqUtils import seq1

def get_seq(pdb_id, protein_chains, dir_opts={}):

    chain_ids = protein_chains.split('_')
    # modified from protein_multimer designed with protein_chains in case of number of protein chains more than 2
    all_seq={}
    all_index2resid={}
    for chain_id in chain_ids:
        pdb_name = pdb_id + chain_id + '.pdb'
        infilename = os.path.join(dir_opts['raw_pdb_dir'], pdb_name)

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(infilename, infilename)
        model = Selection.unfold_entities(struct, "M")[0]
        # Select residues to extract and build new structure

        chain = model.child_dict[chain_id]
        res_type = ''
        index2resid = {}
        for index,res in enumerate(chain.child_list):
            res_type+= seq1(res.resname)
            index2resid[index] = res.id[1]
        all_seq[chain_id] = res_type
        all_index2resid[chain_id] = index2resid
    return all_seq, all_index2resid