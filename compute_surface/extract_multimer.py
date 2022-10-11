import os
from Bio.PDB import *
import numpy as np
Amino_acid_type = [
"ILE",
"VAL",
"LEU",
"PHE",
"CYS",
"MET",
"ALA",
"GLY",
"THR",
"SER",
"TRP",
"TYR",
"PRO",
"HIS",
"GLU",
"GLN",
"ASP",
"ASN",
"LYS",
"ARG",
]

def rotate_mat(axis, radian):
    rot_matrix=np.linalg.expm(np.cross(np.eye(3), axis / np.linalg.norm(axis) * radian))
    return rot_matrix
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1"
# Exclude disordered atoms.
def extractPDB(pair, aug=False, dir_opts={}):

    pdb_id, protein_chains, Nucleic_chains = pair.split(':')
    protein_chains = protein_chains.split('_')
    # chain_id = pdb_id_chain_id[1:]
    # infilename = os.path.join(dir_opts['raw_pdb_dir'],pdb_id+'.cif')
    if not os.path.exists(dir_opts['chain_pdb_dir']):
        os.mkdir(dir_opts['chain_pdb_dir'])

    outfilename = os.path.join(dir_opts['chain_pdb_dir'], pair+'.pdb')

    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    if aug:
        axis_x, axis_y, axis_z=[1, 0, 0], [0, 1, 0], [0, 0, 1]
        yaw=0.7854
        rot_matrix=rotate_mat(axis_z, yaw)
    for chain_id in protein_chains:

        infilename=os.path.join(dir_opts['raw_pdb_dir'], pdb_id + chain_id+'.pdb')
        parser=PDBParser(QUIET=True)
        struct=parser.get_structure(infilename, infilename)
        model=Selection.unfold_entities(struct, "M")[0]
        chain=model.child_dict[chain_id]
        structBuild.init_chain(chain.get_id()[-1])
        for residue in chain:
            if aug:
                for atom in residue.child_list:
                    # coord = atom.coord + np.random.normal(0, 1, 3)
                    coord = np.dot(atom.coord, rot_matrix)
                    atom.set_coord(coord)
            if residue.get_resname().upper() in Amino_acid_type:
                outputStruct[0][chain.get_id()[-1]].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pid = open(outfilename,'w')
    pdbio.save(pid, select=NotDisordered())
    pid.close()

