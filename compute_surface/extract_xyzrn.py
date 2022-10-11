from Bio.PDB import *
import os
import numpy as np
atom_radii = {}
atom_radii["N"] = "1.540000"
atom_radii["O"] = "1.400000"
atom_radii["C"] = "1.740000"
atom_radii["H"] = "1.200000"
atom_radii["S"] = "1.800000"
atom_radii["P"] = "1.800000"
atom_radii["Z"] = "1.39"
atom_radii["X"] = "0.770000"


"""
xyzrn.py: Read a pdb file and output it is in xyzrn for use in MSMS
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

def extract_xyzrn(pair,dir_opts={}):
    """
        pdbfilename: input pdb filename
        xyzrnfilename: output in xyzrn format.
    """
    pdbfilename = os.path.join(dir_opts['protonated_pdb_dir'],pair+'.pdb')
    xyzrnfilename = os.path.join(dir_opts['xyzrn_dir'],pair+'.xyzrn')
    if not os.path.exists(dir_opts['xyzrn_dir']):
        os.makedirs(dir_opts['xyzrn_dir'])

    parser = PDBParser()
    struct = parser.get_structure(pdbfilename, pdbfilename)

    out_list=[]
    coords = []
    for atom in struct.get_atoms():
        coords.append(atom.get_coord())
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        chain = residue.get_parent().get_id()
        atomtype = name[0]
        coords=None
        if atomtype in atom_radii:
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, atomtype
            )
        if coords is not None:
            out_list.append((coords + " " + atom_radii[atomtype] + " 1 " + full_id + "\n"))
    outfile = open(xyzrnfilename, "w")
    outfile.writelines(out_list)
    outfile.close()
