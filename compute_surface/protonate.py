"""
protonate.py: Wrapper method for the reduce program: protonate (i.e., add hydrogens) a pdb using reduce 
                and save to an output file.
Pengpai Li - Qipan village 2021/01/04
Released under an Apache License 2.0
"""

from subprocess import Popen, PIPE
import os
import fcntl
from default_config.bin_path import bin_path

def protonate(pdb_id, dir_opts={}):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    os.environ['REDUCE_BIN']='/home/aoli/.local/bin/reduce'
    # Remove protons first, in case the structure is already protonated
    # print('Protonating PDB {}/{}_protonated.pdb'.format(dir_opts['protonated_pdb_dir'], pdb_id))
    in_pdb_file = os.path.join(dir_opts['chain_pdb_dir'], pdb_id+'.pdb')
    middle_file = os.path.join(dir_opts['protonated_pdb_dir'], pdb_id+'.pdb')
    out_pdb_file = os.path.join(dir_opts['protonated_pdb_dir'], pdb_id+'.pdb')

    if not os.path.exists(dir_opts['protonated_pdb_dir']):
        os.makedirs(dir_opts['protonated_pdb_dir'])

    reduce_bin = bin_path['REDUCE']

    args = [reduce_bin, "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    outfile = open(middle_file, "w")
    fcntl.flock(outfile,fcntl.LOCK_EX)
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()


    #Now add them again.
    args = [reduce_bin, "-HIS", middle_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    fcntl.flock(outfile,fcntl.LOCK_EX)
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()

    #return out_pdb_file

if __name__ == '__main__':
    import sys
    pdb_id = '1xok'
    protonate(pdb_id)