def dir_opts(base_dir = './Dataset/DNA/'):
    dir_opts = {}

    dir_opts['base_dir'] = base_dir

    #pdb files and features
    dir_opts['raw_pdb_dir'] = 'Dataset/PDBs/receptor'
    dir_opts['ligand_dir'] = 'Dataset/PDBs/ligand'
    dir_opts['hhm_dir'] = 'Dataset/hmm'

    dir_opts['fasta_dir'] = dir_opts['base_dir'] + 'fasta/'
    dir_opts['dssp'] = dir_opts['base_dir'] + 'dssp/' #used for DRNAPred
    #data_processing

    dir_opts['protonated_pdb_dir'] = dir_opts['base_dir'] + 'protonated_pdb/'
    dir_opts['chain_pdb_dir'] = dir_opts['base_dir'] + 'chain_pdb/'
    dir_opts['xyzrn_dir'] = dir_opts['base_dir'] + 'xyzrn/'
    dir_opts['msms_dir'] = dir_opts['base_dir'] + 'msms/'
    dir_opts['data_label'] = dir_opts['base_dir'] + 'data_label/'
    # dir_opts['data_label'] = '/home/aoli/Documents/GeoBind/Dataset/DNA/data_label'

    dir_opts['save_dir'] = dir_opts['base_dir'] + 'save/'

    #optional dir
    dir_opts['mesh_dir'] = dir_opts['base_dir'] + './dataset/mesh'
    dir_opts['mesh_fix_dir'] = dir_opts['base_dir'] + './dataset/mesh_fix'
    dir_opts['ply_dir'] = dir_opts['base_dir'] + './dataset/ply'
    return dir_opts

#data_label3 all nuv is not normallized
#data_label4 all nuv is not normallized we normalized the Pij in generating nuv and in geometric conv