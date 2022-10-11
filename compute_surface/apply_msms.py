from default_config.bin_path import bin_path
from compute_surface.read_msms import read_msms
from subprocess import Popen, PIPE
import os
import random
def computeMSMS(pair,dir_opts, probe_radius):
    xyzrn = os.path.join(dir_opts['xyzrn_dir'], pair+'.xyzrn')
    msms_file_base = os.path.join(dir_opts['msms_dir'],  pair + str(random.randint(1,10000000)))
    if not os.path.exists(dir_opts['msms_dir']):
        os.makedirs(dir_opts['msms_dir'])

    msms_bin = bin_path['MSMS']
    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    args = [msms_bin, "-density", "3", "-hdensity", "3", "-probe",\
                    str(probe_radius), "-if",xyzrn,"-of",msms_file_base, "-af", msms_file_base]

    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    vertices, faces, normalv, res_id = read_msms(msms_file_base)
    return vertices, faces, normalv, res_id