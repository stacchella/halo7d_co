'''
Sandro Tacchella
June 24, 2019 : iniate
=> sbatch --array=1-XX submission_script_make_output.sh, with XX given by number_of_bins
'''


# import modules

import os
import glob
import sys
import numpy as np
from prospect.io import read_results as reader
from sedpy.observate import load_filters
import hickle
import argparse

import toolbox_prospector


# define paths

path_wdir = os.environ['WDIR_halo7d']

filter_folder = path_wdir + '/data/filters/'


# load parameter file

sys.path.append(path_wdir + 'runs/')
import param_file as param_file


# pars args

parser = argparse.ArgumentParser()
parser.add_argument("--number_of_bins", type=int, help="number of cores")
parser.add_argument("--idx_file_key", type=int, help="iteration variable")
parser.add_argument("--path_results", type=str, help="path results")
parser.add_argument("--ncalc", type=int, help="number of samples to draw from posterior")
args = parser.parse_args()


run_params = {'number_of_bins': args.number_of_bins,  # this gives number of cores we run on
              'idx_file_key': args.idx_file_key,  # iteration variable
              }


path_res = args.path_results
ncalc = args.ncalc


# define functions

def read_results(filename):
    res, obs, mod = reader.results_from(path_res + filename)
    # update data table
    res['run_params']['data_table'] = path_wdir + 'data/halo7d_with_phot.fits'
    mod = reader.get_model(res)
    # update filters
    filternames = [str(ii) for ii in obs['filters']]
    obs['filters'] = load_filters(filternames, directory=filter_folder)
    # load sps
    sps = reader.get_sps(res)
    return(res, obs, mod, sps)


def investigate(file_input, ncalc, non_param, add_duste=False, add_neb=False, add_jitter=False, add_agn=False, fit_continuum=False, remove_mips24=False):
    print file_input
    # read results
    res, obs, mod, sps = read_results(file_input)
    mod = param_file.build_model(objid=obs['cat_row']+1, non_param_sfh=non_param, add_duste=add_duste, add_neb=add_neb, add_jitter=add_jitter, add_agn=add_agn, fit_continuum=fit_continuum, remove_mips24=remove_mips24)
    output = {}
    nsample = res['chain'].shape[0]
    sample_idx = np.random.choice(np.arange(nsample), size=ncalc, p=res['weights'], replace=False)
    output = toolbox_prospector.build_output(res, mod, sps, obs, sample_idx, ncalc=ncalc, non_param=non_param, shorten_spec=False, elines=None, abslines=None)
    return(obs['id_halo7d'], output)


def get_file_ids(number_of_bins, idx_file_key=1.0, **kwargs):
    idx_all_files = range(len(result_file_list))
    idx_bins_all_files = np.array_split(idx_all_files, number_of_bins)
    print idx_bins_all_files[int(float(idx_file_key))-1]
    return(idx_bins_all_files[int(float(idx_file_key))-1])  # -1 since slurm counts from 1 (and not from 0)


# get files and iterate over them

result_file_list = glob.glob(path_res + '*.h5')

idx_file_considered = get_file_ids(**run_params)

print idx_file_considered

for ii in range(len(idx_file_considered)):
    print result_file_list[idx_file_considered[ii]]
    ID, output = investigate(result_file_list[idx_file_considered[ii]].split('/')[-1], ncalc=ncalc, non_param=False, add_duste=True, add_jitter=True, add_agn=True, fit_continuum=True)
    output['file_name'] = result_file_list[idx_file_considered[ii]].split('/')[-1]
    output['ID'] = ID
    file_name = path_res + "posterior_draws/" + ID + "_output.pkl"
    if os.path.exists(file_name):
        os.remove(file_name)
    f = open(file_name, "w")
    hickle.dump(output, f)
    f.close()


