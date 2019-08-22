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
import param_file_sfh as param_file
#import param_file_sfh_small as param_file
#import param_file as param_file

# pars args

parser = argparse.ArgumentParser()
parser.add_argument("--number_of_bins", type=int, help="number of cores")
parser.add_argument("--idx_file_key", type=int, help="iteration variable")
parser.add_argument("--path_results", type=str, help="path results")
parser.add_argument("--ncalc", type=int, help="number of samples to draw from posterior")
parser.add_argument('--prior_file', type=str, default=path_wdir+"data/halo7d_with_phot.fits",
                    help="Name of file containing priors.")
parser.add_argument('--non_param_sfh', action="store_true",
                    help="If set, fit non-parametric star-formation history model.")
parser.add_argument('--add_jitter', action="store_true",
                    help="If set, jitter noise.")
parser.add_argument('--fit_continuum', action="store_true",
                    help="If set, fit continuum.")
parser.add_argument('--add_duste', action="store_true",
                    help="If set, add dust emission to the model.")
parser.add_argument('--add_agn', action="store_true",
                    help="If set, add agn emission to the model.")
parser.add_argument('--switch_off_spec', action="store_true",
                    help="If set, remove spectrum from obs.")
parser.add_argument('--switch_off_phot', action="store_true",
                    help="If set, remove photometry from obs.")
parser.add_argument('--remove_mips24', action="store_true",
                    help="If set, removes MIPS 24um flux.")
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


def investigate(file_input, ncalc, non_param, add_duste=False, add_neb=False, add_jitter=False, add_agn=False, fit_continuum=False, remove_mips24=False, switch_off_phot=False, switch_off_spec=False, prior_file=False):
    print file_input
    # read results
    res, obs, mod, sps = read_results(file_input)
    mod = param_file.build_model(objid=obs['cat_row']+1, non_param_sfh=non_param, add_duste=add_duste, add_neb=add_neb, add_jitter=add_jitter, add_agn=add_agn, fit_continuum=fit_continuum, remove_mips24=remove_mips24, switch_off_phot=switch_off_phot, switch_off_spec=switch_off_spec, prior_file=prior_file)
    output = {}
    nsample = res['chain'].shape[0]
    sample_idx = np.random.choice(np.arange(nsample), size=ncalc, p=res['weights'], replace=False)
    output = toolbox_prospector.build_output(res, mod, sps, obs, sample_idx, ncalc=ncalc, non_param=non_param, shorten_spec=False, elines=None, abslines=None)
    return(obs['id_halo7d'], obs['id_3dhst'], obs['RA'], obs['DEC'], obs['SN_calc'], output)


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
    ID, id_3dhst, ra, dec, sn, output = investigate(result_file_list[idx_file_considered[ii]].split('/')[-1], ncalc=ncalc, non_param=args.non_param_sfh, add_duste=args.add_duste, add_jitter=args.add_jitter, add_agn=args.add_agn, fit_continuum=args.fit_continuum, remove_mips24=args.remove_mips24, switch_off_phot=args.switch_off_phot, switch_off_spec=args.switch_off_spec, prior_file=args.prior_file)
    output['file_name'] = result_file_list[idx_file_considered[ii]].split('/')[-1]
    output['ID'] = ID
    output['ra'] = ra
    output['dec'] = dec
    output['SN'] = sn
    file_name = path_res + "posterior_draws/" + ID + "_output.pkl"
    if os.path.exists(file_name):
        os.remove(file_name)
    f = open(file_name, "w")
    hickle.dump(output, f)
    f.close()


