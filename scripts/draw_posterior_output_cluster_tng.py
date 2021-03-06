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
import halo7d_param_file_tng as param_file


# pars args

parser = argparse.ArgumentParser()
parser.add_argument("--number_of_bins", type=int, help="number of cores")
parser.add_argument("--idx_file_key", type=int, help="iteration variable")
parser.add_argument("--path_results", type=str, help="path results")
parser.add_argument("--ncalc", type=int, help="number of samples to draw from posterior")
parser.add_argument('--non_param_sfh', action="store_true",
                    help="If set, fit non-parametric star-formation history model.")
parser.add_argument('--n_bins_sfh', type=int, default=8,
                    help="Number of bins for SFH (non parametric).")
parser.add_argument('--dirichlet_sfh', action="store_true",
                    help="If set, fit non-parametric star-formation history model with dirichlet prior.")
parser.add_argument('--add_jitter', action="store_true",
                    help="If set, jitter noise.")
parser.add_argument('--fit_continuum', action="store_true",
                    help="If set, fit continuum.")
parser.add_argument('--add_neb', action="store_true",
                    help="If set, add nebular emission to the model.")
parser.add_argument('--add_duste', action="store_true",
                    help="If set, add dust emission to the model.")
parser.add_argument('--add_agn', action="store_true",
                    help="If set, add agn emission to the model.")
parser.add_argument('--switch_off_spec', action="store_true",
                    help="If set, remove spectrum from obs.")
parser.add_argument('--switch_off_phot', action="store_true",
                    help="If set, remove photometry from obs.")
parser.add_argument('--fixed_dust', action="store_true",
                    help="If set, fix dust to Calzetti and add tight Z prior.")
parser.add_argument('--use_eline_prior', action="store_true",
                    help="If set, use EL prior from cloudy.")
args = parser.parse_args()


run_params = {'number_of_bins': args.number_of_bins,  # this gives number of cores we run on
              'idx_file_key': args.idx_file_key,  # iteration variable
              }

path_res = path_wdir + 'results/' + args.path_results
ncalc = args.ncalc
mock_redshift = 0.7


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


def investigate(file_input, ncalc, non_param, n_bins_sfh, dirichlet_sfh=False, add_duste=False, add_neb=False, add_jitter=False, use_eline_prior=False,
                add_agn=False, fit_continuum=False, switch_off_phot=False, switch_off_spec=False, fixed_dust=False):
    print file_input
    # read results
    res, obs, mod, sps = read_results(file_input)
    mod = param_file.build_model(zred=mock_redshift, non_param_sfh=non_param, n_bins_sfh=n_bins_sfh, dirichlet_sfh=dirichlet_sfh, add_duste=add_duste, add_neb=add_neb, add_jitter=add_jitter, add_agn=add_agn, fit_continuum=fit_continuum, switch_off_phot=switch_off_phot, switch_off_spec=switch_off_spec, fixed_dust=fixed_dust, use_eline_prior=use_eline_prior)
    output = {}
    nsample = res['chain'].shape[0]
    sample_idx = np.random.choice(np.arange(nsample), size=ncalc, p=res['weights'], replace=False)
    output = toolbox_prospector.build_output(res, mod, sps, obs, sample_idx, ncalc=ncalc, non_param=non_param, shorten_spec=False, elines=None, abslines=None)
    return(obs, output)


def get_file_ids(number_of_bins, idx_file_key=1.0, **kwargs):
    idx_all_files = range(len(result_file_list))
    idx_bins_all_files = np.array_split(idx_all_files, number_of_bins)
    print(idx_bins_all_files[int(float(idx_file_key))-1])
    return(idx_bins_all_files[int(float(idx_file_key))-1])  # -1 since slurm counts from 1 (and not from 0)


def compute_chi2(obs, output, switch_off_phot, switch_off_spec):
    if switch_off_spec:
        reduced_chi_square_spec = -99.0
    else:
        spec_tot = output['obs']['spec_wEL']['q50']
        spec_jitter = output['thetas']['spec_jitter']['q50']
        spec_noise = obs['unc']*spec_jitter
        reduced_chi_square_spec = 1.0/np.sum(obs['mask'])*np.sum((obs['spectrum'][obs['mask']]-spec_tot[obs['mask']])**2/spec_noise[obs['mask']]**2)
    if switch_off_phot:
        reduced_chi_square_phot = -99.0
    else:
        mags = output['obs']['mags']['q50']
        reduced_chi_square_phot = 1.0/np.sum(obs['phot_mask'])*np.sum((obs['maggies'][obs['phot_mask']]-mags[obs['phot_mask']])**2/obs['maggies_unc'][obs['phot_mask']]**2)
    return(reduced_chi_square_phot, reduced_chi_square_spec)


# get files and iterate over them

result_file_list = glob.glob(path_res + '*.h5')

idx_file_considered = get_file_ids(**run_params)

print idx_file_considered


for ii in range(len(idx_file_considered)):
    print result_file_list[idx_file_considered[ii]]
    obs, output = investigate(result_file_list[idx_file_considered[ii]].split('/')[-1], ncalc=ncalc, non_param=args.non_param_sfh, n_bins_sfh=args.n_bins_sfh, dirichlet_sfh=args.dirichlet_sfh,
                              add_neb=args.add_neb, add_duste=args.add_duste, add_jitter=args.add_jitter, add_agn=args.add_agn, fit_continuum=args.fit_continuum, use_eline_prior=args.use_eline_prior,
                              switch_off_phot=args.switch_off_phot, switch_off_spec=args.switch_off_spec, fixed_dust=args.fixed_dust)
    output['file_name'] = result_file_list[idx_file_considered[ii]].split('/')[-1]
    output['index_galaxy'] = obs['index_galaxy']
    output['tabular_time'] = obs['tabular_time']
    output['tabular_sfr'] = obs['tabular_sfr']
    output['tabular_mtot'] = obs['tabular_mtot']
    output['mock_snr_phot'] = obs['mock_snr_phot']
    output['mock_snr_spec'] = obs['mock_snr_spec']
    output['true_spectrum'] = obs['true_spectrum']
    output['wavelength'] = obs['wavelength']
    output['true_maggies'] = obs['true_maggies']
    output['mock_params'] = obs['mock_params']
    output['maggies'] = obs['maggies']
    output['maggies_unc'] = obs['maggies_unc']
    output['phot_wave'] = obs['phot_wave']
    output['spectrum'] = obs['spectrum']
    output['unc'] = obs['unc']
    # chi2_phot, chi2_spec = compute_chi2(obs, output, switch_off_phot=args.switch_off_phot, switch_off_spec=args.switch_off_spec)
    # output['chi2_phot'] = chi2_phot
    # output['chi2_spec'] = chi2_spec
    file_name = path_res + "posterior_draws/" + str(obs['index_galaxy']) + "_output.pkl"
    if os.path.exists(file_name):
        os.remove(file_name)
    f = open(file_name, "w")
    hickle.dump(output, f)
    f.close()
