'''
Sandro Tacchella
January 2, 2018   : iniate
run this script after run_make_SFH.py in order to
combine all (numpy) files to one (hdf5) file.

'''

# import modules

import hickle
import glob
import tqdm


# define paths

path_results = '/n/conroyfs1/stacchella/halo7d_co/results/run_param/posterior_draws/'

result_file_list = glob.glob(path_results + '*_output.pkl')


# loop over all result files

output_condensed = {}

for ii_f in tqdm(range(len(result_file_list))):
    output = hickle.load(result_file_list[ii_f])
    output_condensed[output['ID']] = output


# save output
print 'writing summary file...'

f = open(path_results + "summary_param_run.pkl", "w")
hickle.dump(output_condensed, f)
f.close()

