'''
Sandro Tacchella
June 24, 2019   : iniate
run this script to combine all .pkl files (run on interactive node)
python summarize_draw_cluster.py
'''

# import modules

import hickle
import glob
from tqdm import tqdm


# define paths

path_results = '/n/conroyfs1/stacchella/halo7d_co/results/nonparam_limited_new/posterior_draws/'
name_output_file = 'summary_nonparam_limited_run.pkl'

result_file_list = glob.glob(path_results + '*_output.pkl')


# loop over all result files

output_condensed = {}

for ii_f in tqdm(range(len(result_file_list))):
    output = hickle.load(result_file_list[ii_f])
    output_condensed[output['ID']] = output


# save output
print 'writing summary file...'

f = open(path_results + name_output_file, "w")
hickle.dump(output_condensed, f)
f.close()

