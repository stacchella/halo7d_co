#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p conroy,itc_cluster,hernquist,shared
### memory per cpu, in MB
#SBATCH --mem-per-cpu=8000
### constraints
#SBATCH --constraint=intel
### Job name
#SBATCH -J 'nplim8wos'
### output and error logs
#SBATCH -o nplim8wos_%a.out
#SBATCH -e nplim8wos_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
module load python/2.7.14-fasrc01
source activate pro
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/runs/param_file.py \
--objid="${SLURM_ARRAY_TASK_ID}" \
--outfile="halo7d_nonparametric_limited_8_wos" \
--init_run_file='/n/conroyfs1/stacchella/halo7d_co//results/param_init/posterior_draws/summary_param_init_run.pkl' \
--path_files_init_run='/n/conroyfs1/stacchella/halo7d_co/results/param_init/' \
--non_param_sfh \
--n_bins_sfh=8 \
--restrict_dust_agn \
--restrict_prior \
--err_floor_phot=0.05 \
--err_floor_spec=0.01 \
--S2N_cut=5.0 \
--add_duste \
--add_agn \
--switch_off_spec \
--dynesty \
--nested_method="rwalk" \
--nlive_batch=100 \
--nlive_init=100 \
--nested_posterior_thresh=0.05 \
--nested_dlogz_init=0.05 \
--nested_maxcall=3000000 
