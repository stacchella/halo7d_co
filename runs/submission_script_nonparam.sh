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
#SBATCH --mem-per-cpu=4000
### constraints
#SBATCH --constraint=intel
### Job name
#SBATCH -J 'nonhalo7d'
### output and error logs
#SBATCH -o nonhalo7d_%a.out
#SBATCH -e nonhalo7d_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
source activate pro
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/runs/param_file.py \
--objid="${SLURM_ARRAY_TASK_ID}" \
--outfile="halo7d_nonparametric" \
--non_param_sfh \
--err_floor_phot=0.05 \
--err_floor_spec=0.05 \
--f_boost=10.0 \
--S2N_cut=5.0 \
--fit_continuum \
--add_duste \
--add_agn \
--add_jitter \
--dynesty \
--nested_method="rwalk" \
--nlive_batch=200 \
--nlive_init=200 \
--nested_posterior_thresh=0.05 \
--nested_dlogz_init=0.05 \
--nested_maxcall=3000000 
