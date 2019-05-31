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
#SBATCH -J 'v2halo7d'
### output and error logs
#SBATCH -o v2halo7d_%a.out
#SBATCH -e v2halo7d_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
source activate pro
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/runs/param_file_parametric_eline_marg.py \
--objid="${SLURM_ARRAY_TASK_ID}" \
--outfile="halo7d_v2_parametric" \
--err_floor_phot=0.05 \
--err_floor_spec=0.1 \
--S2N_cut=5.0 \
--fit_continuum \
--add_duste \
--dynesty \
--nested_method="rwalk" \
--nlive_batch=100 \
--nlive_init=100 \
--nested_posterior_thresh=0.05 \
--nested_dlogz_init=0.05 \
--nested_maxcall=10000000 
