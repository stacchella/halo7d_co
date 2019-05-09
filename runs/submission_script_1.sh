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
#SBATCH -J 'halo7d_full'
### output and error logs
#SBATCH -o full_%a.out
#SBATCH -e full_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
source activate pro
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/runs/param_file_parametric_eline.py \
--objid="${SLURM_ARRAY_TASK_ID}" \
--outfile="parametric_withEL_full" \
--err_floor=0.05 \
--S2N_cut=5.0 \
--fit_continuum \
--add_duste \
--add_agn \
--dynesty \
--nested_method="rwalk" \
--nlive_batch=1000 \
--nlive_init=1000 \
--nested_dlogz_init=0.1 \
--nested_maxcall=10000 
