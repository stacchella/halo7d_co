#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 1080
### Partition or queue name
#SBATCH -p conroy,itc_cluster,hernquist,shared
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'ppelprior_10'
### output and error logs
#SBATCH -o ppelprior_10_%a.out
#SBATCH -e ppelprior_10_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
module load python/2.7.14-fasrc01
source activate pro
srun -n 1 python $DIR_CONROY/halo7d_co/scripts/draw_posterior_output_cluster.py \
--number_of_bins=200 \
--idx_file_key="${SLURM_ARRAY_TASK_ID}" \
--path_results="nonparam_10_elprior/" \
--ncalc=1000 \
--non_param_sfh \
--n_bins_sfh=10 \
--add_jitter \
--add_neb \
--fit_continuum \
--add_duste \
--add_agn \
--use_eline_prior \
