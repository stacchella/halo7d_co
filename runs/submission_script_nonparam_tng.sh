#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p conroy,itc_cluster,hernquist,shared,conroy_priority
### memory per cpu, in MB
#SBATCH --mem-per-cpu=10000
### constraints
#SBATCH --constraint=intel
### Job name
#SBATCH -J 'tng'
### output and error logs
#SBATCH -o tng_%a.out
#SBATCH -e tng_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
module load python/2.7.14-fasrc01
source activate pro
srun -n 1 python $DIR_CONROY/halo7d_co/runs/halo7d_param_file_tng.py \
--index_galaxy="${SLURM_ARRAY_TASK_ID}" \
--outfile="halo7d_nonparametric_tng" \
--zred=0.7 \
--add_noise \
--draw_snr \
--snr_spec=14.0 \
--snr_phot=20.0 \
--draw_params \
--non_param_sfh \
--n_bins_sfh=10 \
--add_neb \
--fit_continuum \
--add_duste \
--add_agn \
--add_jitter
