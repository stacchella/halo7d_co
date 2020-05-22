#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 20080
### Partition or queue name
#SBATCH -p conroy
### memory per cpu, in MB
#SBATCH --mem-per-cpu=10000
### constraints
#SBATCH --constraint=intel
### Job name
#SBATCH -J 'np10wop'
### output and error logs
#SBATCH -o np10wop_%a.out
#SBATCH -e np10wop_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
module load python/2.7.14-fasrc01
source activate pro
srun -n 1 python $DIR_CONROY/halo7d_co/runs/halo7d_param_file.py \
--objid="${SLURM_ARRAY_TASK_ID}" \
--outfile="halo7d_nonparametric_10_wop" \
--non_param_sfh \
--n_bins_sfh=10 \
--err_floor_phot=0.05 \
--err_floor_spec=0.01 \
--S2N_cut=5.0 \
--switch_off_phot \
--add_neb \
--fit_continuum \
--add_duste \
--add_agn \
--add_jitter \
--add_lsf \
