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
### Job name
#SBATCH -J 'poprparaminit'
### output and error logs
#SBATCH -o poprparaminit_%a.out
#SBATCH -e poprparaminit_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
module load python/2.7.14-fasrc01
source activate pro
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/scripts/draw_posterior_output_cluster.py \
--number_of_bins=200 \
--idx_file_key="${SLURM_ARRAY_TASK_ID}" \
--path_results="/n/conroyfs1/stacchella/halo7d_co/results/param_init/" \
--ncalc=1000 \
--add_jitter \
--fit_continuum \
--add_duste \
--add_agn
