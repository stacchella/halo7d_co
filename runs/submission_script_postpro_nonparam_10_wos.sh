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
#SBATCH -J 'pp_10wos'
### output and error logs
#SBATCH -o pp_10wos_%a.out
#SBATCH -e pp_10wos_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
module load python/2.7.14-fasrc01
source activate pro
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/scripts/draw_posterior_output_cluster.py \
--number_of_bins=200 \
--idx_file_key="${SLURM_ARRAY_TASK_ID}" \
--path_results="/n/conroyfs1/stacchella/halo7d_co/results/nonparam_10_wos/" \
--ncalc=1000 \
--non_param_sfh \
--n_bins_sfh=10 \
--switch_off_spec \
--add_duste \
--add_agn \
--restrict_dust_agn \
--restrict_prior
