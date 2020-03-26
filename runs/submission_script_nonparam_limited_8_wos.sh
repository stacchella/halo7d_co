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
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/runs/halo7d_param_file.py \
--objid="${SLURM_ARRAY_TASK_ID}" \
--outfile="halo7d_nonparametric_limited_8_wos" \
--non_param_sfh \
--n_bins_sfh=8 \
--err_floor_phot=0.05 \
--err_floor_spec=0.01 \
--S2N_cut=5.0 \
--add_duste \
--add_agn \
--switch_off_spec \
