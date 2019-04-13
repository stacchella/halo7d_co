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
#SBATCH -J 'halo7d_simple'
### output and error logs
#SBATCH -o halo7d_simple_we_%a.out
#SBATCH -e halo7d_simple_we_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
source activate pro
srun -n 1 python /n/conroyfs1/stacchella/halo7d_co/runs/param_file_parametric.py \
--objid="${SLURM_ARRAY_TASK_ID}" \
--outfile="parametric_simple_we_" \
--err_floor=0.1 \
--remove_mips24 \
--fit_continuum \
--add_neb \
--dynesty \
--nested_method="rwalk" \
--nlive_batch=200 \
--nlive_init=200 \
--nested_dlogz_init=0.5 \
--nested_maxcall=2000 
