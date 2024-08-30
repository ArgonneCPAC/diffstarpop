#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# # queue name (compute is the default)
# #PBS -q compute

# allocate 1 nodes, each with 1 MPI processes
#PBS -l select=1:ncpus=1:mpiprocs=1

#PBS -l walltime=02:00:00

# uncomment to pass in full environment
# #PBS -V

# Load software
source ~/.bash_profile
conda activate diffhacc_py311
cd /home/ahearin/work/random/0830
python measure_smhm_smdpl_script.py