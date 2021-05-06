#! /bin/bash

## Reserved Memory and CPU slots
## (Specify )
#$ -l h_vmem=4G
#$ -pe smp 2

## Logfile configuration
#$ -j y
#$ -o /data/scc/veit-lorenz.heuthe/Logs/primitive_reward_test.$TASK_ID.log

## Send email on abort
#$ -m a
#$ -M veit-lorenz.heuthe@uni-konstanz.de

## Queue
## (don't use old, our libaries are compiled with modern math instructions)
#$ -q scc,long

## Exclude node scc131 (to old for math instruction set)
#$ -l h='!=scc131'

make -C fortran

python3 <<ENDOFPYTHON

# --- Impose Thread limitations to tensorflow ---

import tensorflow
tensorflow.config.threading.set_inter_op_parallelism_threads(2)
tensorflow.config.threading.set_intra_op_parallelism_threads(2)

# --- Parse parameters and start simulation ---

import os

task_id = int($SGE_TASK_ID) - 1

rewCutoff_range = [6, 8, 10, 12, 14, 16, 18, 20]
rewCutoff = rewCutoff_range[task_id]

job_dir = '/data/scc/veit-lorenz.heuthe/$JOB_NAME/rewCutoff_{}'.format(rewCutoff)
os.system("mkdir -p {}".format(job_dir))

os.system("python3 learning_ROD_reworked.py {} {}".format(rewCutoff, job_dir))

ENDOFPYTHON